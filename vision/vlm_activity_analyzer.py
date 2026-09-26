"""
VLM-Based Human Activity Recognition for Shelby Surveillance.

This module provides activity understanding of tracked people using OpenAI's
vision-language model. Activities are analyzed from temporal frame sequences,
providing higher-level semantic understanding beyond pose keypoints.

Architecture:
- Per track ID, collect a sliding window of frames at configurable interval
- When interval is reached, invoke VLM with sampled frames + prompt
- Return structured activity result (activity, description, confidence, timestamp)
- Handle API failures gracefully without disrupting surveillance

Not blocking: VLM calls happen asynchronously or deferred; main loop is unaffected.
"""

import logging
import json
import cv2
import base64
import io
import numpy as np
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import datetime
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait

import config

logger = logging.getLogger(__name__)


@dataclass
class ActivityResult:
    """Structured output from VLM activity analysis."""
    track_id: int
    activity: str  # primary activity label (e.g., "walking", "bending", "picking_up_object")
    description: str  # natural language description from VLM
    confidence: float  # 0.0-1.0, extracted from VLM response
    timestamp: str  # ISO format
    frame_ids: List[int] = None  # frame indices sampled for analysis


class ActivityFrameBuffer:
    """
    Maintains a sliding window of frames for each tracked person.
    
    When analysis is triggered, samples frames at regular intervals.
    This allows temporal context without storing every frame.
    """

    def __init__(self, window_size: int = 30, sample_rate: int = 3):
        """
        Args:
            window_size: Maximum frames to keep in the buffer
            sample_rate: Sample every Nth frame for VLM (reduces data)
        """
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.buffers: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                "frames": [],
                "frame_ids": [],
                "last_analysis": None,
                "last_analysis_frame_id": 0,
            }
        )

    @staticmethod
    def _crop_person(frame: np.ndarray, bbox, pad_frac: float = 0.25) -> np.ndarray:
        """
        Crop the tracked person out of the frame, with context padding.

        Why this matters: the prompt names a track id, but a full frame carries
        nothing that tells the model WHICH person that id refers to. With more
        than one person in shot the answer is unattributable. Cropping also cuts
        the payload by ~30x, which is what actually bounds how often we can
        afford to re-analyse (see the staleness note in VLMActivityAnalyzer).
        """
        if bbox is None:
            return frame.copy()
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            return frame.copy()
        # Pad so posture is readable in context (a tight crop of a seated person
        # loses the chair, which is most of the evidence for "sitting").
        px, py = bw * pad_frac, bh * pad_frac
        x1 = int(max(0, x1 - px));  y1 = int(max(0, y1 - py))
        x2 = int(min(w, x2 + px));  y2 = int(min(h, y2 + py))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return frame.copy()
        return frame[y1:y2, x1:x2].copy()

    def add_frame(self, track_id: int, frame: np.ndarray, frame_id: int, bbox=None):
        """Add a frame for a tracked person. If bbox is given, store the crop."""
        buf = self.buffers[track_id]
        crop = self._crop_person(frame, bbox)
        # Store the same resolution that will be sent, not 30 full-size crops.
        max_side = getattr(config, "VLM_ACTIVITY_MAX_IMAGE_SIDE", 512)
        h, w = crop.shape[:2]
        if max_side and max(h, w) > max_side:
            scale = max_side / max(h, w)
            crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))),
                              interpolation=cv2.INTER_AREA)
        buf["frames"].append(crop)
        buf["frame_ids"].append(frame_id)

        # Keep buffer size bounded
        if len(buf["frames"]) > self.window_size:
            buf["frames"].pop(0)
            buf["frame_ids"].pop(0)

    def get_sample_for_analysis(self, track_id: int) -> Optional[Dict[str, Any]]:
        """
        Sample frames for VLM analysis.
        
        Returns structured dict with sampled frames + indices, or None if insufficient.
        """
        buf = self.buffers.get(track_id)
        if not buf or len(buf["frames"]) < 3:
            return None

        frames = buf["frames"]
        frame_ids = buf["frame_ids"]

        # Sample at regular intervals, but always include first and last
        sampled_frames = []
        sampled_ids = []

        # Always include first
        sampled_frames.append(frames[0])
        sampled_ids.append(frame_ids[0])

        # Sample intermediate frames at sample_rate
        for i in range(self.sample_rate, len(frames) - 1, self.sample_rate):
            sampled_frames.append(frames[i])
            sampled_ids.append(frame_ids[i])

        # Always include last
        if len(frames) > 1:
            sampled_frames.append(frames[-1])
            sampled_ids.append(frame_ids[-1])

        # Cap how many images a single call carries. Token cost is linear in
        # image count (measured: 5 images at detail=low = 14,484 prompt tokens,
        # 11 images = 28,649), and the OpenAI account has a tokens-per-minute
        # budget shared with the RAG analyst. Keep first and last, subsample
        # evenly in between so the temporal span is preserved.
        max_images = getattr(config, "VLM_ACTIVITY_MAX_IMAGES", 0)
        if max_images and len(sampled_frames) > max_images:
            n = len(sampled_frames)
            keep = sorted({round(i * (n - 1) / (max_images - 1))
                           for i in range(max_images)}) if max_images > 1 else [n - 1]
            sampled_frames = [sampled_frames[i] for i in keep]
            sampled_ids = [sampled_ids[i] for i in keep]

        return {
            "frames": sampled_frames,
            "frame_ids": sampled_ids,
            "original_buffer_size": len(frames),
        }

    def mark_analyzed(self, track_id: int, frame_id: int):
        """Mark that this track has been analyzed at this frame."""
        if track_id in self.buffers:
            self.buffers[track_id]["last_analysis_frame_id"] = frame_id

    def should_analyze(self, track_id: int, current_frame_id: int, interval: int) -> bool:
        """
        Determine if this track is ready for analysis.
        
        Ready if:
        - Current frame_id is a multiple of interval
        - Enough frames have been collected since last analysis
        """
        buf = self.buffers.get(track_id)
        if not buf:
            return False

        last_analyzed = buf["last_analysis_frame_id"]
        frames_since = current_frame_id - last_analyzed

        return frames_since >= interval and current_frame_id % interval == 0

    def cleanup_track(self, track_id: int):
        """Remove track buffer (called when track dies)."""
        if track_id in self.buffers:
            del self.buffers[track_id]


class VLMActivityAnalyzer:
    """
    Analyzes tracked person activities using OpenAI vision API.
    
    Responsibilities:
    - Frame collection and sampling
    - VLM API calls with structured prompts
    - Response parsing and structure validation
    - Error handling and graceful degradation
    - Activity result caching per track
    """

    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: Whether VLM activity analysis is active
        """
        self.enabled = bool(enabled and getattr(
            config, "USE_VLM", getattr(config, "VLM_ACTIVITY_ENABLED", True)
        ))
        self.frame_buffer = ActivityFrameBuffer(
            window_size=getattr(config, "VLM_ACTIVITY_WINDOW_SIZE", 30),
            sample_rate=getattr(config, "VLM_ACTIVITY_SAMPLE_RATE", 3),
        )

        self.analysis_interval = getattr(
            config, "VLM_ACTIVITY_ANALYSIS_INTERVAL", 30
        )
        self.api_timeout = getattr(config, "VLM_ACTIVITY_API_TIMEOUT", 10.0)
        self.model_name = getattr(config, "VLM_MODEL_NAME", "gpt-4o-mini")

        # Cache recent activities to avoid repeated analysis
        self.activity_cache: Dict[int, ActivityResult] = {}
        # RLock, not Lock, and this is load-bearing: analyze_if_ready registers
        # the done-callback while holding this lock, and Future.add_done_callback
        # runs the callback IN THE CALLING THREAD if the future has already
        # finished. _complete_analysis then re-acquires the same lock. With a
        # plain Lock that is a self-deadlock which hangs the surveillance loop,
        # not just the analysis -- it fires whenever the worker completes fast,
        # e.g. an instant exception with no network or an immediate HTTP 429.
        self._lock = threading.RLock()
        self._pending_futures: Dict[int, Future] = {}
        self._executor = None
        self._generation = 0
        self._closed = False
        self._last_seen = {}

        # --- TOKENS-PER-MINUTE THROTTLE -------------------------------------
        # The frame-based interval alone cannot bound API spend: it counts SOURCE
        # frames, so its real-time meaning changes with clip frame rate (30 src
        # frames is 1 s at 30 fps but 0.5 s at 60 fps), and it says nothing about
        # how fast we are actually processing. This is a wall-clock floor across
        # ALL tracks, which is what the account's tokens-per-minute limit is.
        # Measured: 5 images at detail=low = 14,484 prompt tokens/call, so at a
        # 5 s floor we spend ~174k tokens/min against a 200k limit -- and that
        # budget is shared with the RAG analyst's embedding and chat calls.
        self.min_seconds_between_calls = getattr(
            config, "VLM_ACTIVITY_MIN_SECONDS_BETWEEN_CALLS", 5.0
        )
        self._last_call_time = 0.0
        self._backoff_until = 0.0
        self._rate_limited_count = 0

        if self.enabled:
            # Lazy import OpenAI; only if activity analysis is enabled
            try:
                from openai import OpenAI

                # Few SDK-level retries: on a 429 the SDK sleeps inside our
                # worker thread, which holds the per-track in-flight slot and
                # blocks the next analysis. We do our own global backoff instead.
                self.client = OpenAI(
                    api_key=config.OPENAI_API_KEY,
                    max_retries=getattr(config, "VLM_ACTIVITY_API_MAX_RETRIES", 1),
                )
                self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vlm-activity")
                logger.info("VLMActivityAnalyzer initialized (enabled)")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client for VLM: {e}")
                self.enabled = False
                self.client = None
        else:
            self.client = None

    def add_frame(self, track_id: int, frame: np.ndarray, frame_id: int, bbox=None):
        """
        Collect a frame for a tracked person.

        bbox (xyxy, full-frame coords) is optional but strongly preferred: with
        it the buffer stores a padded crop of this person instead of the whole
        scene. Without it the model cannot tell which person the prompt's track
        id refers to once more than one person is in shot.
        """
        with self._lock:
            if not self.enabled or self._closed:
                return
            self._last_seen[track_id] = frame_id
            self.frame_buffer.add_frame(track_id, frame, frame_id, bbox)

    def analyze_if_ready(
        self, track_id: int, current_frame_id: int
    ) -> Optional[ActivityResult]:
        """
        Check if this track is ready for analysis and invoke VLM if so.
        
        Returns:
            ActivityResult if analysis completed, None otherwise.
        """
        if not self.enabled or not self.client or self._closed:
            return None

        if not self.frame_buffer.should_analyze(
            track_id, current_frame_id, self.analysis_interval
        ):
            return None

        # A second freshness gate used to live here:
        #     if current_frame_id - cached.frame_ids[-1] < interval * 2: return cached
        # It DOUBLED the effective interval, because should_analyze had already
        # rate-limited on the same clock. Measured on sitting.mp4 (10.1 s, 60 fps):
        # only 3 analyses ran for the whole clip, each covering under 1 s of video,
        # leaving 4 s unanalysed between calls. That is what made a seated person
        # keep the "walking" label from several seconds earlier. Removed.

        # Submit the network call so the surveillance loop never waits for it.
        with self._lock:
            if self._closed:
                return None
            if track_id in self._pending_futures:
                return self.activity_cache.get(track_id)

            # Wall-clock throttle, global across tracks. Without it the analysis
            # rate is set by clip frame rate and machine speed, neither of which
            # knows anything about the account's tokens-per-minute limit.
            now = time.monotonic()
            if now < self._backoff_until:
                return self.activity_cache.get(track_id)
            if now - self._last_call_time < self.min_seconds_between_calls:
                return self.activity_cache.get(track_id)

            # Copy/sample only when a request can actually be submitted.
            sample = self.frame_buffer.get_sample_for_analysis(track_id)
            if not sample:
                return None
            min_age = getattr(config, "VLM_ACTIVITY_MIN_TRACK_AGE", 0)
            if sample.get("original_buffer_size", min_age) < min_age:
                return self.activity_cache.get(track_id)
            self._last_call_time = now

            future = self._executor.submit(
                self._invoke_vlm,
                track_id,
                sample["frames"],
                sample["frame_ids"],
            )
            self._pending_futures[track_id] = future
            generation = self._generation
            future.add_done_callback(
                lambda completed: self._complete_analysis(
                    track_id, current_frame_id, completed, generation
                )
            )

        return self.activity_cache.get(track_id)

    def _complete_analysis(
        self, track_id: int, frame_id: int, future: Future, generation: int
    ):
        """Store a completed result without propagating worker errors."""
        if future.cancelled():
            return
        try:
            result = future.result()
            with self._lock:
                if generation != self._generation or self._pending_futures.get(track_id) is not future:
                    return
                if result:
                    self.activity_cache[track_id] = result
        except Exception as e:
            logger.warning(f"VLM activity analysis failed for track {track_id}: {e}")
        finally:
            with self._lock:
                if generation == self._generation and self._pending_futures.get(track_id) is future:
                    self._pending_futures.pop(track_id, None)
                    self.frame_buffer.mark_analyzed(track_id, frame_id)

    def _invoke_vlm(
        self, track_id: int, frames: List[np.ndarray], frame_ids: List[int]
    ) -> Optional[ActivityResult]:
        """
        Invoke OpenAI vision API with sampled frames.
        
        Args:
            track_id: BotSort track ID
            frames: List of sampled BGR frames
            frame_ids: Frame IDs corresponding to each frame
            
        Returns:
            Structured ActivityResult or None on failure
        """
        if not self.client or len(frames) == 0:
            return None

        try:
            # Encode frames as base64 JPEG.
            #
            # Two things were wrong here and both were measured:
            #  1. cv2.imencode already expects BGR and writes correct RGB into
            #     the file. Converting BGR->RGB first swapped red and blue in
            #     every image we ever sent (centre patch BGR [171,190,201] came
            #     out as [201,190,171]). Pass the BGR frame straight in.
            #  2. Lossless PNG at source resolution sent ~4 MB per call. The API
            #     downsamples to tiles regardless, so that was pure upload
            #     latency -- and latency is what bounds how fresh the label can
            #     be, since only one analysis per track is in flight at a time.
            max_side = getattr(config, "VLM_ACTIVITY_MAX_IMAGE_SIDE", 512)
            quality = getattr(config, "VLM_ACTIVITY_JPEG_QUALITY", 85)
            image_content = []
            for frame in frames:
                h, w = frame.shape[:2]
                if max_side and max(h, w) > max_side:
                    s = max_side / max(h, w)
                    frame = cv2.resize(
                        frame, (max(1, int(w * s)), max(1, int(h * s))),
                        interpolation=cv2.INTER_AREA,
                    )
                ok, buffer = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
                )
                if not ok:
                    continue
                b64_str = base64.b64encode(buffer).decode("utf-8")
                url = {"url": f"data:image/jpeg;base64,{b64_str}"}
                # detail="low" bills a flat ~2833 prompt tokens per image instead
                # of tiling it. Measured on the same 11 crops: 85,319 prompt
                # tokens without it vs 28,649 with it -- a 3x cut -- and the
                # verdict was identical ("sitting") at every setting tested.
                # Without this the run blew the 200k tokens/min account limit
                # and calls started failing with HTTP 429.
                detail = getattr(config, "VLM_ACTIVITY_IMAGE_DETAIL", "low")
                if detail:
                    url["detail"] = detail
                image_content.append({"type": "image_url", "image_url": url})
            if not image_content:
                return None

            # Add prompt
            prompt_text = self._build_activity_prompt(track_id, len(frames))
            image_content.append({"type": "text", "text": prompt_text})

            # Call OpenAI vision API
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=500,
                messages=[{"role": "user", "content": image_content}],
                response_format={"type": "json_object"},
                timeout=self.api_timeout,
            )

            # A call landed, so whatever backoff we were in has served its
            # purpose; decay it rather than staying pessimistic for the run.
            with self._lock:
                self._rate_limited_count = max(0, self._rate_limited_count - 1)

            # Parse response
            result = self._parse_vlm_response(response, track_id, frame_ids)
            return result

        except Exception as e:
            # A 429 is not a bug, it is the account's tokens-per-minute limit
            # telling us we are asking too often. Back off globally so we stop
            # burning worker threads on requests that cannot succeed, and let
            # the floor decay back down once calls start landing again.
            if "429" in str(e) or "rate_limit" in str(e).lower():
                with self._lock:
                    self._rate_limited_count += 1
                    cap = getattr(config, "VLM_ACTIVITY_MAX_BACKOFF_SECONDS", 60.0)
                    wait = min(cap, self.min_seconds_between_calls * (2 ** min(self._rate_limited_count, 4)))
                    self._backoff_until = time.monotonic() + wait
                logger.warning(
                    f"VLM rate-limited (429) for track {track_id}; "
                    f"pausing activity analysis for {wait:.0f}s "
                    f"(occurrence {self._rate_limited_count}). "
                    f"Raise VLM_ACTIVITY_MIN_SECONDS_BETWEEN_CALLS to stop this recurring."
                )
                return None
            logger.error(f"VLM API call failed for track {track_id}: {e}")
            return None

    def _build_activity_prompt(self, track_id: int, num_frames: int) -> str:
        """
        Build the activity recognition prompt for the VLM.
        
        Instructs the model to analyze human activity from provided frames.
        """
        prompt = f"""Analyze the activity of the tracked person (Track ID {track_id}) shown in these {num_frames} temporal frames.

INSTRUCTIONS:
1. Carefully observe the sequence of frames and identify the person's primary activity.
2. Focus on observable actions and movements only. Do NOT invent or assume actions.
3. The frames are sampled from a sliding window, so there may be gaps in time.
   These are padded crops of one target person. Focus on the central person
   consistently present across frames, not another person entering the crop.
   If the target is ambiguous or occluded, return unknown with low confidence.
   Distinguish visible posture (standing/sitting) from locomotion (walking/running).
   Do not infer conversation, intent, or device operation from proximity alone.
4. Return your analysis as a JSON object with the following structure:
{{
    "activity": "<activity_label>",
    "description": "<natural_language_description>",
    "confidence": <0.0_to_1.0>,
    "reasoning": "<brief_explanation_of_your_analysis>"
}}

ACTIVITY LABELS (examples, but not exhaustive):
- walking
- standing
- sitting
- running
- bending
- picking_up_object
- carrying_object
- putting_down_object
- reaching
- interacting_with_object
- interacting_with_person
- entering_area
- leaving_area
- waiting
- loitering
- looking_at_object
- unknown

CRITICAL RULES:
- Return ONLY the JSON object. Do NOT add any other text.
- If activity is unclear or insufficient evidence, use "unknown" with low confidence.
- Confidence should reflect the clarity of the evidence (1.0 = very clear, 0.0 = no evidence).
- Description should be a single sentence, human-readable explanation.
- NEVER make security judgments (e.g., "this is theft" or "suspicious"). Just describe what is observable.
"""
        return prompt

    def _parse_vlm_response(
        self, response, track_id: int, frame_ids: List[int]
    ) -> Optional[ActivityResult]:
        """
        Parse VLM response and extract structured activity.
        
        Handles malformed responses gracefully.
        """
        try:
            # Extract text from response
            response_text = (
                response.choices[0].message.content.strip()
                if response.choices and response.choices[0].message.content
                else ""
            )
            if response_text.startswith("```json") and response_text.endswith("```"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```") and response_text.endswith("```"):
                response_text = response_text[3:-3].strip()

            # Parse JSON
            activity_data = json.loads(response_text)

            # Validate required fields
            activity = activity_data.get("activity", "unknown")
            description = activity_data.get("description", "Activity unclear")
            confidence = float(activity_data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

            result = ActivityResult(
                track_id=track_id,
                activity=activity,
                description=description,
                confidence=confidence,
                timestamp=datetime.datetime.now().isoformat(),
                frame_ids=frame_ids,
            )

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse VLM JSON for track {track_id}: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"VLM response validation failed for track {track_id}: {e}")
            return None

    def get_activity(
        self, track_id: int, current_frame_id: Optional[int] = None
    ) -> Optional[ActivityResult]:
        """
        Retrieve the cached activity result for a track.

        If current_frame_id is supplied, a result older than
        VLM_ACTIVITY_LABEL_TTL source frames is treated as expired and None is
        returned. Without this a verdict stayed on screen forever: the label a
        seated person was showing had been computed seconds earlier while they
        were still walking to the chair. A stale label is worse than no label,
        because it reads as a current observation.
        """
        with self._lock:
            result = self.activity_cache.get(track_id)
            if result is None or current_frame_id is None:
                return result
            ttl = getattr(config, "VLM_ACTIVITY_LABEL_TTL", 0)
            if ttl and result.frame_ids:
                if current_frame_id - result.frame_ids[-1] > ttl:
                    return None
            return result

    def cleanup_track(self, track_id: int):
        """Remove track from buffers and cache."""
        with self._lock:
            self.frame_buffer.cleanup_track(track_id)
            self._last_seen.pop(track_id, None)
            self.activity_cache.pop(track_id, None)
            pending = self._pending_futures.pop(track_id, None)
            if pending:
                pending.cancel()

    def reset(self):
        """Start a new run; callbacks from the old run cannot repopulate it."""
        with self._lock:
            self._generation += 1
            self.frame_buffer.buffers.clear()
            self._last_seen.clear()
            self.activity_cache.clear()
            pending = list(self._pending_futures.values())
            self._pending_futures.clear()
            for future in pending:
                future.cancel()
            if self.enabled and self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vlm-activity")
            self._closed = False

    def cleanup_inactive(self, current_frame_id: int):
        """Bound crop memory while preserving short tracker occlusions."""
        ttl = max(1, int(getattr(config, "VLM_INACTIVE_TRACK_TTL", 300)))
        with self._lock:
            for track_id, last_frame in list(self._last_seen.items()):
                if current_frame_id - last_frame > ttl:
                    self.cleanup_track(track_id)

    def shutdown(self, timeout: float = 0.0) -> int:
        """Stop submissions and drain for at most timeout seconds.

        Running HTTP calls cannot be forcibly cancelled. After the deadline their
        results are invalidated; SDK timeouts still bound the network operation.
        Returns the number of requests that did not finish within the deadline.
        reset() can reopen the analyzer for a later run.
        """
        with self._lock:
            self._closed = True
            pending = list(self._pending_futures.values())
        if pending:
            wait(pending, timeout=max(0.0, timeout))
        with self._lock:
            unfinished = sum(not f.done() for f in pending)
            # Future completion wakes waiters before its callbacks necessarily
            # finish. Preserve completed results before invalidating callbacks.
            for track_id, future in self._pending_futures.items():
                if future.done() and not future.cancelled():
                    try:
                        result = future.result()
                        if result:
                            self.activity_cache[track_id] = result
                    except Exception:
                        pass  # The completion callback reports worker errors.
            self._generation += 1
            self._pending_futures.clear()
            self.frame_buffer.buffers.clear()
            self._last_seen.clear()
            for future in pending:
                future.cancel()
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        return unfinished
