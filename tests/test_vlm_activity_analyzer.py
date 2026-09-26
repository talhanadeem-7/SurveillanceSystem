import threading
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import numpy as np

from vision.vlm_activity_analyzer import ActivityFrameBuffer, VLMActivityAnalyzer


class VLMActivityAnalyzerTest(unittest.TestCase):
    @staticmethod
    def _rate_limit_state(analyzer, min_seconds=0.0):
        """Attach the throttle/backoff state that __init__ would normally set."""
        analyzer._lock = threading.RLock()
        analyzer.min_seconds_between_calls = min_seconds
        analyzer._last_call_time = 0.0
        analyzer._backoff_until = 0.0
        analyzer._rate_limited_count = 0
        analyzer._generation = 0
        analyzer._closed = False
        analyzer._last_seen = {}

    def setUp(self):
        self.analyzer = VLMActivityAnalyzer.__new__(VLMActivityAnalyzer)
        self.analyzer.client = MagicMock()
        self.analyzer.model_name = "gpt-4o-mini"
        self.analyzer.api_timeout = 10.0
        self._rate_limit_state(self.analyzer)

    def test_openai_request_and_response_parsing(self):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = (
            '{"activity":"walking","description":"Person walks.","confidence":0.91}'
        )
        self.analyzer.client.chat.completions.create.return_value = response

        result = self.analyzer._invoke_vlm(
            7,
            [np.zeros((32, 32, 3), dtype=np.uint8)] * 3,
            [10, 11, 12],
        )

        request = self.analyzer.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "gpt-4o-mini")
        self.assertEqual(request["max_tokens"], 500)
        self.assertEqual(request["timeout"], 10.0)
        self.assertEqual(request["response_format"], {"type": "json_object"})
        content = request["messages"][0]["content"]
        self.assertEqual(content[-1]["type"], "text")
        self.assertIn("Track ID 7", content[-1]["text"])
        self.assertEqual([item["type"] for item in content[:-1]], ["image_url"] * 3)
        self.assertEqual(result.activity, "walking")
        self.assertEqual(result.confidence, 0.91)
        self.assertEqual(result.frame_ids, [10, 11, 12])

    def test_timeout_is_handled(self):
        self.analyzer.client.chat.completions.create.side_effect = TimeoutError("slow")

        result = self.analyzer._invoke_vlm(
            1,
            [np.zeros((16, 16, 3), dtype=np.uint8)] * 3,
            [1, 2, 3],
        )

        self.assertIsNone(result)

    def test_api_failure_is_handled(self):
        self.analyzer.client.chat.completions.create.side_effect = RuntimeError("offline")

        result = self.analyzer._invoke_vlm(
            1,
            [np.zeros((16, 16, 3), dtype=np.uint8)] * 3,
            [1, 2, 3],
        )

        self.assertIsNone(result)

    def test_analysis_is_submitted_to_background_worker(self):
        analyzer = VLMActivityAnalyzer.__new__(VLMActivityAnalyzer)
        analyzer.enabled = True
        analyzer.client = MagicMock()
        analyzer.analysis_interval = 3
        analyzer.model_name = "gpt-4o-mini"
        analyzer.api_timeout = 10.0
        analyzer.activity_cache = {}
        self._rate_limit_state(analyzer)
        analyzer._pending_futures = {}
        analyzer._executor = ThreadPoolExecutor(max_workers=1)
        analyzer.frame_buffer = MagicMock()
        analyzer.frame_buffer.should_analyze.return_value = True
        analyzer.frame_buffer.get_sample_for_analysis.return_value = {
            "frames": [np.zeros((16, 16, 3), dtype=np.uint8)] * 3,
            "frame_ids": [1, 2, 3],
        }
        analyzer._invoke_vlm = MagicMock(side_effect=lambda *args: (time.sleep(0.1), None)[1])

        started = time.perf_counter()
        analyzer.analyze_if_ready(4, 3)
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.08)
        self.assertIn(4, analyzer._pending_futures)
        analyzer._executor.shutdown(wait=True)


    def _throttle_analyzer(self, min_seconds):
        a = VLMActivityAnalyzer.__new__(VLMActivityAnalyzer)
        a.enabled = True
        a.client = MagicMock()
        a.analysis_interval = 1
        a.model_name = "gpt-4o-mini"
        a.api_timeout = 10.0
        a.activity_cache = {}
        a._pending_futures = {}
        a._executor = ThreadPoolExecutor(max_workers=1)
        self._rate_limit_state(a, min_seconds=min_seconds)
        a.frame_buffer = MagicMock()
        a.frame_buffer.should_analyze.return_value = True
        a.frame_buffer.get_sample_for_analysis.return_value = {
            "frames": [np.zeros((16, 16, 3), dtype=np.uint8)] * 3,
            "frame_ids": [1, 2, 3],
            "original_buffer_size": 30,
        }
        a._invoke_vlm = MagicMock(return_value=None)
        return a

    def test_wall_clock_throttle_blocks_rapid_calls(self):
        """The TPM guard: a second call inside the floor must not be submitted."""
        a = self._throttle_analyzer(min_seconds=60.0)
        a.analyze_if_ready(1, 10)
        a._executor.shutdown(wait=True)
        first = a._invoke_vlm.call_count
        a._executor = ThreadPoolExecutor(max_workers=1)
        a.analyze_if_ready(1, 20)          # immediately after -- inside the floor
        a._executor.shutdown(wait=True)
        self.assertEqual(first, 1)
        self.assertEqual(a._invoke_vlm.call_count, 1, "throttle did not block the second call")

    def test_throttle_allows_call_once_floor_has_passed(self):
        a = self._throttle_analyzer(min_seconds=0.0)
        a.analyze_if_ready(1, 10)
        a._executor.shutdown(wait=True)
        a._executor = ThreadPoolExecutor(max_workers=1)
        a.analyze_if_ready(1, 20)
        a._executor.shutdown(wait=True)
        self.assertEqual(a._invoke_vlm.call_count, 2)

    def test_rate_limit_error_triggers_backoff_and_returns_none(self):
        self.analyzer.min_seconds_between_calls = 5.0
        self.analyzer.client.chat.completions.create.side_effect = RuntimeError(
            "Error code: 429 - rate_limit_exceeded"
        )
        result = self.analyzer._invoke_vlm(
            1, [np.zeros((16, 16, 3), dtype=np.uint8)] * 3, [1, 2, 3]
        )
        self.assertIsNone(result)
        self.assertEqual(self.analyzer._rate_limited_count, 1)
        self.assertGreater(self.analyzer._backoff_until, time.monotonic())

    def test_images_are_capped_and_span_is_preserved(self):
        """Token cost is linear in image count, so the cap must hold."""
        buf = ActivityFrameBuffer(window_size=30, sample_rate=3)
        for i in range(30):
            buf.add_frame(1, np.zeros((8, 8, 3), dtype=np.uint8), i)
        sample = buf.get_sample_for_analysis(1)
        cap = __import__("config").VLM_ACTIVITY_MAX_IMAGES
        self.assertLessEqual(len(sample["frames"]), cap)
        self.assertEqual(len(sample["frames"]), len(sample["frame_ids"]))
        # first and last frame must survive subsampling or the window shrinks
        self.assertEqual(sample["frame_ids"][0], 0)
        self.assertEqual(sample["frame_ids"][-1], 29)

    def test_crop_uses_bbox_and_pads(self):
        buf = ActivityFrameBuffer(window_size=5, sample_rate=1)
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        buf.add_frame(1, frame, 1, bbox=[40, 80, 60, 120])
        stored = buf.buffers[1]["frames"][0]
        self.assertLess(stored.shape[0], frame.shape[0], "crop should be smaller than frame")
        self.assertGreater(stored.shape[0], 40, "crop should include padding context")

    def test_stale_label_expires(self):
        from vision.vlm_activity_analyzer import ActivityResult
        self.analyzer.activity_cache = {
            1: ActivityResult(1, "walking", "d", 0.9, "t", frame_ids=[100])
        }
        ttl = __import__("config").VLM_ACTIVITY_LABEL_TTL
        self.assertIsNotNone(self.analyzer.get_activity(1, 100 + ttl))
        self.assertIsNone(self.analyzer.get_activity(1, 100 + ttl + 1))
        self.assertIsNotNone(self.analyzer.get_activity(1), "no frame id => no expiry")

    def test_disabled_mode_creates_no_client_or_executor(self):
        with patch("config.USE_VLM", False), patch("openai.OpenAI") as client:
            analyzer = VLMActivityAnalyzer()
            analyzer.add_frame(1, np.zeros((10, 10, 3), np.uint8), 30)
            self.assertIsNone(analyzer.analyze_if_ready(1, 30))
            analyzer.reset()
            self.assertEqual(analyzer.shutdown(), 0)
            self.assertFalse(analyzer.enabled)
            self.assertIsNone(analyzer._executor)
            self.assertEqual(len(analyzer.frame_buffer.buffers), 0)
            client.assert_not_called()

    def test_old_callback_cannot_overwrite_new_run_or_pending_slot(self):
        from vision.vlm_activity_analyzer import ActivityResult
        a = self._throttle_analyzer(0)
        a.frame_buffer = ActivityFrameBuffer()
        old = Future()
        old.set_running_or_notify_cancel()
        a._pending_futures[1] = old
        generation = a._generation
        old.add_done_callback(lambda f: a._complete_analysis(1, 30, f, generation))
        a.reset()
        replacement = Future()
        a._pending_futures[1] = replacement
        old.set_result(ActivityResult(1, "walking", "old run", .9, "t", [30]))
        self.assertNotIn(1, a.activity_cache)
        self.assertIs(a._pending_futures[1], replacement)
        a.shutdown()

    def test_shutdown_is_bounded_and_ignores_late_results(self):
        from vision.vlm_activity_analyzer import ActivityResult
        a = self._throttle_analyzer(0)
        pending = Future()
        pending.set_running_or_notify_cancel()
        a._pending_futures[1] = pending
        generation = a._generation
        pending.add_done_callback(lambda f: a._complete_analysis(1, 30, f, generation))
        self.assertEqual(a.shutdown(timeout=0), 1)
        self.assertIsNone(a.analyze_if_ready(1, 60))
        pending.set_result(ActivityResult(1, "walking", "late", .9, "t", [30]))
        self.assertEqual(a.activity_cache, {})
        self.assertIsNone(a._executor)
        a.reset()
        self.assertFalse(a._closed)
        self.assertIsNotNone(a._executor)
        a.shutdown()

    def test_throttle_does_not_prepare_samples(self):
        a = self._throttle_analyzer(60)
        a._last_call_time = time.monotonic()
        a.analyze_if_ready(1, 30)
        a.frame_buffer.get_sample_for_analysis.assert_not_called()
        a.shutdown()

    def test_inactive_cleanup_ignores_late_result(self):
        from vision.vlm_activity_analyzer import ActivityResult
        a = self._throttle_analyzer(0)
        a.frame_buffer = ActivityFrameBuffer()
        a.add_frame(1, np.zeros((10, 10, 3), np.uint8), 1)
        old = Future()
        old.set_running_or_notify_cancel()
        a._pending_futures[1] = old
        generation = a._generation
        old.add_done_callback(lambda f: a._complete_analysis(1, 30, f, generation))
        a.cleanup_inactive(10000)
        old.set_result(ActivityResult(1, "walking", "expired", .9, "t", [30]))
        self.assertNotIn(1, a.activity_cache)
        self.assertNotIn(1, a.frame_buffer.buffers)
        a.shutdown()


if __name__ == "__main__":
    unittest.main()
