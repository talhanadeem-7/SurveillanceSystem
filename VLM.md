# VLM Activity Recognition — Complete Reference

Vision-Language Model activity recognition for the Shelby Surveillance System.
This is the **only** thing in the codebase that answers *"what is this person doing?"*.

**Status:** working and verified end-to-end (2026-09-01).
**Model:** `gpt-4o-mini` (vision) via the OpenAI API.
**Module:** [vision/vlm_activity_analyzer.py](vision/vlm_activity_analyzer.py)

> **Rule for this document:** every number here was **measured**, not inferred from
> reading code. Where something is unverified it is labelled as such. This mirrors the
> discipline in [CLAUDE.md](CLAUDE.md), which has been wrong before by trusting source.

---

## Table of Contents

1. [What it is and what it replaced](#1-what-it-is-and-what-it-replaced)
2. [When it comes into play](#2-when-it-comes-into-play)
3. [Architecture](#3-architecture)
4. [The full call path, step by step](#4-the-full-call-path-step-by-step)
5. [Configuration reference](#5-configuration-reference)
6. [Cost and rate control](#6-cost-and-rate-control)
7. [Output format](#7-output-format)
8. [Where results go](#8-where-results-go)
9. [Error handling and failure modes](#9-error-handling-and-failure-modes)
10. [Concurrency model](#10-concurrency-model)
11. [Bugs found and fixed](#11-bugs-found-and-fixed-with-evidence)
12. [Performance and measured numbers](#12-performance-and-measured-numbers)
13. [Tuning guide](#13-tuning-guide)
14. [Testing](#14-testing)
15. [Troubleshooting](#15-troubleshooting)
16. [Known limitations and future work](#16-known-limitations-and-future-work)

---

## 1. What it is and what it replaced

The VLM answers a question the rest of the pipeline cannot. YOLOv8n-pose gives keypoints,
BotSort gives a persistent identity, DeepFace gives a name, and the zone/depth logic gives
*intrusion* and *theft*. None of those describe **behaviour**.

### It replaced a pose-heuristic action recognizer

There used to be `vision/action_recognizer.py`: an ST-GCN / threshold-based classifier that
emitted verbs like `Walking`, `Sitting`, `Bending`, `Picking Up`, `Running` by measuring joint
velocities and body-fold ratios.

It was **deleted in commit `760a6c2`**. Its residue was removed later:

| Residue | Where | Status |
|---|---|---|
| `ACTION_MODEL_PATH`, `ACTION_WINDOW`, `RUNNING_VELOCITY_THRESHOLD`, `BENDING_RATIO_THRESHOLD`, `PICKUP_HAND_KNEE_RELATION` | `config.py` | removed |
| `"Behavior"` narration branch + `last_action_per_entity` state machine | `utils/csv_utils.py` | removed |
| `behavior_logged_ids`, `logged_general_ids`, `track_ids_logged_general` (written, never read) | `streamlit_app.py` | removed |
| "Walking, Sitting" cited as example log kinds | `agents/reasoning_agent.py` | updated |
| Same three dead sets | `app.py` | **left in place** — `app.py` is the out-of-scope legacy desktop variant |

**There was never a conflict between the two.** The recognizer was already gone before the VLM
was wired in; nothing was competing with or shadowing it. Verified by measurement:
`storage/event_logs.csv` contains **0** `Behavior` rows (Action column is Intrusion 195 /
Identity 67 / Access 37 / Theft 15 / Removal 4).

### Why a VLM instead of fixing the heuristics

| | Pose heuristics | VLM |
|---|---|---|
| Vocabulary | 6 hardcoded verbs | open-ended, 17 suggested labels + anything it observes |
| Object context | none — cannot see the chair | sees the chair, the bag, the laptop |
| Tuning | 3 magic thresholds per verb | a prompt |
| Failure mode | silently wrong | returns `unknown` with low confidence |
| Cost | free | ~14,500 prompt tokens per call |
| Latency | microseconds | ~1–4 s, **off the main thread** |

The trade is cost and latency for semantic understanding. The architecture exists to make that
trade survivable: the VLM **never blocks the surveillance loop**.

---

## 2. When it comes into play

The VLM does **not** run per frame. A call happens only when **all** of these hold:

| # | Gate | Where | Condition |
|---|---|---|---|
| 1 | Master switch | `config.USE_VLM` | must be `True` |
| 2 | Client built | `VLMActivityAnalyzer.__init__` | OpenAI client constructed successfully |
| 3 | Not ephemeral | `streamlit_app.py` | `current_id >= 0` — unmatched detections get negative ids and accumulate no state |
| 4 | Frame interval | `ActivityFrameBuffer.should_analyze` | `frames_since_last >= INTERVAL` **and** `frame_id % INTERVAL == 0` |
| 5 | Enough frames | `get_sample_for_analysis` | buffer holds ≥ 3 frames |
| 6 | Track old enough | `analyze_if_ready` | `buffer_size >= VLM_ACTIVITY_MIN_TRACK_AGE` |
| 7 | None in flight | `analyze_if_ready` | no pending future for this track id |
| 8 | Not backing off | `analyze_if_ready` | `now >= _backoff_until` (set by a prior HTTP 429) |
| 9 | **Wall-clock floor** | `analyze_if_ready` | `now - _last_call_time >= VLM_ACTIVITY_MIN_SECONDS_BETWEEN_CALLS` — **global across all tracks** |

Gate 9 is the one that actually bounds spend. Gates 4–6 are frame-based and therefore depend on
clip frame rate and machine speed; gate 9 does not.

### ⚠️ Gate 4 is frame-rate dependent

`VLM_ACTIVITY_ANALYSIS_INTERVAL` counts **source** frames. Its real-time meaning changes with the
clip:

| Source fps | `INTERVAL = 30` means |
|---|---|
| 30 fps | 1.0 s of video |
| 60 fps | **0.5 s of video** |

The config comment claiming "≈ 2 seconds" assumes 30 fps *and* `FRAME_SKIP=2`. On the 60 fps
`sitting.mp4` test clip the same value means half a second. **This is why gate 9 exists.**

---

## 3. Architecture

```
run_surveillance() main loop  [streamlit_app.py]
        │
        │  per processed frame, per non-ephemeral track
        ▼
  add_frame(track_id, frame, frame_id, bbox)
        │
        ▼
  ActivityFrameBuffer  ─── crops the person from the frame (padded)
        │                  keeps a sliding window per track_id
        │                  window_size=30, evicts oldest
        ▼
  analyze_if_ready(track_id, frame_id)
        │
        ├── gates 4-9 above ──► returns cached result, NO network call
        │
        └── passes ──► ThreadPoolExecutor.submit(_invoke_vlm)   ◄── NON-BLOCKING
                              │                                     returns immediately
                              │  (worker thread)
                              ▼
                       crop → resize 512px → JPEG q85 → base64
                       + detail="low"  + prompt
                              │
                              ▼
                       OpenAI chat.completions  (gpt-4o-mini, JSON mode)
                              │
                              ▼
                       _parse_vlm_response  ──► ActivityResult
                              │
                              ▼
                       _complete_analysis (done-callback)
                              │
                              ▼
                       activity_cache[track_id] = result
        │
        ▼
  get_activity(track_id, frame_id)   ◄── applies TTL; expired => None
        │
        ▼
  draw_surveillance_ui()  ──► "sitting (0.90)" beside the box
  sidebar status panel    ──► last 3 activities
```

### The two classes

**`ActivityFrameBuffer`** — pure, no network. Per-track sliding window of person crops, plus the
sampling and interval logic. Testable without an API key.

**`VLMActivityAnalyzer`** — owns the OpenAI client, the thread pool, the result cache, the
wall-clock throttle and the 429 backoff.

**`ActivityResult`** — the dataclass that crosses the boundary:

```python
@dataclass
class ActivityResult:
    track_id: int
    activity: str          # "sitting", "walking", "picking_up_object", ...
    description: str       # one human-readable sentence
    confidence: float      # 0.0-1.0, clamped
    timestamp: str         # ISO 8601, wall clock
    frame_ids: List[int]   # the frames this verdict was computed from
```

`frame_ids[-1]` is what the TTL is measured against. It is **the whole basis of freshness** — see
[§11](#11-bugs-found-and-fixed-with-evidence).

---

## 4. The full call path, step by step

### 4.1 Frame collection — `add_frame`

Called once per processed frame per non-ephemeral track, from `streamlit_app.py`:

```python
st.session_state.activity_analyzer.add_frame(current_id, frame, frame_id, bbox)
```

`bbox` is xyxy in **full-frame** coordinates. The buffer stores a **crop** of the person with
25% padding on each side, not the whole scene:

```python
px, py = bw * pad_frac, bh * pad_frac      # pad_frac = 0.25
x1 = max(0, x1 - px);  x2 = min(w, x2 + px)
y1 = max(0, y1 - py);  y2 = min(h, y2 + py)
```

**Why crop:** the prompt names a track id, but a full frame carries nothing telling the model
*which* person that id refers to. With two people in shot the answer is unattributable.

**Why pad:** a tight crop of a seated person loses the chair, which is most of the evidence for
"sitting".

If `bbox` is `None` the whole frame is stored — supported, but the attribution problem returns.

### 4.2 Sampling — `get_sample_for_analysis`

From a window of up to 30 crops:

1. Always keep the **first** frame.
2. Keep every `VLM_ACTIVITY_SAMPLE_RATE`-th frame (default 3).
3. Always keep the **last** frame.
4. **Cap at `VLM_ACTIVITY_MAX_IMAGES`** (default 5) by even subsampling that preserves first
   and last, so the temporal span survives the cut.

Returns `{"frames", "frame_ids", "original_buffer_size"}`.

### 4.3 Encoding — inside `_invoke_vlm`

Per image, in order:

1. **Downscale** so the longest side ≤ `VLM_ACTIVITY_MAX_IMAGE_SIDE` (512), `INTER_AREA`.
2. **JPEG encode** at `VLM_ACTIVITY_JPEG_QUALITY` (85).
3. **base64** into a `data:image/jpeg;base64,...` URL.
4. Attach **`detail: "low"`**.

⚠️ **Do not insert a `cv2.cvtColor(BGR2RGB)` here.** `cv2.imencode` already expects BGR and
writes correct RGB into the file. Converting first double-swaps red and blue — see
[§11.2](#112-colour-channels-were-double-swapped-in-every-image-ever-sent).

### 4.4 The prompt — `_build_activity_prompt`

Sent as the **last** content item, after the images. It asks for a JSON object and is called with
`response_format={"type": "json_object"}`, so the model is constrained to valid JSON.

Suggested labels (explicitly "not exhaustive"):

```
walking · standing · sitting · running · bending · picking_up_object
carrying_object · putting_down_object · reaching · interacting_with_object
interacting_with_person · entering_area · leaving_area · waiting
loitering · looking_at_object · unknown
```

Critical rules in the prompt:

- Return **only** the JSON object.
- Use `unknown` with low confidence if the evidence is unclear.
- Confidence reflects clarity of evidence.
- **Never make security judgments** — no "this is theft", no "suspicious". Describe only what is
  observable. Security classification is the RAG analyst's job, driven by zone and identity
  evidence, not by the VLM's impression.

### 4.5 Parsing — `_parse_vlm_response`

Strips ` ```json ` fences if present, `json.loads`, then:

- `activity` defaults to `"unknown"`
- `description` defaults to `"Activity unclear"`
- `confidence` is `float()`-cast and **clamped to [0.0, 1.0]**
- `JSONDecodeError` / `KeyError` / `ValueError` / `TypeError` → logged at WARNING, returns `None`

A `None` result is never cached, so a malformed response leaves the previous label in place until
its TTL expires.

---

## 5. Configuration reference

All keys live in [config.py](config.py) under `--- VLM ACTIVITY RECOGNITION ---` and
`--- VLM COST / RATE CONTROL ---`.

### Core

| Key | Default | Unit | What it does |
|---|---|---|---|
| `USE_VLM` | `True` | bool | Master switch. `False` = no client, worker, crop buffering, or API calls. |
| `VLM_MODEL_NAME` | `"gpt-4o-mini"` | str | Must be a vision-capable OpenAI model. |
| `VLM_ACTIVITY_ANALYSIS_INTERVAL` | `30` | **source frames** | Trigger cadence. ⚠️ frame-rate dependent — see [§2](#2-when-it-comes-into-play). |
| `VLM_ACTIVITY_WINDOW_SIZE` | `30` | processed frames | Sliding window depth per track. |
| `VLM_ACTIVITY_SAMPLE_RATE` | `3` | frames | Keep every Nth frame from the window. |
| `VLM_ACTIVITY_API_TIMEOUT` | `10.0` | seconds | Per-request timeout. |
| `VLM_ACTIVITY_MIN_TRACK_AGE` | `15` | frames | Don't judge a track barely seen. **Now actually read** — it was dead config. |

### Cost and rate control

| Key | Default | Effect |
|---|---|---|
| `VLM_ACTIVITY_IMAGE_DETAIL` | `"low"` | **Biggest single lever — a 3× token cut.** `None` = full detail. |
| `VLM_ACTIVITY_MAX_IMAGES` | `5` | Images per call. Token cost is **linear** in this. |
| `VLM_ACTIVITY_MIN_SECONDS_BETWEEN_CALLS` | `6.0` | Wall-clock floor across **all** tracks. The real spend bound. |
| `VLM_ACTIVITY_MAX_BACKOFF_SECONDS` | `60.0` | Cap on exponential backoff after HTTP 429. |
| `VLM_ACTIVITY_API_MAX_RETRIES` | `1` | SDK-level retries. Kept low — SDK retries sleep **inside our worker thread**. |
| `VLM_ACTIVITY_MAX_IMAGE_SIDE` | `512` | Longest side in px. Affects **upload size, not token count** at `detail="low"`. |
| `VLM_ACTIVITY_JPEG_QUALITY` | `85` | JPEG quality. |
| `VLM_ACTIVITY_LABEL_TTL` | `200` | **source frames** an on-screen label stays valid before it is dropped. |

### Dead config

| Key | Status |
|---|---|
| `VLM_ACTIVITY_MIN_MOVEMENT` | **Read by nothing.** Kept as documented intent, not a live knob. The idea — skip stationary people — conflicts with the goal, since "sitting" and "loitering" are exactly the stationary states worth reporting. |

---

## 6. Cost and rate control

This is the part most likely to bite you, so it is spelled out in full.

### Measured token cost

Real API calls on real crops from `sitting.mp4`, reading `response.usage.prompt_tokens`:

| Images | `detail` | Prompt tokens | Payload | Verdict returned |
|---|---|---|---|---|
| 11 | `auto` | **85,319** | 242 KB | `sitting` ✅ |
| 11 | `low` | 28,649 | 242 KB | `sitting` ✅ |
| **5** | **`low`** | **14,484** | **121 KB** | **`sitting` ✅** ← shipping default |
| 5 | `low` (384px) | 14,484 | 77 KB | `sitting` ✅ |
| 4 | `low` | 11,651 | 97 KB | `sitting` ✅ |

**Every setting returned the correct label.** The cheap settings cost accuracy nothing on this
footage — so the expensive ones were pure waste.

### The three levers, in order of impact

**1. `detail: "low"` — a flat 3× cut.**
`low` bills a flat ~2,833 prompt tokens per image instead of tiling it by resolution.
85,319 → 28,649 on the identical 11 images.

**2. `VLM_ACTIVITY_MAX_IMAGES` — linear.**
11 → 5 images halved it again: 28,649 → 14,484.

**3. `VLM_ACTIVITY_MIN_SECONDS_BETWEEN_CALLS` — bounds the rate.**

> ⚠️ **`VLM_ACTIVITY_MAX_IMAGE_SIDE` is NOT a token lever at `detail="low"`.**
> Measured: 512 px and 384 px both cost exactly **14,484** tokens. It only changes upload bytes
> (121 KB → 77 KB), which affects *latency*. Do not lower it expecting a cheaper bill.

### The tokens-per-minute budget

The account limit is **200,000 tokens/min** for `gpt-4o-mini`, and it is **shared with the RAG
analyst** (`text-embedding-3-small` ingestion plus `gpt-4o-mini` chat answers). The VLM must not
spend all of it.

At 14,484 tokens/call:

| Floor | Calls/min | Tokens/min | Headroom for the analyst |
|---|---|---|---|
| 5.0 s | 13.3 | 192,788 | 7,212 — **too tight** |
| **6.0 s** | **9.8** | **142,124** | **57,876 ✅ shipping default** |
| 8.0 s | 7.5 | 108,630 | 91,370 — very safe, labels refresh slowly |

Measured end-to-end on `sitting.mp4` at the 6.0 s default: **3 calls in 18.3 s = 9.8 calls/min =
142,124 tokens/min.**

### Adaptive backoff on HTTP 429

A 429 is not a bug — it is the account saying you asked too often. On any error whose text
contains `429` or `rate_limit`:

```
wait = min(MAX_BACKOFF, MIN_SECONDS_BETWEEN_CALLS * 2 ** min(occurrences, 4))
_backoff_until = monotonic() + wait
```

All analysis pauses globally until `_backoff_until`. Each **successful** call decrements the
occurrence counter, so the system decays back to normal cadence rather than staying pessimistic
for the rest of the run. The WARNING names the knob to raise.

### Rough money cost

At `gpt-4o-mini` pricing and the shipping default of ~142k prompt tokens/min, a **10-minute**
surveillance run is on the order of **1.4M prompt tokens**. Check current pricing before
budgeting a long run — and note this scales with *wall-clock* run length, not video length,
because the floor is wall-clock.

---

## 7. Output format

The model is called in JSON mode and returns:

```json
{
    "activity": "sitting",
    "description": "The person is sitting in a chair with a relaxed posture, occasionally shifting their position.",
    "confidence": 0.9,
    "reasoning": "<brief explanation>"
}
```

`reasoning` is requested to improve the answer but is **not** stored on `ActivityResult`.

### Verified output on the real clip

Through the shipped path on `sitting.mp4` (60 fps, 10.1 s):

| Video time | Activity | Conf | Description |
|---|---|---|---|
| 1.5 s | `walking` | 0.90 | "walking with a consistent gait through the space" |
| 4.5 s | `interacting_with_object` | 0.80 | "leaning towards and possibly reaching for a chair" |
| 9.5 s | `sitting` | 0.90 | "sitting in a chair with a relaxed posture" |

The middle label — reaching for the chair — is the transition, correctly caught.

---

## 8. Where results go

### 8.1 On-frame overlay ✅ live

[utils/video_utils.py](utils/video_utils.py) `draw_surveillance_ui`, to the right of the box:

```python
activity_label = f"{activity.activity} ({activity.confidence:.2f})"
cv2.putText(frame, activity_label, (x2 + 5, max(20, y1 + 15)), ...)
```

### 8.2 Sidebar status panel ✅ live

`streamlit_app.py` renders the **last 3** cached activities as
`Track {id}: **{activity}** ({confidence})`.

### 8.3 TTL — how a label stops being shown ✅ live

The renderer reads through `get_activity(track_id, frame_id)`, which returns `None` once
`frame_id - result.frame_ids[-1] > VLM_ACTIVITY_LABEL_TTL`. The caller then **pops** the entry:

```python
completed_activity = analyzer.get_activity(current_id, frame_id)
if completed_activity:
    activity_cache[current_id] = completed_activity
else:
    activity_cache.pop(current_id, None)
```

> **`VLM_ACTIVITY_LABEL_TTL` must be ≥ the source frames that elapse between calls, or labels
> blank out between refreshes.** At the 6.0 s floor, ~10 processed fps and `FRAME_SKIP=2`, that
> is `6 × 10 × 2 ≈ 120` source frames. The default **200** leaves margin on slower machines while
> still bounding how old a displayed label can be.

### 8.4 CSV persistence ❌ NOT WIRED

[storage/activity_observer.py](storage/activity_observer.py) defines `ActivityObserver`, which
appends to `storage/activity_observations.csv`:

```
Timestamp, Track_ID, Activity_Label, Description, Confidence, Frame_ID
```

**Nothing imports it.** Verified: the only occurrence of `ActivityObserver` in the repo is its own
`class` statement. Consequences:

- `storage/activity_observations.csv` is never written.
- Activity results live only in memory and are **discarded when the run ends**.
- The **RAG analyst never sees activity data** — it reads `storage/event_logs.csv` only.

Wiring it is a small change:

```python
from storage.activity_observer import ActivityObserver
observer = ActivityObserver()
...
if completed_activity:
    observer.log_activity(completed_activity)
```

**Design intent — keep it in its own CSV.** Activities fire every few seconds; events
(intrusion, theft) are rare. Mixing them would swamp `event_logs.csv`, which is the seam between
the vision half and the analyst, and would change what the analyst narrates.

---

## 9. Error handling and failure modes

| Failure | Handling | Surveillance impact |
|---|---|---|
| No API key / client build fails | logged ERROR, `self.enabled = False`, `client = None` | none — silently disabled |
| Request timeout | caught, logged ERROR, returns `None` | none |
| HTTP 429 rate limit | **global backoff**, logged WARNING naming the knob | analysis pauses, video keeps processing |
| Malformed / non-JSON response | caught, logged WARNING, returns `None` | previous label stands until TTL |
| Missing JSON fields | defaults: `"unknown"` / `"Activity unclear"` / `0.5` | none |
| `confidence` out of range | clamped to `[0.0, 1.0]` | none |
| Any other API exception | caught, logged ERROR, returns `None` | none |
| Image encode fails | that image skipped; if **all** fail, returns `None` | none |

**Nothing in this module can abort a run.** Every path through `_invoke_vlm` is wrapped, and
`_complete_analysis` swallows worker exceptions so a failed future never propagates into the
Streamlit thread.

---

## 10. Concurrency model

- `ThreadPoolExecutor(max_workers=4, thread_name_prefix="vlm-activity")`.
- `analyze_if_ready` **submits and returns immediately** — the main loop never waits on the
  network. Verified by test: submission returns in < 0.08 s while the worker sleeps 0.1 s.
- **One analysis in flight per track**, guarded by `_pending_futures`.
- The wall-clock floor and backoff are **global**, not per track.
- `_lock` guards `activity_cache`, `_pending_futures` and the throttle state.

### ⚠️ `_lock` must be an `RLock`

This is load-bearing and easy to break:

`analyze_if_ready` registers the done-callback **while holding `_lock`**, and
`Future.add_done_callback` runs the callback **in the calling thread** if the future has already
finished. `_complete_analysis` then re-acquires the same lock.

With a plain `threading.Lock` that is a **self-deadlock that hangs the entire surveillance loop**,
not just the analysis. It fires whenever the worker completes fast — an instant exception with no
network, or an immediate HTTP 429.

This was a real latent bug, present before the current fixes, surfaced by
`test_throttle_allows_call_once_floor_has_passed`. There is a regression test for it now.

### Lifecycle

| Method | Effect |
|---|---|
| `cleanup_track(id)` | drops the buffer + cache entry, cancels a pending future |
| `reset()` | clears everything, cancels all futures — called at the start of each run |

---

## 11. Bugs found and fixed (with evidence)

All four were found by measurement on `sitting.mp4`. Recorded as **VF-42** and **VF-43** in
[CLAUDE.md](CLAUDE.md).

### 11.1 The label was stale — the VLM was never wrong

**Symptom:** a clearly seated person displayed `walking (0.90)`.

**Cause:** with the shipped gating, only **3 analyses ran for the entire 10.1 s clip**, each
covering **under 1 s of video**, leaving **4 s unanalysed** between them. The trigger frames were
dumped and inspected — at frame 30 and frame 270 the person genuinely **was** walking toward the
chair. The frame-510 sitting analysis never landed before the run ended.

**Proof the model is accurate:** fed the sitting frames directly, `gpt-4o-mini` returned
`sitting` at **confidence 1.0** — and returned `sitting` **with the colour bug still present**, so
colour was not the cause.

Three compounding root causes:

1. A **second freshness gate** in `analyze_if_ready`
   (`current_frame_id - cached.frame_ids[-1] < interval * 2`) that **doubled** the effective
   interval on top of `should_analyze`, which already rate-limited on the same clock. Removed.
2. Results **never expired**, so a verdict displayed forever. Fixed with `VLM_ACTIVITY_LABEL_TTL`.
3. Only one analysis per track in flight, so a ~10 s API call blocked refresh for 10 s of wall
   time. Mitigated by cutting payload and token cost.

**Result:** 3 analyses → **20** (one every 0.5 s of video) before the throttle, settling at a
sustainable **9.8 calls/min** after.

### 11.2 Colour channels were double-swapped in every image ever sent

`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` followed by `cv2.imencode`, which itself expects BGR.

**Measured:** a centre patch of BGR `[171, 190, 201]` arrived as `[201, 190, 171]`. Skin rendered
blue. Fixed by passing the BGR frame straight to `imencode`.

### 11.3 Payload was ~3.26 MB per call

Full-frame lossless PNG at source resolution.

> ⚠️ **Cropping to the person made it *worse*, not better — 3.95 MB.** The clip is portrait, so
> the crop is barely smaller than the frame, and PNG compresses a detailed person crop poorly.
> An earlier claim that cropping would cut payload ~30× was **wrong** and was corrected by
> measurement.

What actually worked: **downscale to 512 px + JPEG q85 → 0.280 MB mean, 0.375 MB max (11.6×
smaller)**. This is a correctness fix, not just a cost one: upload latency bounds label freshness,
because only one analysis per track is in flight.

### 11.4 Blew the tokens-per-minute limit

Once staleness was fixed, the higher call rate hit **HTTP 429: Limit 200000, Used 200000**. Fixed
by `detail="low"` + `MAX_IMAGES=5` + the wall-clock floor + adaptive backoff — [§6](#6-cost-and-rate-control).

---

## 12. Performance and measured numbers

### Before vs after, on `sitting.mp4` (607 frames, 60 fps, 10.1 s)

| Metric | Before | After |
|---|---|---|
| Analyses per clip | 3 | 20 (throttled to 3 sustainable) |
| Video covered per call | < 1 s | ~1 s |
| Video unanalysed between calls | 4 s | 0.5 s (pre-throttle) |
| Images per call | 11 | 5 |
| Payload per call | 3.26 MB PNG | **0.280 MB JPEG** (11.6×) |
| Prompt tokens per call | 85,319 | **14,484** (5.9×) |
| Tokens/min | ~350,000 (429s) | **142,124** ✅ |
| Colour | R/B swapped | correct |
| Subject | whole room | padded person crop |
| Label expiry | never | 200 source frames |

### Main-loop cost

The VLM adds **no measurable time to the frame budget**. Per frame per track it does a crop
(`frame[y1:y2, x1:x2].copy()`), a list append, and a handful of integer comparisons. All network
and encoding work happens on worker threads.

This matters because the pipeline is already the binding constraint: end-to-end throughput is
**~10.6 processed fps** (VF-41) against the **15 fps** a 30 fps camera needs at `FRAME_SKIP=2`.
The VLM must not make that worse, and it does not.

---

## 13. Tuning guide

### "Labels are stale / lag behind what the person is doing"

1. Lower `VLM_ACTIVITY_MIN_SECONDS_BETWEEN_CALLS` — **check the TPM table first**.
2. Lower `VLM_ACTIVITY_ANALYSIS_INTERVAL`.
3. Lower `VLM_ACTIVITY_MAX_IMAGES` to cut latency and cost, buying a lower floor.
4. Keep `VLM_ACTIVITY_LABEL_TTL` ≥ the source frames between calls, or labels blank out.

### "I'm getting HTTP 429"

1. **Raise `VLM_ACTIVITY_MIN_SECONDS_BETWEEN_CALLS`** — the direct fix.
2. Lower `VLM_ACTIVITY_MAX_IMAGES` (linear token saving).
3. Confirm `VLM_ACTIVITY_IMAGE_DETAIL = "low"`.
4. Remember the RAG analyst shares the budget; ingesting a large `event_logs.csv` spikes it.

### "It costs too much"

In order of impact: `detail="low"` (3×) → fewer `MAX_IMAGES` (linear) → longer floor (linear) →
`USE_VLM = False` (free).
**Do not** lower `MAX_IMAGE_SIDE` for cost — measured, it changes nothing at `detail="low"`.

### "Labels are wrong / vague"

1. Raise `VLM_ACTIVITY_MAX_IMAGES` — more temporal context.
2. Set `VLM_ACTIVITY_IMAGE_DETAIL = None` for full detail (**~3× the tokens**).
3. Raise `VLM_ACTIVITY_MAX_IMAGE_SIDE` — costs upload time, not tokens.
4. Widen `VLM_ACTIVITY_WINDOW_SIZE` so the window spans more video.
5. Edit the label list in `_build_activity_prompt` for your domain.

### "Labels blank out and reappear"

`VLM_ACTIVITY_LABEL_TTL` is too low for your call rate. Raise it, or lower the floor.

---

## 14. Testing

```bash
python -m unittest discover -s tests      # run from the project root
```

10 tests in [tests/test_vlm_activity_analyzer.py](tests/test_vlm_activity_analyzer.py), all
network-free (`MagicMock` client):

| Test | Guards against |
|---|---|
| `test_openai_request_and_response_parsing` | request shape, model, JSON mode, parsing |
| `test_timeout_is_handled` | timeout → `None`, no raise |
| `test_api_failure_is_handled` | generic API error → `None`, no raise |
| `test_analysis_is_submitted_to_background_worker` | main loop never blocks (< 0.08 s) |
| `test_wall_clock_throttle_blocks_rapid_calls` | the TPM guard actually holds |
| `test_throttle_allows_call_once_floor_has_passed` | throttle isn't permanently stuck — **also the RLock deadlock regression** |
| `test_rate_limit_error_triggers_backoff_and_returns_none` | 429 sets backoff, returns `None` |
| `test_images_are_capped_and_span_is_preserved` | image cap holds, first+last survive |
| `test_crop_uses_bbox_and_pads` | crop is smaller than the frame but padded |
| `test_stale_label_expires` | TTL boundary, and no expiry without a frame id |

### Manual verification against the real API

Scripts used during debugging (not checked in — recreate in `scratchpad/` if needed):

| What to measure | Approach |
|---|---|
| What the model actually sees | Run the encode path, `base64` **decode** it back, `cv2.imwrite` the result, and look at it. This is what found the colour and framing bugs. |
| Real token cost | Read `response.usage.prompt_tokens` from a real call — do not estimate. |
| Call rate / TPM | Drive the real detector + real gating with the network stubbed; count `create` calls over wall time. |
| Accuracy | Real API call on crops from a known segment; assert the label. |

---

## 15. Troubleshooting

| Symptom | Likely cause | Check |
|---|---|---|
| No labels at all | `USE_VLM = False`, or client failed to build | look for `VLMActivityAnalyzer initialized (enabled)` at startup |
| `openai.OpenAIError` on startup | **the key must be in `GOOGLE_API_KEY`** — `config.OPENAI_API_KEY = os.getenv("GOOGLE_API_KEY")` | `echo $GOOGLE_API_KEY` |
| Labels only on some people | ephemeral tracks (negative ids) are skipped by design | normal |
| Labels never update | check for 429 WARNINGs; check the floor | logs |
| Labels blank out | `LABEL_TTL` < frames between calls | [§8.3](#83-ttl--how-a-label-stops-being-shown--live) |
| `activity_observations.csv` empty | `ActivityObserver` is not wired | [§8.4](#84-csv-persistence--not-wired) |
| Analyst can't answer activity questions | activities never reach `event_logs.csv` | [§8.4](#84-csv-persistence--not-wired) |
| **UI frozen, video stopped** | `_lock` is a `Lock` not an `RLock` | [§10](#10-concurrency-model) |
| Blue-looking people in debug dumps | a `cvtColor(BGR2RGB)` was reintroduced before `imencode` | [§11.2](#112-colour-channels-were-double-swapped-in-every-image-ever-sent) |

---

## 16. Known limitations and future work

### Current limitations

1. **Results are not persisted** — `ActivityObserver` is unwired ([§8.4](#84-csv-persistence--not-wired)). Biggest open gap.
2. **The analyst is blind to activity** — follows from 1.
3. **`VLM_ACTIVITY_ANALYSIS_INTERVAL` is frame-rate dependent** — 30 frames is 1 s at 30 fps and 0.5 s at 60 fps. The wall-clock floor compensates but the knob is still misleading.
4. **`VLM_ACTIVITY_MIN_MOVEMENT` is dead config.**
5. **A label is up to `LABEL_TTL` source frames old.** Bounded, but not "now".
6. **Cost scales with wall-clock run length**, not video length.
7. **Requires network.** No offline fallback since the pose recognizer was deleted.
8. **One person per call.** N people ⇒ N× the calls, all sharing one global floor, so per-person freshness degrades linearly with crowd size. **Untested on crowd footage.**
9. **Confidence is self-reported** by the model and is not calibrated against ground truth.

### Future work

| Idea | Value |
|---|---|
| **Wire `ActivityObserver`** | high — cheap, unblocks persistence and the analyst |
| Feed activity into the RAG analyst | high — "what was Talha doing at 14:32?" |
| Express the interval in **seconds of video**, not frames | medium — removes the frame-rate trap |
| Batch multiple tracked people into one call | medium — the fix for limitation 8 |
| Only re-analyse when the crop changes materially | medium — skip calls when nothing moved |
| Local VLM (LLaVA / Qwen-VL) | removes cost and network dependence; costs throughput the pipeline cannot spare |
| Calibrate confidence against labelled clips | closes limitation 9 |

---

## Appendix: quick reference

```python
# Enable / disable
config.USE_VLM = True

# The three cost levers, in order of impact
config.VLM_ACTIVITY_IMAGE_DETAIL = "low"            # 3x cut
config.VLM_ACTIVITY_MAX_IMAGES = 5                  # linear
config.VLM_ACTIVITY_MIN_SECONDS_BETWEEN_CALLS = 6.0 # bounds the rate

# Freshness (must be >= source frames between calls)
config.VLM_ACTIVITY_LABEL_TTL = 200
```

```python
# Public API
analyzer = VLMActivityAnalyzer()
analyzer.add_frame(track_id, frame, frame_id, bbox)   # per frame, per track
analyzer.analyze_if_ready(track_id, frame_id)         # non-blocking; may submit
result = analyzer.get_activity(track_id, frame_id)    # TTL-aware read
analyzer.cleanup_track(track_id)                      # track died
analyzer.reset()                                      # new run
```

**Golden rules**

1. Never add `cvtColor(BGR2RGB)` before `imencode`.
2. `_lock` must stay an `RLock`.
3. `LABEL_TTL` ≥ source frames between calls.
4. `MAX_IMAGE_SIDE` is not a token lever at `detail="low"`.
5. Measure `response.usage.prompt_tokens`; never estimate it.
