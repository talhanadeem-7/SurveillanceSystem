# CLAUDE.md — Shelby Surveillance System

> Read this file before starting any task. Update it after finishing any task.
> Rule for VERIFIED FACTS: **never record a claim that came from reading code alone.**
> If it was not measured, it belongs in OPEN QUESTIONS. We have already been wrong
> twice by inferring behaviour from source.

---

## 1. What this project is

Shelby is a single-machine, offline **video-file** surveillance analysis system. A user
uploads a recorded clip through a Streamlit UI, draws restricted (rectangle) and passive
(polygon) zones on frame 0, and runs a per-frame pipeline: YOLOv8n-pose detects people →
boxmot BotSort assigns track IDs → an in-house `IdentityGuardian` optionally corrects ID
switches by appearance → DeepFace/ArcFace matches faces against enrolled photos → a
2D+MiDaS-depth test decides zone intrusion → an edge-density test decides object removal
("theft"). Every decision is appended to `storage/event_logs.csv`, which is the **only**
interface to the second half of the system: a LangChain RAG agent ("Shelby Analyst") that
embeds those log rows into Chroma and answers natural-language questions with `gpt-4o-mini`.
It is not a live-camera system, has no authentication, and has no database server.

**Entry point:** `streamlit run streamlit_app.py` (run from the project root — `config.MODEL_PATH`
is a relative string). `app.py` is a legacy OpenCV-desktop variant and is **out of scope** —
do not modify it. Python **3.11/3.12 only** (`venv311/` is 3.11.9; `venv/` is 3.13 and cannot
run boxmot).

---

## 2. Current phase and task list

**Phase 1 — hardening pass.** Original six tasks were reordered after measurement
invalidated several premises.

| Task | Status | Notes |
|---|---|---|
| 1 — Reduce tracker `max_age` 1200→75, expose as `TRACKER_MAX_AGE` | **done** | Implemented as specified, but **inert**: `max_age` does not govern removal (VF-1). Kept because it drives `max_obs`. |
| 1c — Ephemeral ID collision fix | **done** | Monotonic counter in `detector.py` + skip-negative guards on every stateful dict in `run_surveillance`. Measured: collisions 244→0, 155→0, 33→0. |
| 1c-2 — Confidence dead band | **done** | `TRACKER_NEW_TRACK_THRESH = 0.45` **committed** (not just exposed). Mode C rejected, Mode B verified genuine. VF-19/20/21/22. **Must stay aligned with `CONFIDENCE_THRESHOLD` or the band reopens.** |
| 1b — Empty-frame tracker freeze | **done** | Fixed in **both** `tracker.py` and `detector.py` (VF-23) — the tracker.py fix alone was provably insufficient. Empty scene now ages identically to a populated one (VF-24). office cctv now resolves to **6** tracks; ruled OQ-10 = accept 6, **VF-13 rule rewritten** (old value 5 was a freeze artefact, VF-25). Re-entry study re-measured post-fix (VF-27). |
| 1a — Plumb `TRACKER_TRACK_BUFFER` + `TRACKER_MATCH_IOU_THRESHOLD` | **done** | Defaults 30 and 0.2, behaviour preserved; all three clips byte-identical after. `track_buffer` comment records VF-1 so nobody re-derives it; match-IoU comment records VF-8 (exposed for visibility, not a working lever). |
| Guardian A/B — `IDENTITY_GUARDIAN_ENABLED`, default False | **done** | Flag added (default **False**), gated in `detector.py`; class left fully intact. Measured: Guardian is pure over-merge on these clips (VF-11/12/13), **but** OSNet alone cannot cover long gaps (VF-14) — see OQ-6. |
| 2 — MiDaS module-level singleton | **NOT DONE — carried to Phase 2** | Never reached; the queue was reprioritised into tracker correctness. Still reloaded per `run_surveillance()` call via `PoseAnalyzer.__init__`. |
| 3 — Gate `align_zones` (`ZONE_ALIGN_INTERVAL`=15, `ZONE_ALIGN_ENABLED`=True) | **NOT DONE — carried to Phase 2** | Never reached. ORB+RANSAC still runs every processed frame. |
| 4 — Per-frame try/except in `run_surveillance` + `finally` release | **NOT DONE — carried to Phase 2** | Never reached. The main loop still has no error handling and no `finally`; one bad frame aborts the run, leaks `cap`, and never sets `processing_complete`. |
| 5 — Confirmed vs predicted track states | **probably dropped** | Premise likely false (VF-4). User will drop it if the end-to-end assertion holds (OQ-3). `get_track_state_for_display` is unusable regardless (VF-3). |
| 6a — IdentityGuardian embedding edge-normalisation fix | **on hold** | Do not implement while the Guardian may be disabled outright. Bug confirmed and is total, not partial (VF-5). |
| 6b — `IDENTITY_GUARDIAN_ENABLED` flag | folded into Guardian A/B | Default changed from True to **False**. |
| 6c — Comment block marking Guardian as secondary ReID layer | **not started** | |

---

## 2b. PHASE 1 CLOSEOUT

*Written for someone who has not seen any of the work. Phase 1 was scoped as a six-task hardening
pass. Measurement invalidated the premises of several tasks, so the queue was reprioritised into
tracker correctness. Everything below was verified by running the pipeline, not by reading code.*

### What shipped

All changes are in `config.py`, `vision/tracker.py`, `vision/detector.py` and `streamlit_app.py`.

1. **Ephemeral ID collision fix** (the largest correctness win, and it was not on the original
   list). Detections the tracker cannot match used to get IDs by *position* — index 0 was always
   `-1` — so unrelated people collided in every per-ID dict. Now a monotonic counter, never reused.
   Measured: colliding detections **244→0, 155→0, 33→0**. In addition, negative IDs are now barred
   from accumulating any cross-frame state (identity, intrusion persistence, trajectory); they are
   still drawn and still evaluated for the current frame.
2. **Empty-frame tracker freeze fix.** An empty scene never advanced BotSort's clock, so lost
   tracks survived forever and were re-matched to whoever appeared next. Required fixes in **two**
   places — `detector.py` returns before `tracker.py` is reached, so the obvious one-line fix was
   unreachable. Low impact on file playback, critical for a live camera (VF-26).
3. **Confidence dead-band closure.** `TRACKER_NEW_TRACK_THRESH = 0.45`, aligned with
   `CONFIDENCE_THRESHOLD`. Cut detections invisible to the event pipeline from **31% → 15%** on the
   worst clip. The opposite fix (raising `CONFIDENCE_THRESHOLD`) was measured and **rejected**.
4. **IdentityGuardian disabled behind `IDENTITY_GUARDIAN_ENABLED = False`.** It was merging
   visibly different people (a yellow hoodie with dark jackets). Class left fully intact for
   comparison.
5. **Four tracker knobs exposed in `config.py`**: `TRACKER_MAX_AGE`, `TRACKER_TRACK_BUFFER`,
   `TRACKER_NEW_TRACK_THRESH`, `TRACKER_MATCH_IOU_THRESHOLD`, each with a comment recording what
   measurement showed it actually does.

### What did NOT ship (carried into Phase 2)

Tasks **2** (MiDaS singleton), **3** (gate `align_zones`) and **4** (per-frame error handling in
`run_surveillance`) were **never reached** — the queue was consumed by tracker correctness work.
They remain open and unstarted. Task 4 in particular is a real robustness hole: the main loop still
has no `try`/`except` and no `finally`, so one bad frame aborts a run and leaks the capture handle.

### What was dropped, and why

- **Task 5 (confirmed vs predicted track states) — pending one assertion.** Its premise was that
  Kalman-predicted boxes reach the zone logic. Measurement says they cannot: BotSort emits only
  `Tracked + is_activated` tracks (VF-4), and our geometry comes from YOLO detections. Formally
  closed by **OQ-3**. Separately, boxmot's own `get_track_state_for_display` is unusable — it
  always returns `"confirmed"` (VF-3).
- **Task 6a (Guardian embedding fix) — dropped.** The bug is real and total (the embedding is 100%
  edge energy; colour never participated, VF-5), but fixing the embedding of a layer that is now
  disabled is wasted effort. Revisit only if OQ-6 forces the Guardian back.

### Regression baseline at Phase 1 close

`office cctv.mp4`, Guardian off, committed defaults → **6 tracks**, split at frames 430-448
(track 4 ends 429, track 6 starts 450), median life 147, 29 ephemeral (2.8%).
`crowd sample.mp4` → 787 dets / 117 ephemeral / 27 tracks / median 23.
`crowded sample2.mp4` → 2502 dets / 124 ephemeral / 43 tracks / median 40.

### Open questions carried into Phase 2

| ID | Question | Priority |
|---|---|---|
| **OQ-9** | **Split-intrusion attribution defect** — a confirmed intrusion split across two `Person_<id>` rows corrupts the CSV the analyst reads. Cheap mitigation, independent of any gallery. | **highest — corrupted record, not a missed detection** |
| OQ-6 | Long-gap ReID gallery (OSNet-based). Deferred to Phase 6; K=0 on target footage. | Phase 6 |
| OQ-12 | Why OSNet failed to re-associate at 21 updates when it recovered at 30. Likely `proximity_thresh` + Kalman drift. Cheaper lead than the gallery. | medium |
| OQ-13 | Enrol someone who appears in `office cctv.mp4` — the one step that makes M meaningful. | medium, cheap |
| OQ-3 | Assert no predicted box reaches `check_trespassing`, then formally drop Task 5. | low |
| OQ-1 | What deleted `data/reid_snapshots/`. | low, unexplained |
| OQ-4, OQ-7, OQ-8 | Guardian merge correctness; gaps 120/300 unmeasured; full-res crowd sheet inspection. | low |

---

## 3. VERIFIED FACTS

Established by measurement. Evidence in one line each.

- **VF-1 — BotSort track removal is governed by `track_buffer`, NOT `max_age`.**
  Measured: gap of 31 tracker updates preserves the ID, 32 does not; boundary unchanged at
  `max_age` = 5, 75 and 1200; boundary moves to 10 when `track_buffer` is set to 10.

- **VF-2 — `max_age` has no measurable effect on our pipeline.**
  Measured: full baseline over 3 clips at `max_age` 75 vs 1200 produced byte-identical
  raw-ID counts, corrected-ID counts, ID-correction counts and unmatched counts (only wall-clock
  `seconds` differed). (A source grep additionally shows it reaching only `max_obs` and a display
  helper — that part is code-reading, not measurement.)

- **VF-3 — `get_track_state_for_display` is degenerate for BotSort; do not build on it.**
  Measured: a track driven into `TrackState.Lost` (state=2) still reported
  `time_since_update=0` and the helper still returned `"confirmed"`. `time_since_update` is
  declared in `botsort/basetrack.py` and never assigned anywhere in the package.

- **VF-4 — BotSort emits only Tracked + is_activated tracks; lost/Kalman-predicted boxes are
  not returned.** Measured: `update()` returned **0 output rows** while `lost_stracks` held 1 track.

- **VF-5 — `IdentityGuardian`'s embedding is 100% edge energy; colour never participated.**
  Measured across 394 real crops: edge block's share of pre-normalisation vector energy had
  median, p10 **and** p90 all = 100.00%. The three HSV histograms are L2-normalised to 1 each
  (3 units total) while the raw Canny band sums reach ~10⁹.

- **VF-6 — The Guardian is over-merging, confirmed visually.** Measured/observed: 40 raw IDs →
  8 identities on a full-colour clip, with red top, red trousers, beige coat, white t-shirt and
  black outfits all in identity 3; 5 raw IDs → 1 identity on office footage merging a bright
  yellow hoodie with dark clothing. Contact sheets in scratchpad.

- **VF-7 — Empty detection arrays freeze the tracker's clock.** Measured through
  `StrongSortTracker`: with the scene empty, an ID survived gaps of 32, 60 and 150 updates
  (it should die after 31); with other people detected during the gap, it died at 32 as expected.

- **VF-8 — Unmatched detections are structural, not a threshold problem.** Measured: 100% of
  unmatched detections (244/244 and 156/156) were "starved" — BotSort emitted fewer track boxes
  than there were detections; **zero** failed because IoU fell below `TRACKER_MATCH_IOU_THRESHOLD`.
  91.4% of unmatched detections have confidence in the dead band between our
  `CONFIDENCE_THRESHOLD=0.45` and BotSort's `new_track_thresh=0.6` (median unmatched conf 0.511
  vs 0.622 matched), where a detection can never spawn a new track.

- **VF-9 — Ephemeral IDs were colliding by construction; fixed.** The old `-(idx + 1)` scheme
  made detection index 0 always `-1`. Measured before/after: colliding detections 244→0
  (8 distinct IDs → 244), 155→0, 33→0 across the three clips.

- **VF-11 — IdentityGuardian sits downstream of BotSort and can ONLY relabel output IDs. It
  cannot affect tracking quality, only attribution.** Measured: with the flag ON vs OFF,
  track-lifetime distributions are byte-identical in every column across all three clips
  (office cctv median 214 / mean 203.4 / max 361; crowd sample median 23 / mean 24.7 / max 58;
  crowded sample2 median 46 / mean 58.6 / max 223; identical <10-update and <30-update counts).
  **Consequence: every measurement ever taken of the Guardian was measuring a labelling layer,
  not a tracking layer.** No Guardian change can improve or degrade tracking.

- **VF-12 — With the Guardian OFF, raw BotSort ID counts are 5 / 22 / 40; with it ON the final
  identity counts collapse to 1 / 10 / 8.** Measured across office cctv, crowd sample,
  crowded sample2.

- **VF-13 — `office cctv.mp4` is the STANDING REGRESSION CASE for all ReID work.**
  The clip contains **5 visually distinct people** (dark green top, navy jacket, navy hoodie,
  **yellow hoodie**, black t-shirt) — confirmed on the guardian-OFF contact sheet.
  **With a correct tracker clock, BotSort resolves them into 6 TRACKS**, splitting one person
  across the empty-frame run at processed frames **430-448**: track 4 ends at frame 429, track 6
  begins at frame 450 (a ~21-update absence BotSort fails to bridge — see VF-25, OQ-12).
  > **RULE: `office cctv.mp4` must resolve to 6 tracks, AND the split must occur at exactly that
  > gap (track ending ~429, successor starting ~450).** If a change moves the split, or produces a
  > different count, **investigate** — do not assume either direction is an improvement.
  ⚠️ **The old rule said 5. That value was an ARTEFACT of the empty-frame freeze** (frames 429 and
  449 looked adjacent to BotSort, so it bridged them trivially). It was correct for the wrong
  reason. **Do not restore 5.** We do not preserve a number produced by a bug.

- **VF-14 — BotSort+OSNet recovers a track ID across gaps of 10 updates on both clips, 30 updates
  on office cctv only, and FAILS at 60 on both. Crowds fail earlier than sparse scenes.**
  Measured on real footage by withholding one person's detection for N updates and re-introducing
  it (target identified exactly via boxmot's `det_ind` column; BotSort driven directly so
  `frame_count` advances). Consistent with `track_buffer=30` (VF-1). At FRAME_SKIP=2 on 25-30fps
  footage, 30 updates ≈ **2-2.4 seconds of video**.
  **Gaps of 120 and 300 are UNMEASURED, not passing** — neither benchmark clip has a contiguous
  single-person run long enough to test them (longest runs: 122 and 135 frames). ≤60 on both
  clips is the established bracket.

- **VF-15 — ⛔ SUPERSEDED-PENDING (do NOT rely on these numbers).** Measured **before the Task 1b
  empty-frame fix**, i.e. under the frozen tracker clock: tracks aged only while someone was on
  screen, so they died less often than they should have and re-entry events are **undercounted**.
  VF-25 shows a genuine office cctv split that the freeze was masking. The Phase 6 deferral (OQ-6)
  rests on the K=0 figure below and is therefore also pending. Being re-measured — see OQ-11.
  Historical values follow.
  Re-entry events: 0 on office cctv, ~1 per second of video on dense crowd footage.
  Measured with the Guardian OFF, counting track deaths followed 30-200 updates later by a
  plausible re-entry, scored with **boxmot's own OSNet features** and a **6-box-height spatial
  gate**: office cctv **N=0**; crowd sample **N=12** (~14 s); crowded sample2 **N=28** at
  similarity ≥0.5, **18** at ≥0.6 (~22 s). Tracks whose intrusion counter was live at death:
  **K = 0 / 3 / 8**. Tracks that could have held a face identity: **M = 0 / 0 / 0**.

- **VF-27 — RE-ENTRY STUDY, RE-MEASURED POST-1b (supersedes VF-15). K is still 0 on office cctv,
  so the Phase 6 deferral stands.** Same harness, correct tracker clock:
  | Clip | N @0.5 (was) | N @0.6 (was) | **K** @0.5 (was) | M (was) |
  |---|---|---|---|---|
  | office cctv | **1** (0) | **1** (0) | **0** (0) | **1** (0) |
  | crowd sample | **14** (12) | **12** (12) | **6** (3) | 0 (0) |
  | crowded sample2 | **30** (28) | **20** (18) | **8** (8) | 0 (0) |
  The freeze was undercounting, as predicted — most sharply on crowd sample, where K **doubled
  3 → 6**. On the target clip the single new event is `dead track 1 → reborn track 6, gap 76,
  sim 0.658, persist@death 0`. Note this is **not** the VF-25 split (track 4 → 6, ~21 updates),
  which falls below the study's 30-update window.
  ⚠️ **M rose 0 → 1 on office cctv**: that dead track had **8 face-crop hits**, so this event
  *would* have cost a locked face identity had the person been enrolled.

- **VF-28 — THE SHAPE OF THE PHASE 6 RISK (read this before re-opening OQ-6).**
  The honest statement, in two halves:
  1. **Intrusion attribution: costs nothing measurable on target footage.** K = 0 on office cctv
     both before and after the 1b fix. This half is measured and settled.
  2. **Face-identity continuity: UNMEASURED, NOT ZERO.** The original M=0 was uninformative — the
     enrolled gallery holds one person (Talha) who appears in **none** of the test clips, so M
     could only ever be 0 for reasons unrelated to the gap. Post-1b there is now **one event on
     the target clip** where a face-capable track (**8 face-crop hits**) died and was reborn
     (gap 76, sim 0.658). Had that person been enrolled, the identity would have been lost.
  **Do not cite "the gap costs nothing" without this qualifier.** The face half of the risk is
  simply not yet quantifiable. Closing it is cheap — see OQ-13.

- **VF-29 — The re-entry harness has a BLIND SPOT below 30 updates.**
  `reentry_gap_impact.py` only counts re-entries with a gap in **[30, 200]** updates. The VF-25
  office cctv split is a **~21-update** gap and is therefore **invisible to that study entirely** —
  the VF-27 event (`track 1 → track 6`, gap 76) is a *different, independent* event.
  **Do not read the harness's N as "all re-entries".** Short-gap association failures (OQ-12) need
  a different instrument.

- **VF-16 — M=0 measures clip capability, not gap harmlessness.** The crowd clips have **zero
  face-capable tracks** (distant overhead views; no track yielded ≥2 extractable face crops at the
  app's `FACE_CHECK_INTERVAL` cadence). office cctv has exactly 1 face-capable track, and it had
  no re-entry event. Separately, the enrolled gallery holds only Talha, who appears in none of
  these clips, so real locked identities would be 0 regardless.

- **VF-17 — K is an UPPER BOUND.** The benchmark clips have no user-drawn zones, so a synthetic
  30%x30% zone was placed at the detection centroid — deliberately in **peak traffic** — and the
  **pure-2D** `depth_map=None` path was used, which fires more readily than the depth-gated path.
  Both choices inflate K.

- **VF-18 — OSNet similarities of 0.55-0.60 in crowds are weak identity evidence.** The re-entry
  proxy **cannot separate "same person returned" from "different person who looks alike"**, so the
  crowd-clip N figures are an order of magnitude, not a count of true re-identifications.

- **VF-19 — Closing the confidence dead band from BELOW fails the VF-13 regression rule; closing
  it from ABOVE is safe and halves ephemeral detections.** Measured three ways:
  | Mode | office cctv identities | ephemeral (crowd sample) | office cctv median life |
  |---|---|---|---|
  | A `conf=0.45, ntt=0.60` (current) | 5 ✅ | 244 (31.0%) | 214 |
  | B `conf=0.45, ntt=0.45` | 5 ✅ | **117 (14.9%)** | 215 |
  | C `conf=0.60, ntt=0.60` | **6 ❌ FAIL** | 32 (9.3%) | **115** |
  ⚠️ **These three modes were measured BEFORE the Task 1b fix**, when the regression baseline was
  still 5 (VF-25). Read the identity counts as *relative* — Mode C produced one more identity than
  the then-current baseline. The other columns are unaffected by 1b.
  **Mode C REJECTED.** Raising `CONFIDENCE_THRESHOLD` to 0.6 fragmented the regression clip
  (identity count up by one, median track life halved **214 → 115**) and **discarded 56% of
  detections on crowd sample** (787 → 345). It drops a person mid-track and re-acquires them as a
  new ID. Not re-measured post-1b; the rejection rests on the detection-loss and lifetime columns,
  which do not depend on the baseline.
  **Mode B ADOPTED** (`TRACKER_NEW_TRACK_THRESH = 0.45`, committed): preserves detection count and
  track lifetimes, passes the regression, and cuts ephemeral detections **31% → 15%** on the worst
  clip. Re-verified at the committed default: office cctv = 5 identities, median life 215,
  ephemeral 2.7%.

- **VF-20 — The tracks Mode B adds are genuine people, not duplicates.** Measured by matching
  tracks between the 0.60 and 0.45 runs by spatio-temporal IoU (ids are not comparable across
  runs), then scoring each unmatched track against every **concurrent** track in its own run:
  crowd sample 6 new tracks, crowded sample2 3 new tracks, **all with 0.0% of their life
  overlapping a concurrent track** (max IoU seen 0.193). Visually confirmed on isolated contact
  sheets: 8 of 9 are clearly distinct individuals. The exception is crowd sample track 31
  (18 updates) which produced **no crops at all** — every detection fell below
  `MIN_CROP_H/W` — so it is un-verifiable visually, though its zero overlap and 18-update life
  are consistent with a genuine distant person.

- **VF-21 — Since Task 1c, ephemeral detections accumulate NO state at all.** No intrusion
  persistence, no face identity, no trajectory, no zone attribution. **Therefore the ephemeral
  percentage is literally the fraction of detections invisible to the event pipeline** — which is
  what makes VF-19's 31% → 15% a coverage improvement rather than a cosmetic one.

- **VF-22 — The VF-13 regression rule has already earned its keep.** Mode C looked *best* on the
  headline metric (ephemeral 31% → 9.3%, the largest reduction of the three modes) and **would
  have shipped on that number alone**. Only the office cctv identity count exposed that it was
  fragmenting the target scenario. **Lesson: always check the regression rule before accepting a
  change that improves a single aggregate metric.**

- **VF-23 — The empty-frame freeze needed TWO fixes, not one; `tracker.py` alone is provably
  insufficient.** `detector.py` returns at `if result.boxes is None or len(result.boxes) == 0`
  **before** `self.tracker.update` is ever called, so on a genuinely empty scene the guard inside
  `StrongSortTracker.update` is unreachable. Measured end-to-end through `PoseDetector` with blank
  frames (0 detections): with only `tracker.py` fixed the ID survived gaps of 10/31/32/60/150 —
  **still frozen at every gap**. Both call sites are now fixed.

- **VF-24 — After both fixes, the empty scene behaves exactly like the populated one.** Measured
  end-to-end: gap 31 SAME, gap 32/60/150 NEW ID. Same boundary as VF-1, so the tracker clock now
  advances identically whether or not anyone is in frame.

- **VF-25 — ⚠️ The fix makes office cctv resolve to 6 identities, FAILING the VF-13 rule — and the
  old "5" was partly an artefact of the freeze.** Measured cause: office cctv contains **39 empty
  frames**, including one stretch of **19 consecutive** (processed frames 430-448). Track spans:
  | | old (frozen clock) | new (clock advances) |
  |---|---|---|
  | | track 4: frames 358-739, len 362 | track 4: frames 358-**429**, len 71 |
  | | — | track 6: frames **450**-739, len 290 |
  One person is split across the empty stretch: last seen at 429, re-detected at 449-450, a
  ~21-update absence that BotSort could not bridge (Kalman drift plus `proximity_thresh=0.5`
  blocking the ReID match). With the frozen clock, frames 429 and 449 looked *adjacent* to BotSort,
  so it bridged them trivially. **The 5-identity baseline was correct for the wrong reason.**
  Not silently accepted — awaiting the user's call (see OQ-10).

- **VF-26 — Impact of the 1b fix is confined to clips that contain empty frames.** Measured:
  crowd sample and crowded sample2 have **0 empty frames** and are byte-identical before and after
  (787 dets / 117 ephemeral / 27 tracks / median 23; 2502 / 124 / 43 / median 40). office cctv has
  39 empty frames and changed (5→6 identities, median life 215→147). **On file playback the scene
  is rarely empty; a live camera on an empty corridor overnight is the pathological case, so the
  near-zero measured impact here must NOT be read as the fix being unnecessary — this is a
  Phase 2 correctness fix that happens to be cheap now.**

- **VF-10 — Snapshot banks saturate almost immediately.** Measured embedding updates per identity
  vs `MAX_SNAPSHOTS=8`: 302 on office cctv, up to 122 on crowded sample2. The overwritten file is
  **`snap_008.jpg`** (bank length is post-append, capped at 8), not `snap_007.jpg`.

---

## 4. OPEN QUESTIONS

- **OQ-1 — What deleted `data/reid_snapshots/`?** It held folders `1` and `2` at the start of the
  2026-08-27 session and was empty later the same session. No script in this repo deletes that
  path (the only `rmtree` is `agents/retriever.py` on `chroma_db`). Unexplained.

- **OQ-2 — ANSWERED (VF-14): no.** OSNet recovers only within ~30 updates (~2 s of video), while
  the Guardian attempted recovery up to `MAX_FRAMES_MISSING=450`. See OQ-6 for what follows.

- **OQ-6 — Long-gap ReID is now an unfilled capability gap.** Disabling the Guardian removed a
  broken *implementation* of a feature the system still needs: nothing now re-identifies a person
  who is occluded or leaves frame for more than ~2 s. The replacement is a proper long-gap gallery
  keyed on **OSNet** embeddings (not hand-rolled HSV/edge histograms), with a similarity threshold
  tuned against labelled pairs. Until then, expect a person who walks behind a pillar for 3 s to
  return as a new `Person_<id>` with reset intrusion persistence and lost face identity.
  **DECIDED: Phase 6 — reconfirmed post-1b (VF-27, K still 0 on office cctv).** Impact quantified: zero on sparse room footage, ~1 event/second on
  dense street crowds; office cctv is the target scenario and K=0 there.
  ⚠️ **This call rests on exactly ONE sparse clip (`office cctv.mp4`).** Re-run
  `scratchpad/reentry_gap_impact.py "<clip>.mp4"` (takes clip names as argv) against any new
  room-monitoring footage to firm it up. If K is non-zero on a second sparse clip, revisit.

- **OQ-10 — RULED (accept 6, rule rewritten).** The absence was 21 updates, **below** the removal
  boundary of 31, so the track was still alive: this is an **association** failure, not a removal
  failure. That rules out raising `TRACKER_TRACK_BUFFER` — VF-1 already established removal is not
  the binding constraint. Six is the honest count of what BotSort resolves on that clip with a
  correct clock; five was the count under a bug. VF-13 rewritten accordingly.

- **OQ-13 — Enrol someone who actually appears in `office cctv.mp4`, then re-run the re-entry
  study.** This is the single concrete step that would make **M meaningful for the first time**
  (VF-28 half 2). Today `data/authorized_faces/` holds only `Talha.jpeg`, who is in none of the
  benchmark clips, so every M figure ever recorded is structurally 0. Adding one enrolled face
  from that clip and re-running `scratchpad/reentry_gap_impact.py "office cctv.mp4"` would
  quantify the face-identity half of the Phase 6 risk. **Do not do this now** — logged so the
  cheap path to closing it is not lost. (Remember to delete `data/embeddings/face_embeddings.pkl`
  after changing the folder, or the new face will not be picked up.)

- **OQ-12 — Why did OSNet fail to re-associate at 21 updates here when VF-14 showed recovery at
  30 updates on this very clip?** A 21-update absence is well inside the range BotSort+OSNet
  should handle, so this is **not** a long-gap-gallery problem (OQ-6) — it is something else.
  Leading hypothesis from the VF-25 investigation: `proximity_thresh=0.5` gating the ReID match
  combined with Kalman drift across the absence, i.e. the person re-entered too far from the
  predicted box for appearance matching to even be consulted. **Not being chased now**, but this is
  a more tractable lead than the gallery and may be a cheap win. Note VF-14's recovery test held
  the target roughly in place; here the person left and re-entered.

- **OQ-11 — RESOLVED (VF-27).** Re-ran the re-entry study post-1b on all three clips. **K = 0 on
  office cctv still**, so the Phase 6 deferral stands. Counts did rise elsewhere (crowd sample K
  doubled 3 → 6), confirming the freeze had been undercounting. One residual: **M rose 0 → 1 on
  office cctv** — the target clip now has one event that would cost a face identity. Not enough to
  reopen the priority on its own, but it is no longer strictly zero-cost there.

- **OQ-9 — SPLIT-INTRUSION ATTRIBUTION DEFECT (does NOT belong to Phase 6).**
  Measured within VF-15: five re-entry events had `intrusion_persistence` **above**
  `PERSISTENCE_THRESHOLD=5` at track death (counters 31, 26, 41, 7, 5). That is a **confirmed,
  actively-logging intrusion being split across two `Person_<id>` entries in the event log** —
  distinct from the counter-reset case K_strict counts. It is not a missed detection, it is a
  **corrupted record**: one continuous event appears as two different people.
  This matters more than the counter reset because **`storage/event_logs.csv` is the seam between
  the vision half and the RAG analyst** — the analyst will narrate the two rows as two separate
  people, confidently and wrongly.
  **Mitigation is cheap and completely independent of the gallery:** when a track dies while
  `confirmed_tres` was true, write a log row recording that an intrusion was in progress at track
  death. The analyst then has evidence the two rows may be one event instead of silently believing
  they are two.
  **Do not absorb this into the Phase 6 gallery work.** The gallery would reduce its frequency but
  **never eliminate it**. Not yet implemented — deliberately deferred, not forgotten.

- **OQ-7 — Do gaps of 120/300 updates behave as predicted?** Not measured; neither benchmark clip
  has a contiguous single-person run long enough. Needs a purpose-shot or longer clip.

- **OQ-8 — Are BotSort's raw IDs clean on the crowd clips?** Verified only on office cctv (VF-13).
  The 40-row guardian-OFF sheet for crowded sample2 was not inspected at full resolution.

- **OQ-3 — Does any Kalman-predicted box actually reach `check_trespassing`?** VF-4 makes this
  unlikely, but it has not been asserted end-to-end. Must be verified by instrumenting a real run
  (assert every box reaching `check_trespassing` corresponds to a detection in this frame's YOLO
  output) before Task 5 is dropped.

- **OQ-4 — Are the remaining Guardian merges correct when it is enabled?** Contact sheets show
  clear over-merges; whether *any* of its merges are genuine recoveries has not been assessed.

- **OQ-5 — ANSWERED (VF-19).** Closing from above (`TRACKER_NEW_TRACK_THRESH` 0.6→0.45) is safe
  and halves ephemeral detections; closing from below (`CONFIDENCE_THRESHOLD` 0.45→0.6) **fails
  the VF-13 regression rule**. Knob is plumbed; **default left at 0.6 (behaviour preserved)**
  pending the user's choice of value.

---

## 5. Gotchas

**Configuration**
- Roughly **a third of `config.py` is read by nothing.** Dead keys include all eight
  `REID_EMBEDDING_*` / `REID_SNAPSHOT_INTERVAL` / `REID_SNAPSHOT_MIN_COUNT` /
  `REID_IDENTITY_MATCH_MARGIN` / `REID_MAX_EMBEDDINGS_PER_IDENTITY` / `REID_REASSIGN_GRACE_FRAMES`,
  plus `DEPTH_SCORE_THRESHOLD`, `MAX_INTERACTION_SCALE`, `MIN_INTERACTION_SCALE`,
  `COLOR_TRESPASSER`, `COLOR_DRAWING`, `ACTION_*` and the behaviour thresholds. **Grep before tuning.**
- **The ReID thresholds that actually matter are module constants in `vision/identity_guardian.py`**
  (`REID_SIMILARITY_THRESHOLD=0.82`, `MAX_FRAMES_MISSING=450`, `MAX_SNAPSHOTS=8`,
  `MIN_FRAMES_TO_ENROLL=2`, `FRAMES_MISSING_BEFORE_REID=1`, `MIN_CROP_H/W=60/25`), **not**
  the `REID_*` keys in `config.py`.
- `config.OPENAI_API_KEY = os.getenv("GOOGLE_API_KEY")` — the OpenAI key must be placed in a
  **`GOOGLE_API_KEY`** env var. Passing an explicit `None` suppresses langchain's own env fallback,
  so the analyst fails with `openai.OpenAIError` if only `OPENAI_API_KEY` is set.
- `FRAME_SKIP=2`, `INFERENCE_WIDTH=640`, `DEPTH_REFRESH_INTERVAL=5` and `PERSISTENCE_THRESHOLD=5`
  are hardcoded inside `run_surveillance`, not in `config.py`. `FRAME_SKIP` silently rescales the
  real-time meaning of **every** frame-count threshold in the system.

**Known defects not yet fixed**
- `st.session_state.identity_map` is never populated (`run_surveillance` builds a local dict), so
  the Heatmap tab always labels people `Person_<id>`.
- "Stop Surveillance" cannot stop the loop — `stop_btn` is bound once before the `while`.
- `run_surveillance` has no try/except and no `finally`; one bad frame aborts the run, leaks `cap`,
  and never sets `processing_complete`.
- Unbounded dicts, never pruned: `PoseAnalyzer.person_depth_filters` / `zone_depth_filters`,
  `IdentityGuardian._embeddings` / `_frame_count` / `_missing`, `st.session_state.track_positions`,
  and `data/reid_snapshots/` on disk.

**README.md inaccuracies** (fix when next touching it)
- Says the repeatedly-overwritten snapshot is `snap_007.jpg`; it is **`snap_008.jpg`** (VF-10).
- Records `boxmot==18.0.0` from package metadata, but the runtime banner prints `BoxMOT v17.0.0`.
- Describes `max_age=1200` as keeping tracks alive ~80 s; that was never true (VF-1).

**Environment**
- Activate with `.\venv311\Scripts\Activate.ps1` (PowerShell) or
  `source venv311/Scripts/activate` (Git Bash). Always run from the project root.
- CPU throughput is ~15 processed fps on 1080p footage.
- Benchmark clips live in `data/uploaded_videos/`. `crowd sample.mp4` is **greyscale**
  (measured mean HSV saturation 0.7/255); `crowded sample2.mp4` is full colour (79/255).

**Process**
- Debugging artifacts go in `scratchpad/` or `scripts/`, never into the package, and are never
  wired into `streamlit_app.py`.
- Measurement scripts and JSON dumps for this phase live in the session scratchpad:
  `baseline_reid.jsonl`, `after_1c.jsonl`, `guardian_*.json`, `contact_sheet_*.png`,
  `osnet_gap_test.json`, `reentry_gap_impact.json`.

**Reusable measurement scripts** (all take clip names as argv, all Guardian-flag aware):

| Script | Answers |
|---|---|
| `scratchpad/reentry_gap_impact.py "<clip>.mp4"` | How often the long-gap ReID hole bites: N re-entry events, M face-capable, K intrusion-counter-live-at-death. **Run this on any new room-monitoring footage** — the Phase 6 call rests on one clip (OQ-6). |
| `scratchpad/osnet_gap_test.py "<clip>.mp4"` | At what gap length BotSort+OSNet stops recovering a track ID. |
| `scratchpad/contact_sheet.py on\|off "<clip>.mp4"` | Visual audit of ReID merging; one PNG row per identity. Use `off` for the regression check in VF-13. |
| `scratchpad/guardian_ab.py` | Identity counts + track-lifetime distributions, Guardian ON vs OFF. |
| `scratchpad/deadband_ab.py` | Detections / ephemeral % / lifetimes across the three dead-band modes, with the VF-13 regression check built in. |
| `scratchpad/newtracks_045.py` | Isolates tracks that exist at one `new_track_thresh` but not another and scores them duplicate-vs-genuine. |
| `scratchpad/baseline_reid.py [out.jsonl]` | Guardian stats baseline across the three clips. |
