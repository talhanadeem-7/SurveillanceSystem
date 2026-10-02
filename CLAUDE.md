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
("theft"). Every decision is stored in SQLite through `storage/repository.py`, the persistence
interface to the second half of the system: a LangChain RAG agent ("Shelby Analyst") that
embeds event and activity narration into Chroma and answers natural-language questions with `gpt-4o-mini`.
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
| 2 — MiDaS module-level singleton | **done** | Lazy, thread-safe, keyed by (model type, device). Second run in-process: 3.34 s -> 0.015 s (VF-32). Per-run state unaffected. |
| 3 — Gate `align_zones` | **done** | **`ZONE_ALIGN_ENABLED=False` (ruled)**, `ZONE_ALIGN_INTERVAL=15`, `ZONE_ALIGN_NFEATURES=1000` (unchanged default). Interval counter inside `PoseAnalyzer`. Recompute failures now logged at WARNING with frame number + reason, counters surfaced in the run summary. Measured 33-74% speed-up (VF-36). Uncovered VF-37 (align_zones already broken on moving footage) and OQ-17. |
| 4 — Per-frame try/except in `run_surveillance` + `finally` release | **done** | Per-frame try/except (incl. `cap.read()`), consecutive-failure abort at 30, total-failure count reported, `cap.release()` + `processing_complete` in `finally`. `KeyboardInterrupt`/`SystemExit` re-raised. Verified against the real function with streamlit stubbed (VF-30). Carry-overs: OQ-14 (storage fatal branch), OQ-15 (`processing_complete` ambiguity). |
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

### What did NOT ship

Nothing from the original six-task scope remains unimplemented. Tasks 2, 3 and 4 were reached in a
later pass and are complete. Two items were deliberately **dropped** (below), and Task 6c
(a comment block marking the Guardian as a secondary ReID layer) is cosmetic and unstarted.

6. **Per-frame error handling** (Task 4). The main loop had no `try`/`except` and no `finally`, so
   one bad frame aborted a run, leaked the capture handle and left `processing_complete` unset.
   Now every frame is guarded; failures are logged with frame number and traceback and skipped;
   30 consecutive failures abort the run as systemic; the total skipped count is reported so a run
   that quietly lost 200 frames does not look clean; and `cap.release()` plus `processing_complete`
   sit in a `finally`. `KeyboardInterrupt`/`SystemExit` are re-raised, never swallowed.
7. **MiDaS loaded once per process** (Task 2). It was reloaded on every run via
   `PoseAnalyzer.__init__`. A lazy, thread-safe singleton keyed by (model type, device) cut a
   second consecutive run from **3.34 s to 0.015 s**, while per-run state stays per-instance.
8. **Zone alignment gated, then disabled by default** (Task 3). `align_zones` was the single most
   expensive stage in the pipeline: 33-45% of the frame budget, more per frame than YOLO inference.
   Gating behind `ZONE_ALIGN_INTERVAL` was implemented, but measurement then showed the feature
   should not run at all on the deployment target -- on fixed cameras it corrects 0.10 px of
   movement, and on moving footage it is actively harmful. `ZONE_ALIGN_ENABLED` now defaults to
   **False**. End-to-end throughput rose from **7.2 fps to ~10.6 fps**.

### What was measured and rejected

- **`CONFIDENCE_THRESHOLD` -> 0.6** to close the confidence dead band: fragmented the regression
  clip and discarded 56% of detections on a crowd clip (VF-19). Lowered `new_track_thresh` instead.
- **`FRAME_SKIP` -> 3**: reached live-camera real time on one clip of three, but dropped the
  tracker from 15 Hz to 10 Hz, moved the VF-13 split and shortened track life everywhere
  (VF-38/39). Wrong trade for an attribution system.
- **`ZONE_ALIGN_NFEATURES` -> 300**: 56% cheaper and accurate to 0.30 px, but the test footage has
  no genuine camera motion, so the result is inconclusive rather than positive (VF-40).

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

Verified at the final shipping defaults (Guardian off, alignment off, FRAME_SKIP=2):
`office cctv.mp4` -> **6 tracks**, split at frames 430-448 (track 4 ends 429, track 6 starts 450),
median life 147, 29 ephemeral (2.8%), 1050 detections.
`crowd sample.mp4` -> 787 dets / 117 ephemeral / 27 tracks / median 23.
`crowded sample2.mp4` -> 2502 dets / 124 ephemeral / 43 tracks / median 40.
Throughput at those defaults: 10.64 / 11.28 / 9.55 processed fps (VF-41).
Run `scratchpad/regression.py` after any change; it exits non-zero on drift.

### Open questions carried into Phase 2

Ranked by priority.

| Rank | ID | Question | Why it ranks here |
|---|---|---|---|
| 1 | **OQ-9** | **Split-intrusion attribution defect** -- a confirmed intrusion split across two `Person_<id>` rows. | **Corrupted record, not a missed detection.** The CSV is the seam to the analyst, which will narrate one event as two people, confidently and wrongly. Mitigation is cheap and needs no gallery. |
| 2 | **OQ-17** | `align_zones` accepts a successful-but-absurd homography. | Real correctness bug; **dormant only because alignment is now off by default**. Must be fixed before alignment is ever re-enabled. |
| 3 | **OQ-14 resolved** | Storage failures abort immediately. | Measured read-only failure and lock recovery; see VF-45. |
| 4 | **OQ-16** | Run ORB on the downscaled frame. | Highest-value untested optimisation, but demoted by the alignment default change -- only matters if alignment returns. |
| 5 | OQ-6 | Long-gap ReID gallery (OSNet-based). | Deferred to Phase 6; K=0 on target footage, but the face half of the risk is unmeasured (VF-28). |
| 6 | OQ-12 | Why OSNet failed to re-associate at 21 updates when it recovered at 30. | Cheaper and more tractable lead than the gallery. |
| 7 | OQ-13 | Enrol someone who appears in `office cctv.mp4`. | The one cheap step that makes M meaningful for the first time. |
| 8 | OQ-15 | `processing_complete` cannot distinguish "finished" from "stopped". | UI honesty; an aborted or interrupted run looks complete. |
| 9 | OQ-3 | Assert no predicted box reaches `check_trespassing`, then formally drop Task 5. | Closes a dangling task. |
| 10 | OQ-1, OQ-4, OQ-7, OQ-8 | Unexplained `reid_snapshots` deletion; Guardian merge correctness; gaps 120/300 unmeasured; full-res crowd sheet inspection. | Low. |

### Phase 1 in one paragraph

Phase 1 was scoped as a six-task hardening pass and delivered all six, but the most valuable
changes were not on the original list. Measurement repeatedly invalidated the premises behind the
planned work: the tracker knob everyone assumed governed track lifetime (`max_age`) turned out to
be inert, the appearance-based `IdentityGuardian` turned out to be merging visibly different people
rather than fixing ID switches, and the zone-alignment feature turned out to cost a third of the
frame budget while either doing nothing or throwing zones off-screen. What actually shipped was: a
fix for ephemeral track IDs that were colliding by construction and merging unrelated people's
identities, intrusion counters and trajectories; a fix for a tracker clock that froze whenever the
scene was empty, which needed changes in two files because the obvious one was unreachable;
closure of a confidence dead band that had made 31% of detections invisible to the event pipeline;
the `IdentityGuardian` disabled behind a flag after contact sheets showed it collapsing five
visibly distinct people into one identity; per-frame error handling so a single bad frame can no
longer abort a run or leak the capture handle; and two performance fixes that raised end-to-end
throughput from 7.2 to about 10.6 fps. Every claim in this file is backed by a measurement rather
than by reading code, several widely-believed numbers were corrected along the way, and the system
still does not keep up with a live 30 fps camera -- which is the binding constraint on Phase 2.

---|---|---|
| **OQ-9** | **Split-intrusion attribution defect** — a confirmed intrusion split across two `Person_<id>` rows corrupts the CSV the analyst reads. Cheap mitigation, independent of any gallery. | **highest — corrupted record, not a missed detection** |
| OQ-6 | Long-gap ReID gallery (OSNet-based). Deferred to Phase 6; K=0 on target footage. | Phase 6 |
| OQ-12 | Why OSNet failed to re-associate at 21 updates when it recovered at 30. Likely `proximity_thresh` + Kalman drift. Cheaper lead than the gallery. | medium |
| OQ-13 | Enrol someone who appears in `office cctv.mp4` — the one step that makes M meaningful. | medium, cheap |
| OQ-3 | Assert no predicted box reaches `check_trespassing`, then formally drop Task 5. | low |
| OQ-1 | What deleted `data/reid_snapshots/`. | low, unexplained |
| OQ-4, OQ-7, OQ-8 | Guardian merge correctness; gaps 120/300 unmeasured; full-res crowd sheet inspection. | low |

---

## 2c. PHASE 2 PLANNING CONSTRAINTS

**The pipeline does not currently keep up with a live camera. This is the binding constraint on
Phase 2, and it must not be buried in a benchmark table.**

Measured (VF-31): a full run of `office cctv.mp4` through the real `run_surveillance()` processes
**7.2 frames per second** end to end. At `FRAME_SKIP=2`, a 30fps camera produces 15 frames per
second that need processing, so the system runs at **roughly half real time**. Against a live
source it would fall behind continuously — the backlog grows for as long as the camera is on.

The ~15 fps figure quoted previously is **detection and tracking only**. Roughly half the frame
budget goes to everything else (zones, depth, faces, drawing). Any Phase 2 live-camera plan needs
one of: a per-stage optimisation pass (start from the profile), a higher `FRAME_SKIP`, GPU
inference (currently `torch-2.11.0+cpu`), or an explicit frame-dropping policy that keeps latency
bounded instead of queueing.

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

- **VF-30 — Task 4 per-frame error handling verified against the real `run_surveillance()`.**
  Measured by calling the shipped function with `streamlit` stubbed out (not a copy of the loop),
  on `Demo_MIdas.mp4`:
  | Scenario | Raised | `processing_complete` | UI result |
  |---|---|---|---|
  | clean run | None | True | 1 success |
  | 8 transient frame failures | None | True | warning "8 frame(s) were skipped", then success |
  | permanent failure | None | True | error "Aborted after 30 consecutive frame failures at frame 60" |
  | KeyboardInterrupt | **KeyboardInterrupt** | True | propagated; `finally` still ran |
  Abort fires at exactly `MAX_CONSECUTIVE_FAILURES=30`; the `finally` executes on every path.
  Detection/tracking numbers **byte-identical** on all three clips afterwards, and VF-13 passes
  (6 tracks, split at 430-448) — Task 4 touches only `streamlit_app.py`.

- **VF-31 — ⚠️ END-TO-END THROUGHPUT IS 7.2 fps, NOT 15. The system does not keep up with a live
  camera.** Measured: a full `office cctv.mp4` run through the real `run_surveillance()` took
  **739 processed frames in 102.8 s = 7.2 fps**. The often-quoted **~15 fps is detection-and-
  tracking ALONE** (the baseline harness, which calls `PoseDetector` directly and skips zones,
  depth, faces and drawing). **Roughly half the frame budget is spent outside YOLO and the
  tracker.** At `FRAME_SKIP=2` a 30fps source needs 15 processed fps to stay real-time, so at
  7.2 fps the pipeline runs at **about half real time**. See section 2c.

- **VF-32 — MiDaS singleton: 3.34 s → 0.015 s on a second run in the same process (223x).**
  Measured before/after in one process: old behaviour (a `torch.hub.load` pair per
  `PoseAnalyzer`) cost 4.78 s then 3.34 s; with the singleton, construction costs 3.38 s once
  then 0.015 s. Verified the model and transform objects are shared (`is` identical) while
  per-run state (ORB reference, Kalman dicts) stays distinct per instance.

- **VF-33 — `person_depth_filters` growth is negligible on file playback.** Measured over a full
  `office cctv.mp4` run with one restricted zone: **13 entries**, from 10 distinct person ids
  (5 of them ephemeral/negative), 1-2 keys each. Bound is
  *(ids that reach a zone) × (zone types) × (≤2 keypoint indices)*, each entry a `KalmanSmoother`
  of four floats. Unbounded only under a long-lived live camera; not worth fixing for Phase 1.

- **VF-34 — PER-STAGE PROFILE. `align_zones` is the single largest stage, costing more per frame
  than YOLO inference.** Measured with manual timers around each stage in a full real
  `run_surveillance()` run (cProfile attributes into torch/cv2 internals and cannot answer
  "which stage costs what"). st.image() was stubbed, so browser-render cost is excluded.
  | Stage | office cctv (1080p) | crowded sample2 (640x480) |
  |---|---|---|
  | **align_zones (ORB+RANSAC)** | **35.18 s — 33.5% — 47.61 ms/call** | **16.07 s — 28.8% — 47.83 ms/call** |
  | YOLO inference | 24.68 s — 23.5% — 33.40 ms | 12.54 s — 22.4% — 37.31 ms |
  | tracker (incl. OSNet) | 18.16 s — 17.3% | 15.12 s — 27.0% |
  | — of which OSNet embedding | 16.91 s — 16.1% | 13.50 s — 24.1% |
  | face identify_person (DeepFace) | 6.61 s — 6.3% | 1.72 s (1 call = lazy model load) |
  | MiDaS depth | 4.80 s — 4.6% | 2.62 s — 4.7% |
  | zone geometry / theft / drawing / logging | <3.5% combined | <4% combined |
  | unattributed (read, resize, box rescale, loop) | 12.18 s — 11.6% | 5.04 s — 9.0% |
  **Key sub-finding: the align_zones cost is RESOLUTION-INDEPENDENT** — 47.61 ms at 1920x1080 and
  47.83 ms at 640x480. The time is in `BFMatcher(NORM_HAMMING, crossCheck=True)`, which is O(n²)
  in `nfeatures`, not in feature detection. See VF-35 and OQ-16.

- **VF-35 — `ZONE_ALIGN_NFEATURES` is a real lever, but there is a fixed floor; "an order of
  magnitude cheaper" was WRONG.** Measured on office cctv (1080p, 73 sampled frames):
  | nfeatures | ms/call | vs 1000 | homography failures | corner error vs n=1000 (mean / p95 / max) |
  |---|---|---|---|---|
  | 1000 | 54.50 | — | 0/73 | — |
  | 600 | 33.32 | −39% | 0/73 | 0.23 / 0.56 / 3.52 px |
  | 300 | 24.02 | −56% | 0/73 | 0.30 / 0.74 / 3.36 px |
  Fitting `cost = a + b·n²` gives **a ≈ 20.9 ms fixed** (ORB detection + grayscale on 1080p) plus
  ~34 ms of quadratic matching at n=1000; the model predicts 55.3 ms vs 54.5 measured. So the
  matcher is quadratic as claimed, but the fixed floor means 1000→300 cuts features 3.3× and time
  only 2.3×. Accuracy holds well under a pixel at both lower settings.
  ⚠️ Measured on 1080p only; the fixed floor scales with frame area, so proportions differ at
  lower resolution. **RULED: stays at 1000.** The evidence was INCONCLUSIVE, not negative — see
  VF-40. Largely moot now that `ZONE_ALIGN_ENABLED` is False; kept logged in case alignment is
  re-enabled.

- **VF-36 — Gating `align_zones` delivers a 33-74% end-to-end speed-up.** Measured end-to-end
  through the real `run_surveillance()` (st.image stubbed, so excluded equally from all rows):
  | Clip | interval 1 | **interval 15** | disabled |
  |---|---|---|---|
  | office cctv | 6.99 fps (align 33.7%) | **11.37 fps (align 3.8%)** | 11.86 fps |
  | crowd sample | 5.67 fps (align 44.7%) | **9.89 fps (align 5.3%)** | 10.92 fps |
  | crowded sample2 | 6.93 fps (align 32.1%) | **9.19 fps (align 3.2%)** | 9.13 fps |
  Interval 15 captures most of what disabling gives (office cctv 11.37 vs 11.86). On crowded
  sample2, interval 15 and disabled are identical within noise. Projection from VF-34 was
  10.2 fps for office cctv; actual 11.37, i.e. the projection was conservative.

- **VF-37 — ⚠️ `align_zones` IS ALREADY BROKEN on moving/crowded footage, and it fails SILENTLY.**
  Measured frame-to-frame zone-corner movement under the ORIGINAL every-frame behaviour
  (interval 1), with a central 30% test zone:
  | Clip | mean | p95 | max | frames with zone thrown off-screen | recompute failures |
  |---|---|---|---|---|---|
  | office cctv (fixed camera) | **0.10 px** | 1.00 px | 4.00 px | 0/739 | 0 |
  | crowd sample | **198.96 px** | 845 px | 9629 px | **9/170** | 0 |
  | crowded sample2 | **51.25 px** | 188 px | 5201 px | **3/336** | 0 |
  On the crowd clips ORB matches frame 0 against a scene dominated by moving people, RANSAC finds
  a "consensus" from moving points, and the zones are hurled thousands of pixels — sometimes
  entirely out of frame. **Zero recompute failures**: the homography SUCCEEDS and is garbage.
  This is pre-existing, not caused by gating. It also means the Task 3(d) WARNING does NOT catch
  the failure mode that actually occurs — see OQ-17.
  **Consequence: the interval-15 "zone drift" figures on the crowd clips (mean 298 px / 29 px)
  are measured against a garbage reference and say nothing about gating.** The only trustworthy
  drift figure is office cctv: **mean 0.16 px, p95 1.00 px, max 3.00 px** — gating is safe there.

- **VF-38 — No configuration reaches live-camera real time at FRAME_SKIP=2; only the target clip
  does at FRAME_SKIP=3.** Measured end-to-end, `source_fps = processed_fps x FRAME_SKIP`
  (threshold for a 30fps camera is 30):
  | Clip | FS=2 processed / source | FS=3 processed / source | real time at FS=3? |
  |---|---|---|---|
  | office cctv | 9.79 / **19.58** | 10.64 / **31.93** | **YES** |
  | crowd sample | 10.30 / **20.61** | 9.29 / **27.86** | no |
  | crowded sample2 | 9.24 / **18.49** | 8.41 / **25.23** | no |
  Note processed_fps does not rise proportionally at FS=3 (and falls on two clips), because
  `cap.read()` still runs on every source frame regardless of skipping.

- **VF-39 — FRAME_SKIP=3 changes the regression materially; VF-13 would need restating.**
  Measured (FS=2 baseline in brackets): office cctv dets 699 [1050], ephemeral 14 [29],
  **tracks 6 [6]**, median life 100 [147], **split moves to track4 ending 286 / track6 starting
  342** [429/450]. crowd sample 516 dets [787], 25 tracks [27], median 15 [23].
  crowded sample2 1672 dets [2502], 37 tracks [43], median 31 [40].
  Track count on the regression clip happens to stay 6, but every other number moves and the
  split frames change, so **the VF-13 rule as written would fail at FS=3**.
  **RULED: FRAME_SKIP stays 2. FS=3 measured and REJECTED — do not re-propose without new
  evidence.** It reaches real time on one clip of three, and buys that by dropping the tracker
  from 15 Hz to 10 Hz. For a system whose purpose is attributing events to specific people, that
  is the wrong trade. Rationale is also recorded in the code comment at `FRAME_SKIP`.
  **Separate finding: `cap.read()` runs on EVERY source frame regardless of skipping**, so
  FRAME_SKIP saves inference time but not decode time. Decode is a Phase 2 reader-thread problem,
  not a FRAME_SKIP tuning problem.

- **VF-40 — The nfeatures=300 outliers correlate WEAKLY with camera motion, and office cctv has
  too little motion to settle it.** Re-measured densely (369 frames, every 2nd processed, vs 73
  before): corner error mean 0.50 px, p95 1.04 px, **max 4.65 px** — denser sampling found worse
  outliers than the first pass. Of 19 outliers at/above p95, outlier median |translation| is
  **4.19 px vs 2.76 px** for non-outliers, rotation 0.029 deg vs 0.015 deg.
  Pearson r(corner_err, translation) = **+0.338**, rotation +0.241, scale +0.267.
  So the correlation is real but weak (motion explains ~11% of variance), **and the entire motion
  range in this clip is 2-6 px of translation and <0.1 deg of rotation** — a fixed camera with
  jitter. **This does NOT answer whether nfeatures=300 degrades under genuine camera motion**,
  because no such motion exists in the test footage.
  **RULED: nfeatures stays 1000.** Recorded as inconclusive, not negative: a future test needs
  footage from a camera that genuinely moves, and per VF-37 that footage first needs OQ-17's
  sanity check or alignment is garbage there regardless of feature count.

- **VF-41 — SHIPPING CONFIGURATION THROUGHPUT (alignment disabled, FRAME_SKIP=2).** These are the
  numbers that reflect what actually ships; VF-36's interval-15 row does not.
  | Clip | processed fps | source fps | vs old every-frame alignment |
  |---|---|---|---|
  | office cctv | **10.64** | 21.29 | 1.52x |
  | crowd sample | **11.28** | 22.56 | 1.99x |
  | crowded sample2 | **9.55** | 19.09 | 1.38x |
  End-to-end throughput therefore rose from **7.2 fps (VF-31) to ~10.6 fps** on the target clip.
  Still below the 30 source-fps live-camera threshold (see section 2c) — this is a speed-up, not
  a solution to real time.

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

- **OQ-17 — ⭐ HIGHEST PRIORITY BEFORE ALIGNMENT IS EVER RE-ENABLED. `align_zones` needs a sanity
  check on the computed homography, not just on failure.** PROMOTED: this is a real correctness
  bug, not an optimisation — the homography succeeds, reports no error, and is nonsense, so zones
  silently sit in the wrong place and intrusion decisions are made against them. **Dormant, not
  fixed:** `ZONE_ALIGN_ENABLED` is now False by default, which is the only reason it does not
  block Phase 1. ⚠️ **The Task 3(d) WARNING catches the WRONG failure mode** — zero recompute
  failures occurred on any clip; the failure is a successful-but-absurd homography.
  VF-37 shows the real failure mode is a homography that succeeds and is garbage (zones thrown
  off-screen with zero recompute failures). The Task 3(d) WARNING only fires when computation
  FAILS, so it catches the case that does not happen and misses the one that does. A cheap guard:
  reject a new M if it moves the zone corners more than some fraction of the frame, or if its
  scale/shear is implausible, and keep the previous homography instead — with a WARNING. Needs a
  threshold chosen against measurement, not guessed. **Not implemented; Task 3 was closed on
  its specified scope.**

- **OQ-16 — ⭐ HIGHEST-VALUE UNTESTED OPTIMISATION: run ORB on the downscaled frame.**
  `run_surveillance` already builds a 640-wide copy for YOLO (`infer_frame`), but calls
  `analyzer.align_zones(frame, zones)` with the **full-resolution** frame. VF-35 showed the
  align_zones cost splits into ~21 ms of fixed ORB detection + grayscale (which scales with frame
  **area**) plus ~34 ms of O(n²) matching. On office cctv the area ratio is
  1920×1080 → 640×360, i.e. **9× fewer pixels**, so the fixed component should fall to roughly
  2-3 ms. A homography is only a linear map, so the result scales back to full resolution with a
  straightforward coordinate transform (conjugate M by the scale matrix).
  **This is orthogonal to both existing levers** — it attacks the fixed floor that
  `ZONE_ALIGN_NFEATURES` cannot touch, and it compounds with `ZONE_ALIGN_INTERVAL` gating.
  Potentially better than either.
  ⚠️ **RISK, and why a timing number alone is not enough:** downscaling reduces the number of
  distinguishable ORB keypoints, so match quality may degrade in a way the nfeatures sweep does
  **not** predict — fewer pixels is not the same as fewer requested features. Any test must repeat
  the **corner-reprojection accuracy check** from VF-35 (reproject zone corners through the
  downscaled-and-rescaled homography, compare in pixels against the full-resolution result), not
  just report ms/call. Deliberately **not implemented** in Task 3, which was closed on its
  specified scope.

- **OQ-14 - RESOLVED (2026-09-26, VF-45).** Repository failures raise
  `StorageFatalError`; the real pipeline catches this before its generic frame handler.
  A read-only SQLite write aborted after one detector call, with zero frame failures,
  a storage-specific UI error, and persisted `runs.status = aborted`. Persistent-lock
  finalization uses a local recovery record; it cannot update an unwritable database
  until access returns. Actual lock/recovery and read-only tests cover both cases.

- **OQ-15 - Run outcome can now be read from `runs.status`.** The database records
  completed, aborted, or interrupted, with frame counters and end time. Current
  `processing_complete` retains the EOF-only behavior introduced by the CPU work.
  A richer Analyst/history UI can now distinguish outcomes using the persisted status;
  that UI work is deferred. The historical Task 4 always-True behavior is superseded.

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
- **VF-42 — The VLM was never wrong; the LABEL WAS STALE.** Reported as "sitting shown as walking".
  Measured on `sitting.mp4` (607 frames, **60 fps**, 10.1 s): the shipped gating ran only **3**
  analyses for the entire clip, each covering **under 1 s of video**, with **4 s unanalysed between
  them**. The trigger frames were dumped and inspected: at frame 30 and frame 270 the person really
  **was** walking toward the chair; the frame-510 sitting analysis never landed before the run
  ended. The two calls in the user's log (01:57:38, 01:57:49) are exactly those two walking
  moments. Proof the model is accurate: fed the sitting frames directly, `gpt-4o-mini` returned
  `sitting` at **confidence 1.0** — and returned `sitting` **with the colour bug still present**,
  so the colour bug was not the cause.
  Root causes, all fixed: (a) a **second freshness gate** in `analyze_if_ready`
  (`current_frame_id - cached.frame_ids[-1] < interval*2`) that **doubled** the effective interval
  on top of `should_analyze`, which already rate-limited on the same clock; (b) results **never
  expired**, so a verdict displayed forever; (c) only one analysis per track in flight, so a ~10 s
  API call blocked refresh for 10 s of wall time.
  After the fix, same clip: **20 analyses, one every 0.5 s of video**. Labels through the shipped
  path now track reality — t=1.5 s `walking` (0.9), t=4.5 s `interacting_with_object` ("reaching
  for a chair", 0.8), t=9.5 s `sitting` (0.9).
- **VF-43 — Two further VLM defects, both measured.**
  1. **Colour channels were double-swapped in every image ever sent.** `cv2.cvtColor(BGR2RGB)`
     followed by `cv2.imencode`, which itself expects BGR. Measured: a centre patch of BGR
     `[171,190,201]` arrived as `[201,190,171]`. Skin rendered blue. Fixed by passing BGR straight
     to `imencode`.
  2. **Payload was ~3.26 MB per call** (full-frame lossless PNG at source resolution). ⚠️ Cropping
     to the person **made it worse, not better** — 3.95 MB — because the clip is portrait and the
     crop is barely smaller while PNG compresses a detailed person crop poorly. *An earlier claim
     that cropping would cut payload ~30x was wrong and was corrected by measurement.* What
     actually worked: downscale to `VLM_ACTIVITY_MAX_IMAGE_SIDE=512` + JPEG q85 →
     **0.280 MB mean, 0.375 MB max (11.6x smaller)**. This matters for correctness, not just cost:
     upload latency is what bounds label freshness.
- **The frames sent are now a padded CROP of the tracked person**, not the whole scene. The prompt
  names a track id, but a full frame carried nothing identifying which person that id was — with
  two people in shot the answer was unattributable. `add_frame` now takes `bbox`.
- **`VLM_ACTIVITY_MIN_TRACK_AGE` is now read** (it was dead config). `VLM_ACTIVITY_MIN_MOVEMENT`
  is **still dead** — documented as intent, not a live knob.
- ⚠️ **`VLM_ACTIVITY_ANALYSIS_INTERVAL` is in SOURCE frames, so its real-time meaning depends on
  clip frame rate.** The config comment says "~2 seconds at FRAME_SKIP=2", which assumes 30 fps
  source. On this 60 fps clip the same value is **0.5 s**. Not fixed; flagged.
- **VLM persistence and Analyst integration are wired.** Observations are saved through
  `ActivityObserver` (VF-50). `LogAnalyzer` now narrates confidence-filtered activity
  spans alongside security events for Chroma ingestion (VF-51).

**action_recognizer removal (2026-09-01)** — `vision/action_recognizer.py` itself was already
deleted in commit `760a6c2`; the residue it left behind has now been removed:
- `config.py`: `ACTION_MODEL_PATH`, `ACTION_WINDOW`, `RUNNING_VELOCITY_THRESHOLD`,
  `BENDING_RATIO_THRESHOLD`, `PICKUP_HAND_KNEE_RELATION` (removed in the working tree before
  this pass; README updated to match).
- `utils/csv_utils.py`: the whole `"Behavior"` narration branch and its `last_action_per_entity`
  state machine, which existed solely to narrate the recognizer's Walking/Sitting/Bending/
  Picking Up/Running rows. **Verified dead by measurement, not by reading**: `storage/event_logs.csv`
  contains 0 `Behavior` rows (Action column is Intrusion 195 / Identity 67 / Access 37 / Theft 15 /
  Removal 4) and 0 rows with any of those Status verbs. `get_all_logs_formatted()` returns the same
  318 documents before and after.
- `streamlit_app.py`: `behavior_logged_ids`, `logged_general_ids`, `track_ids_logged_general` —
  all three were written to and never read. Tuple arity of `setup_surveillance_memory` went 11 → 8;
  verified by AST that the unpack site and the `return` still match.
- `agents/reasoning_agent.py`: prompt no longer cites "Walking, Sitting" as example log kinds.
- **`app.py` still contains all three dead sets** (lines 58-60, 193-194). Left untouched
  deliberately — `app.py` is the out-of-scope legacy desktop variant.

**Corrections to the record** (claims that were asserted and turned out to be untrue)
- **"A comment noting `person_depth_filters` unbounded growth was added in Task 1c."** Never true.
  Task 1c did not touch `vision/pose_analyzer.py` at all; the note existed only in this file's
  Gotchas. A real code comment was added in **Task 2**. Logged because the claim was stated as
  established fact and would have been trusted.
- **"Pipeline throughput is ~15 fps."** True only for detection-and-tracking in isolation.
  End-to-end is **7.2 fps** (VF-31).

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
| `scratchpad/regression.py` | **Run after every change.** The standing Phase 1 regression: VF-13 (office cctv = 6 tracks, split at 430-448) plus ephemeral counts and median track life on all three clips. Exits non-zero on any drift. |
| `scratchpad/deadband_ab.py` | Detections / ephemeral % / lifetimes across the three dead-band modes, with the VF-13 regression check built in. |
| `scratchpad/newtracks_045.py` | Isolates tracks that exist at one `new_track_thresh` but not another and scores them duplicate-vs-genuine. |
| `scratchpad/baseline_reid.py [out.jsonl]` | Guardian stats baseline across the three clips. |


## CPU performance validation ? 2026-09-23

This update supersedes earlier CPU throughput/configuration notes above. See
`docs/CPU_PERFORMANCE.md` for the measured table and reproduction steps.

- USE_VLM is now the canonical switch (old name is an alias). Disabled-mode tests
  verify no API client, executor, or crop buffering. Generation guards reject old
  callbacks; EOF drains for up to three seconds. All three live-API sample runs
  finished without outstanding requests.
- Same-weight FP32 ONNX reduced isolated pose prediction from 41.46 to 25.00 ms.
  Full Streamlit steady analyzed FPS, VLM off: sample12 11.72 -> 14.60, sample11
  10.08 -> 11.84, sample20 13.34 -> 16.87. Ryzen 5 5600, CPU only, FRAME_SKIP=2.
  No profiler in these comparisons; browser rendering/RTSP not measured.
- Complete sample comparisons preserved detection counts and all IDs on 901
  analyzed frames. ONNX coordinate differences were below 0.002 pixels.
- PyTorch versus ONNX/fused OSNet regression: office cctv retains six tracks
  and 29 ephemeral detections; crowd sample 27/117; crowded sample2 43/124.
  All per-frame IDs and lifetime distributions matched.
- OSNet fusion, cached zone statistics/reference edges, vectorized theft grids,
  grab/retrieve skipping, independent preview throttling, and bounded histories
  are implemented. Models/thresholds/ReID cadence were preserved. CPU threads
  are configurable; four won the local small-model sweep by a small margin.
- 22 unit tests passed. Final same-session sample12 rerun check passed twice
  (209 analyzed frames per run, no errors, disabled VLM has no executor).
- app.py and chatbot/RAG execution remained excluded. Analyst startup is now
  explicit; repeated surveillance starts use fresh tracker/identity/zone state.
- Still an uploaded-file UI. Live RTSP capture/reconnect/backpressure and a
  responsive independent stop control remain work for deployment.
- Test scripts/logs/crops/JSON folders were removed after validation. Generated
  ONNX deployment weights and manifest remain in data/models (Git-ignored).

## SQLite storage migration - 2026-09-26

### Audit before implementation

Production references, excluding virtual environments and the legacy `app.py`:

- `config.LOG_PATH` named `storage/event_logs.csv`.
- `storage/event_logger.py`: created the header and appended event rows;
  `log_event()` called its writer after the existing debounce check.
- `streamlit_app.py`: two direct identity-writer calls, six `log_event()` call
  sites for zone/asset events, and an existence check to offer existing logs.
- `utils/csv_utils.py`: `LogAnalyzer.load_data()` used `pandas.read_csv()`.
  `agents/retriever.py` consumed `get_all_logs_formatted()` and wrapped each
  narration string in one LangChain Document for Chroma. `reasoning_agent.py`
  reached these documents through the retriever.
- `storage/activity_observer.py`: created/appended its separate CSV, but had
  no caller in the pipeline. No production reader or writer referenced
  `video_metadata.csv`; the file was zero bytes with no header.
- Actual event columns: `Timestamp,Entity,Action,Status,Location`. Actual activity
  columns: `Timestamp,Track_ID,Activity_Label,Description,Confidence,Frame_ID`.
  Event data had 338 rows; activity and metadata data had zero rows.

### Implementation and operation

`storage/db.py` owns SQLAlchemy schema/engine setup; `storage/repository.py`
owns all data access. SQLAlchemy imports are confined to those modules.
`DATABASE_URL` defaults to `sqlite:///storage/shelby.db`. SQLAlchemy 2.x is the
only added dependency. SQLite connection setup enables WAL and foreign keys;
data operations use portable SQLAlchemy expressions. PostgreSQL execution was
not tested; deployment there also requires its DBAPI driver.

Tables: `cameras`, `runs`, `zones`, `persons`, `events`, `activity_observations`,
plus historical `csv_imports` receipts retained in the existing database.
Fresh databases no longer create the retired import-receipt table. Run status has a database constraint;
person track IDs are unique within a run. Events index `(run_id, timestamp)`
and `action`. Event details retain original entity/location text independently
of later person display-name changes. Zone geometry excludes image patches.
Runs also carry JSON metadata so unspecified legacy metadata can be retained.

Uploaded videos register cameras; executions create separate runs. Events and
VLM observations commit at their frame boundary. Person-only updates commit
every 30 source frames; finalization flushes remaining work. The observer
deduplicates polled VLM results and saves final drained results. `frames_skipped`
counts failed analysis frames, not deliberate FRAME_SKIP downsampling.

`LogAnalyzer(run_id=...)`, `LogRetriever(run_id=...)`, and
`SecurityAnalyst(run_id=...)` optionally restrict analysis to a run. Their default
remains all events. Run-specific Chroma stores are separate from the all-run store.
Runtime production code no longer reads CSVs. The later cleanup removed
`LOG_PATH`, renamed the analyst utility to `utils/log_analyzer.py`, and retired the importer. `app.py`, all vision modules, thresholds and models were
unchanged. No API layer was added. Heatmap positions remain in session state.

Run verification from the project root (the completed importer was retired during cleanup):

```powershell
.\venv311\Scripts\python.exe -m unittest discover -s tests -v
.\venv311\Scripts\python.exe scripts/verify_storage_pipeline.py --label check
```

The completed migration preserved legitimate duplicate rows, committed each file
atomically with its receipt, and refused to re-import a changed previously imported
file. The source CSVs were subsequently deleted at the user's request after exact verification. Legacy rows have no video provenance, so they belong to explicitly labelled
legacy runs with unknown completion (`interrupted`), not invented video matches.

### Measured results

- **VF-44 - Migration and narration preserved exactly.** Actual migration imported
  **338** events: Intrusion **199**, Identity **75**, Access **41**, Theft **18**,
  Removal **5**. A second invocation imported **0**. All **338** narration strings
  were byte-for-byte identical to the pre-change formatter output, not merely a
  sampled semantic comparison. Example before and after:
  `At 2026-07-15 21:37:15, Person_1 was detected intruding at the Laptop.`
- **VF-45 - Fatal storage errors abort immediately; lifecycle is tested.**
  `tests/test_storage.py` executes the shipped `run_surveillance()` AST with UI
  and vision dependencies stubbed. An actual SQLite read-only write failed on
  the first detector call: **0** frame errors, storage-specific UI error,
  `processing_complete=False`, and `runs.status=aborted`. Restoring write access
  after that failure allowed terminal status persistence. A separate actual
  write-lock test verified rollback, pending status recovery after unlocking,
  and no persisted failed event. Completed, user-stop, KeyboardInterrupt, and
  exactly 30 ordinary frame-failure paths were also tested. Final full suite:
  **36 tests passed**, including **14** storage tests and all **22** existing tests.
- **VF-46 - Full office pipeline storage comparison.** Real detection, tracking,
  depth, face matching, zone handling and preview preparation ran through
  `run_surveillance()`; Streamlit rendering was stubbed and VLM disabled equally.
  Both runs analyzed **739 / 1479** source frames without frame errors.

  | Storage | Loop seconds | Analyzed FPS | Change from CSV |
  |---|---:|---:|---:|
  | CSV baseline | 52.4128 | 14.0996 | baseline |
  | Initial every-frame DB commit | 54.9296 | 13.4536 | -4.582% |
  | Final event/frame + person batching | 54.0301 | 13.6776 | **-2.9933%** |

  The final result narrowly meets the requested 3% bound. These are individual
  local runs, not confidence intervals; browser transport and live VLM were not
  benchmarked. Evidence JSON/logs remain under the local ignored `scratchpad/`.
- **VF-47 - Tracking regression retained after migration.** The historical
  `scratchpad/regression.py` was absent in this checkout. The retained
  `scripts/verify_storage_pipeline.py` therefore measures the real pipeline and
  asserts the recorded reference counts/lifetimes without changing vision code.

  | Clip | Detections | Ephemeral | Tracks | Integer median track life |
  |---|---:|---:|---:|---:|
  | office cctv | 1050 | 29 | 6 | 147 |
  | crowd sample | 787 | 117 | 27 | 23 |
  | crowded sample2 | 2502 | 124 | 43 | 40 |

  Office before/after counts and every track's first/last frame and lifetime
  matched exactly. Track 4 still ends at **429** and track 6 starts at **450**;
  the required split across **430-448** is preserved. Both crowd assertions also
  passed with no UI errors. Use `--clip "crowd sample.mp4"` or
  `--clip "crowded sample2.mp4"` to reproduce those checks.

### Limitations and follow-up

OQ-14 is resolved as measured above. An unwritable database cannot itself store
`aborted`: if finalization remains blocked, a per-database JSON recovery record
is written under `storage/pending_runs/`, then applied by the next repository
opening after access returns. If even that filesystem is full/unwritable, the
status cannot be guaranteed durable and the UI reports failure. This is not a
claim that a locked/read-only database was updated while still unwritable.

OQ-15's richer UI can now use `runs.status`; heatmap persistence remains deferred.
The initial migration left CSVs untouched; the later cleanup removed them after
verification. This verification did not invoke paid embedding/chat
or VLM APIs, so live Chroma ingestion and live activity API behavior were not
retested; event narration and the database observer were tested locally.

## CSV retirement and persistence audit - 2026-09-26

At the user's explicit request, removed all three legacy storage CSVs and the
temporary baseline CSV, retired `scripts/migrate_csv_to_db.py` and its receipt
helpers/model and migration-only tests, removed unused `LOG_PATH`, and renamed
`utils/csv_utils.py` to `utils/log_analyzer.py` with all active imports updated.
Historical import receipt rows remain in the existing database as audit data.
The legacy `_write_to_csv_and_terminal` alias remains solely because untouched
`app.py` calls it; it delegates to the database writer and performs no CSV I/O.

- **VF-48 - Source data safe to retire.** Before deletion, all **338** source rows
  matched the database's preserved legacy rows exactly; all three import-receipt
  hashes matched their source files. A SQLite backup was saved at the local ignored
  `scratchpad/pre_csv_cleanup.db`. After cleanup, every row of every existing table
  matched that backup, SQLite integrity returned `ok`, and foreign-key checks
  returned zero violations. All **338** narration documents still matched the
  original baseline without any CSV files present.
- **VF-49 - New-run persistence verified by readback.** A fresh real office-video
  execution in isolated `scratchpad/cleanup_verified.db` persisted **1 completed
  run, 1 zone, 6 persons, and 10 events**. Every emitted event's entity/action/status/
  location matched its stored record in order. Camera, person and zone links,
  rectangle geometry, frame numbers and run counters passed assertions. The video
  retained **1050 detections / 29 ephemeral / 6 tracks**, with no frame errors.
  VLM was disabled for this real-video check, so it produced **0 activities**.
- **VF-50 - Field and activity persistence tests.** Final suite: **37 passed**
  (**15 storage**, **22 existing**). Reopening a separate test database preserved
  camera/run fields, rectangle/polygon geometry, enrolled person identity and
  timestamps, all five event types, details, snapshot-path values, activity label,
  description, confidence, timestamps and sampled frame IDs. Pipeline tests also
  saved activity results received during shutdown and for a no-longer-detected
  track before inactive cleanup removed its cache entry. The latter revealed a
  potential gap in the original integration; completed activities are now collected
  at every frame boundary, independent of current detections. These activity tests
  use synthetic VLM results and make no paid API requests.

The main database still contains historical migration data only: **1 camera,
1 legacy run, 35 persons, 338 events**, and no zones or activity observations.
Historical CSVs supplied no zone geometry or activity rows; those cannot be
reconstructed. Verification data was kept out of the main database. Heatmap and
trajectory positions still live in session state, as previously deferred.

Database reset (2026-09-26, explicitly requested): cleared all rows from the configured storage/shelby.db in one transaction, including historical import receipts. Verified all seven tables contain zero rows; schema retained. The historical counts above describe the pre-reset verification. Local scratchpad verification databases and backup were not part of this reset.

## Analyst activity narration - 2026-09-27

`Repository.get_activities(run_id=None)` now returns observations in timestamp/ID
order with the linked person's current display name. `LogAnalyzer` merges security
events and activity documents by timestamp (collapsed spans use their start time).
Event sentences retain their existing format. Activities use the known display
name or `Person_<track_id>`, label, confidence and description.

`VLM_ACTIVITY_MIN_LOG_CONFIDENCE = 0.5` is an Analyst-only filter: stored rows and
vision decisions are unchanged. Consecutive identical labels are collapsed within
each `(run_id, track_id)` into a From/To document, with minimum confidence and
distinct descriptions retained in observation order. A changed label or a
below-floor observation breaks the span; other tracks and security events do not.
No collapse crosses runs. Spans describe sampled observations, not proof of
continuous behavior. The Analyst prompt explains this and the uncertainty of VLM
observations. Existing optional run filtering flows through `SecurityAnalyst`,
`LogRetriever`, and `LogAnalyzer`. Existing Analyst sessions must be reopened or
their retriever re-ingested to rebuild an already-created event-only Chroma index.

- **VF-51 - Activities reach Analyst ingestion, with event text preserved.**
  Measured by **7** new synthetic database tests: known/fallback names and exact
  narration, timestamp ordering including out-of-order inserts, configurable
  inclusive confidence floor, low-confidence span boundaries, per-track collapsing
  and label changes, run isolation, and the real Analyst/retriever construction
  path with embeddings/chat/Chroma mocked. Captured Chroma documents contained
  exactly the selected run's activity narration. No paid API calls were made.
  Event-only output matched all **6** stored fixture sentences byte-for-byte,
  covering all five event types and both location prepositions.
  Full unit suite: **44 tests passed** (log: local ignored
  `scratchpad/activity_rag_all_tests.log`).
- **VF-52 - Pipeline regression still passes after Analyst integration.**
  `scripts/verify_storage_pipeline.py --label activity_rag_verified` exited **0**:
  office video **739** analyzed frames, **1050** detections, **29** ephemeral,
  **6** tracks; track 4 ends at **429**, track 6 starts at **450**. Isolated database
  readback verified **1 completed run, 1 zone, 6 persons, 10 events**, with every
  emitted event matching storage and no frame/UI errors. This real-video check
  disabled VLM, so activity ingestion is covered by the synthetic tests above,
  not a live paid VLM/embedding/chat run. `app.py`, vision code and existing vision
  thresholds were unchanged.
