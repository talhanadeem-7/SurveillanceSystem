# Phase 1 README Updates — Complete Summary

**Date:** August 31, 2026  
**Scope:** Updated README.md to reflect all Phase 1 hardening changes from CLAUDE.md  
**Verification:** All changes cross-referenced to measured facts (VF-1 through VF-41)

---

## What Was Updated

### 1. **Component Reference Sections**

#### PoseDetector (vision/detector.py)
- ✅ Documented empty-frame guard (VF-23, VF-24)
- ✅ Documented ephemeral ID collision fix via monotonic counter (VF-9, VF-21)
  - Measured: collisions **244→0, 155→0, 33→0** across three clips
- ✅ Clarified negative IDs now barred from accumulating state

#### StrongSortTracker (vision/tracker.py)
- ✅ Documented empty-frame clock advancement
- ✅ Cross-referenced VF-23, VF-24

#### BotSort Parameters Table (NEW DETAILS)
- ✅ **Exposed all four tracker knobs in config.py:**
  - `TRACKER_MAX_AGE` (default 1200) — **INERT**, only affects `max_obs`
  - `TRACKER_TRACK_BUFFER` (default 30) — **THE ACTUAL removal lever**
  - `TRACKER_NEW_TRACK_THRESH` (default 0.45) — **Closes dead band from above**
  - `TRACKER_MATCH_IOU_THRESHOLD` (default 0.2) — For visibility, not working lever
- ✅ Added measurement evidence (VF-1, VF-2, VF-8, VF-19)
- ✅ **CRITICAL:** Must keep `TRACKER_NEW_TRACK_THRESH` aligned with `CONFIDENCE_THRESHOLD`

#### IdentityGuardian (vision/identity_guardian.py)
- ✅ **DISABLED BY DEFAULT:** `IDENTITY_GUARDIAN_ENABLED = False`
- ✅ Documented over-merging issue (VF-6, VF-12)
  - 5 distinct people → 1 identity (yellow hoodie + dark jackets)
- ✅ Clarified does NOT affect **tracking** quality (VF-11), only **attribution**
- ✅ Added edge-block embedding bug note (VF-5): **100% silhouette energy**
  - Colour never participates; deferred fix (Task 6a, depends on OQ-6)

#### PoseAnalyzer (vision/pose_analyzer.py)
- ✅ **MiDaS as lazy singleton** (VF-32)
  - First load: 3.38s
  - Second in-process run: 0.015s (223x speedup)
  - Thread-safe, keyed by (model type, device)
- ✅ **Zone alignment GATED and DISABLED** (VF-36, VF-37, VF-41)
  - `ZONE_ALIGN_ENABLED = False` (default)
  - `ZONE_ALIGN_INTERVAL = 15` (when enabled)
  - Cost: 47.6 ms/call (33% of frame budget)
  - Speed-up with disable: 33-74% end-to-end
  - **Known bug:** Matches moving people on crowds, throws zones off-screen
    - Silent failure (VF-37); no recompute warnings
    - Needs homography sanity check before re-enabling (OQ-17)

### 2. **Configuration Reference Section**

#### Detection & Tracking (NEW TABLE)
- ✅ All five Phase 1 tracker parameters now documented
- ✅ Added measurement cross-references
- ✅ Added `IDENTITY_GUARDIAN_ENABLED` documentation

#### Depth / Intrusion / Zone Alignment (RESTRUCTURED)
- ✅ Split zone alignment into separate configuration entries
- ✅ Documented `ZONE_ALIGN_ENABLED` and `ZONE_ALIGN_INTERVAL`
- ✅ Added feature-count measurement summary (VF-35)
- ✅ Marked dead config keys as **Unused**

#### Loop-local Constants (UPDATED)
- ✅ `FRAME_SKIP=2` note on throughput (VF-31)
  - End-to-end: **7.2 fps** (full pipeline)
  - After Phase 1 optimizations: **~10.6 fps**
  - VF-38, VF-39: Rejected raising to 3 (degrades tracker, kills lifetimes)
- ✅ `PERSISTENCE_THRESHOLD=5` note on VF-13 regression update
  - Old baseline 5 was freeze artefact
  - **New baseline: 6 tracks, split at frames 430-448**

### 3. **Verified Defects Section**

**FIXED (5 items):**
- ✅ ~~Negative ID collisions~~ — monotonic counter (VF-9)
- ✅ ~~No per-frame error handling~~ — try/except with finally (VF-30)
- ✅ ~~`identity_map` never populated~~ — now written back to session_state

**STILL BROKEN (2 items):**
- ❌ API key never reaches OpenAI (config issue)
- ❌ "Stop" button cannot stop loop (evaluation timing)

### 4. **Behavioural Caveats Section**

- ✅ Added zone alignment disabled note (Phase 1 default, VF-41)
- ✅ Updated zone alignment failure description (VF-37, OQ-17)
- ✅ Changed "runs every frame" to "disabled by default"
- ✅ Added throughput and CPU performance context

### 5. **"Do Not Assume" Section (ENHANCED)**

**NEW Phase 1 clarifications:**
- ✅ Guardian is disabled by default
- ✅ Zone alignment is disabled by default
- ✅ `max_age` is inert; use `track_buffer` (VF-1, VF-2)
- ✅ Regression baseline is **6 identities**, not 5 (VF-25, VF-13)
- ✅ Track IDs now unique per-frame (VF-9) — not colliding

### 6. **NEW: Phase 1 Hardening Summary Section**

Added before "AI Development Context":
- ✅ All 8 Phase 1 tasks listed with status (✅ DONE)
- ✅ Each task cross-referenced to evidence (VF numbers)
- ✅ Regression baseline at close
- ✅ Open questions for Phase 2 (OQ-9, OQ-17, OQ-6, OQ-12, OQ-14, OQ-15)

---

## Key Metrics at Phase 1 Close

| Metric | Value | Evidence |
|---|---|---|
| **End-to-end throughput** | ~10.6 fps (was 7.2) | VF-31, VF-41 |
| **Ephemeral detections** | 31% → 15% (worst clip) | VF-19, VF-21 |
| **Regression baseline** | 6 tracks (was 5, artefact) | VF-13, VF-25 |
| **MiDaS reload speedup** | 3.34s → 0.015s (223x) | VF-32 |
| **Zone alignment CPU savings** | 33-74% when disabled | VF-36, VF-41 |
| **ID collision fixes** | 244 → 0, 155 → 0, 33 → 0 | VF-9, VF-21 |

---

## Critical Configuration Notes

### ⚠️ MUST STAY ALIGNED
```
CONFIDENCE_THRESHOLD = 0.45
TRACKER_NEW_TRACK_THRESH = 0.45  ← These MUST match
```
If misaligned, confidence dead-band reopens and ephemeral detections jump from 15% back to 31% (VF-19, Mode C rejected).

### ⚠️ REGRESSION RULE (VF-13)
```
office cctv.mp4 → 6 tracks
Split: track 4 ends at frame 429
       track 6 starts at frame 450
```
If this changes, **investigate** — do not assume either direction is an improvement. The old "5" was produced by a bug (empty-frame freeze); we keep 6 because it's the honest count.

### ⚠️ ZONE ALIGNMENT DISABLED (VF-37, OQ-17)
```
ZONE_ALIGN_ENABLED = False  ← Default
```
- When enabled: runs every frame, costs 33% of budget
- Known failure: matches moving people on crowds, throws zones thousands of pixels off-screen
- **Before re-enabling:** must implement homography sanity check (OQ-17)

### ⚠️ GUARDIAN DISABLED (VF-6, VF-12)
```
IDENTITY_GUARDIAN_ENABLED = False  ← Default
```
- Over-merges visibly different people (5 people → 1 identity)
- Does NOT affect tracking quality (VF-11) — only attribution
- Revisit only if long-gap ReID gallery is implemented (Phase 6, OQ-6)

---

## What Was NOT Changed

✓ Project structure (no files added/deleted)
✓ Configuration schema (only documented, not restructured)
✓ Code behavior (documentation only — changes verified already in code)
✓ Backward compatibility (all new configs have defaults)
✓ The two live defects (API key, Stop button) remain known issues

---

## For Next Maintainers

1. **If you modify `FRAME_SKIP`:** Re-measure VF-13 regression and re-calibrate all frame-count thresholds (VF-38, VF-39 show why raising to 3 was rejected)

2. **If you re-enable zone alignment:** Implement OQ-17's homography sanity check first (reject if zone corners move >X% of frame)

3. **If Guardian defects surface:** Fix embedding first (VF-5: 100% silhouette), then measure re-merging vs real recovery with ground truth pairs

4. **If long-gap ReID needed (OQ-6):** Measure on at least 2 sparse-footage clips before closing Phase 6; office cctv alone shows K=0 but unmeasured on other scenarios

5. **Before every code change:** Run `scratchpad/regression.py` — it checks VF-13 and exits non-zero on drift

---

## Files Modified

- **README.md** — Complete Phase 1 integration (17 major sections updated)
- **PHASE1_README_UPDATES.md** — This summary

---

## Session Memory

All Phase 1 updates documented in `/memories/repo/phase1_readme_updates.md` for future reference.
