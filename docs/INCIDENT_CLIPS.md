# On-demand footage in Shelby Analyst

Normal answers show text only, with citations retained internally. No incident
rows, clip buttons or players appear below a normal answer. Ask **show me the
footage** to request the preceding answer's relevant security event. Only one
player is shown, for the latest request, with autoplay disabled.

Footage targets authorized access (`Access`), unauthorized access (`Intrusion`),
authorized removal (`Removal`) and unauthorized removal (`Theft`). Identity and
VLM activity citations are excluded from footage selection. `that` / `it` and
`show me the footage` resolve when exactly one eligible incident was cited;
`the last one` selects the latest eligible incident within one run. Several matching events prompt
with numbered choices without generating clips. The pending choices are saved,
so `the earlier one`, `second`, or `2` can select an incident on the next turn.
Relative time selections across different runs require clarification.
Explanation questions and cancellation clear a pending request. Explicit requests such as
`show footage of unauth access at 04:17:55` retrieve a matching security event.
The abbreviations auth/unauth and the spelling acess are supported, as are
12-hour event times and `Can I see what happened?`. Clear requests use local
rules and meaningful field matching. Unclear requests use one extra call to the
configured chat model, constrained to numbered retrieved event candidates.
Invalid IDs, multiple selections and API/parse failures produce clarification,
never a guessed clip. That optional call has the provider's normal cost; tests
mock it. Model intent interpretation still requires real-world evaluation.

The Analyst returns `(answer_text, cited_sources)`. Its single LLM answer identifies
the numbered retrieved records it used; the application rejects invalid IDs,
deduplicates numbers and limits citations to five. It does not treat all retrieval
hits as citations. A malformed/non-JSON model answer gets no inferred citations.
Citation relevance still depends on the model's attribution; this is not an
independent semantic proof that every claim is supported.

Surveillance answers with citations are rendered by Python into Summary, Events
(ordered by recorded time) and Identity note. Event references must belong to the
validated citation list. Greetings remain short. The model provides the prose;
the application controls section layout and rejects invalid event references.

## Provenance and timing

Every narration document carries `source_type`, `source_id`, `run_id`, and
`frame_no`, plus display fields. Activity spans point to the **first observation**,
using that observation's stored frame. Narration strings retain their previous
wording and ordering. Historical rows without frames use `frame_no=-1` in Chroma
and cannot produce clips; the application does not invent provenance.

Runs already linked through `camera_id` to `cameras.source` and stored native FPS.
New runs additionally snapshot the absolute source path, file size and modification
time in run metadata. A missing/replaced source produces `Clip unavailable` with
a reason, even if an old cache file remains. Old runs fall back to camera source;
they cannot retrospectively detect replacement of a same-named file.

The pipeline increments its source frame counter before retrieval: frame 1 is
video index 0. Clips use `(frame_no - 1) / native_fps`, never processing FPS or
`FRAME_SKIP`. Existing alert narration uses `frame_no / fps`, a one-frame clock
difference retained intentionally to preserve narration and alert behavior.
Frame timestamps assume constant-frame-rate recordings; variable-frame-rate
recordings need stored per-frame presentation timestamps for exact timing.

## Encoding and cache

`storage.clips.get_clip(source_type, source_id)` returns a path or `None`;
`unavailable_reason()` reports a failure within the current request context.
Only the Analyst's user-request handlers call it. This interface can later resolve
archived live-camera/ring-buffer media without changing the UI.

`CLIP_PRE_SECONDS=3` and `CLIP_POST_SECONDS=5` are clamped to recording bounds.
FFmpeg re-encodes to H.264/yuv420p MP4 with faststart (video only), using the PATH
executable or the `imageio-ffmpeg` wheel fallback. Re-encoding uses FFmpeg's
[default accurate seeking](https://www.ffmpeg.org/ffmpeg.html), not keyframe-only
stream copying. Cache files under ignored `storage/clips/` are keyed by incident,
run, frame, source fingerprint, timing configuration and encoding version.
Successful files are atomically published; repeat requests reuse them. No clips
are created during surveillance. Cache retention is manual; no eviction policy
or live-camera archive is implemented in this change.

## Re-ingest existing indexes

Close/restart existing Streamlit sessions after installing the updated requirements,
then click **Open Shelby Analyst**. `SecurityAnalyst` initialization calls
`LogRetriever.ingest_logs()` and rebuilds its index with provenance from SQLite.
For a programmatically held instance, call `analyst.retriever.ingest_logs()`.
Repeat for each run-scoped index in use. Old in-memory answers lack provenance;
ask the question again after restarting. SQLite records do not require migration.
Re-ingestion uses the configured embeddings provider and may incur its normal cost;
validation below mocks that provider and chat.

## Repeat validation

```powershell
.\venv311\Scripts\python.exe -m unittest discover -s tests -v
.\venv311\Scripts\python.exe scripts/verify_storage_pipeline.py --label incident_check
.\venv311\Scripts\python.exe scripts/verify_incident_clips.py --database scratchpad/incident_check.db --ffprobe C:\path\to\ffprobe.exe
```

The latter runs the real Analyst retrieval/answer interface with mocked embedding,
Chroma and chat clients against real-run narration. It submits an explicit footage request through the production
Streamlit chat using AppTest, verifies no clip/video before that request, checks
H.264 and duration with ffprobe, and compares decoded start/event frames with
the source. Its temporary clips are automatically removed. Remove the isolated
`incident_check.db` and `.json` after validation. Browser playback must additionally
be checked in an available browser; AppTest verifies the widget and autoplay flag,
not the browser decoder.
