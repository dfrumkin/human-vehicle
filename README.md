# human-vehicle

Interview homework: analyzing humans and vehicles in video, using OpenCV, Pillow, matplotlib,
and a vision-language model — Gemini, or Qwen3.5 run locally. Work happens in Jupyter notebooks;
reusable code lives in `src/human_vehicle/`.

## Prerequisites

- [uv](https://docs.astral.sh/uv/). Python 3.13 itself is installed by uv.
- ffmpeg, supplying both `ffmpeg` and `ffprobe`; both are used. `brew install ffmpeg` on macOS,
  or your distribution's package elsewhere.
- For a local model, one of:
  - **macOS on Apple Silicon**, where the local backend runs on MLX; or
  - **a CUDA machine**, where it runs on transformers and torch.

  `uv sync` installs whichever of those your platform can use — the split is by environment marker
  in `pyproject.toml`, so there is nothing to choose. Intel Macs are not supported: torch has
  published no x86_64 macOS wheel since 2.2, which `ultralytics` already requires.
- For Gemini: a `.env` file at the repository root holding `GEMINI_API_KEY=...`. It is git-ignored
  and you supply it yourself.

## Setup

```bash
uv sync
uv run pre-commit install
```

## Daily commands

```bash
uv run jupyter lab             # notebooks
uv run pytest                  # tests
uv run ruff check .            # lint (add --fix to apply fixes)
uv run ruff format .           # format
uv run pyright --warnings      # type check; warnings fail too
```

## Detection and tracking

`track_video` runs a YOLO detector with the TrackTrack tracker over a clip and returns per-frame
boxes and track ids. `relabel_tracks` renumbers those ids into the two classes the labels use.
`render_tracked_video` draws the boxes onto the original frames and writes a new mp4 — same
resolution, same frame rate, CRF 15 — so a vision-language model can read the scene and the tracking
at once.

```python
from human_vehicle.labels import relabel_tracks
from human_vehicle.overlay import render_tracked_video
from human_vehicle.tracking import track_video

tracks = track_video("Videos/iMGR_0AG3a8_2_3.mp4")
relabelled, translation = relabel_tracks(tracks)
render_tracked_video(relabelled, "annotated.mp4")
```

People and vehicles (car, motorcycle, bus, truck) are kept; everything else is ignored.

Each box is labelled with its category letter and its number — `P29` is person 29 and `V4` is
vehicle 4. There are only these two classes: a car, a motorcycle, a bus and a truck are all `V`,
though the record still says which one each box was. Color repeats the split: azure for people,
orange for vehicles.

A box is marked at its four corners rather than fully outlined. The corners fix its extent just as
well, and leave the object inside it unobscured.

The overlay is terse on purpose. The annotated clip exists to be read by a vision-language model,
and every pixel it covers is a pixel the model cannot see, so it must not bury the people the
footage is about. Labels are glyphs over a dark halo, placed so they do not land on each other,
and the whole overlay costs about 2% of the frame. **A prompt pointed at an annotated clip should
be given the same key**,
since `P29` means nothing to a model that has not been told that `P` is a person and `V` a vehicle.

### Showing confidence, for debugging

`draw_boxes` and `render_tracked_video` take `show_confidence=True`, which appends each box's
detection confidence to its label — `P29 0.531` instead of `P29`.

```python
render_tracked_video(relabelled, "annotated.mp4", show_confidence=True)
```

**It is off by default and should stay off for any clip a model will read.** A label goes from two
characters to eight, which is straight out of the pixel budget the overlay exists to respect, and
wider labels collide more often, so more of them end up detached from their box on busy frames.

### Renumbering

The tracker issues ids across all classes at once, so its people come out as `P3`, `P17`, `P42`.
`relabel_tracks` renumbers them: people become `1..n` and vehicles `1..m`, each sequence contiguous
and independent of the other, in order of first appearance. `P5` is then the fifth person and `V5`
the fifth vehicle, and the two have nothing to do with each other.

**Run it before rendering any clip that will be read downstream.** `render_tracked_video` accepts a
raw tracking record and cannot tell the difference, so skipping the step costs contiguous numbering
rather than correctness — the category letter already keeps a person and a vehicle apart even where
they share a tracker id.

Its second return value is the translation, mapping each original `(category, track_id)` identity to
its new label:

```python
{(Category.PERSON, 17): "P2", (Category.VEHICLE, 4): "V1", ...}
```

It exists for debugging — print it, save it, trace a `V7` in the video back to the tracker's own
output. Nothing downstream needs it, so it can simply be dropped.

### Choosing the models

The detector, the tracker and the ReID model are plain strings:

```python
tracks = track_video(
    "Videos/iMGR_0AG3a8_2_3.mp4",
    weights="yolo26x.pt",
    reid="auto",
)
```

| Parameter | Default | Values |
| --- | --- | --- |
| `weights` | `"yolo26s.pt"` | `yolo26n.pt` … `yolo26x.pt`, smallest and fastest to largest and most accurate |
| `tracker` | `"tracktrack.yaml"` | any tracker the installed Ultralytics ships, or a path to your own YAML |
| `reid` | `"none"` | `"none"`, `"auto"`, or a ReID model such as `"yolo26s-reid.onnx"` |
| `buffer_seconds` | `3.0` | how long a lost track stays re-findable before its id is retired |

Ultralytics 8.4.154 ships `botsort`, `tracktrack`, `bytetrack`, `deepocsort`, `ocsort` and
`fasttrack`; the installed version is what the code validates against, and a later one may add
more.

**Only BoT-SORT, TrackTrack and Deep OC-SORT have a ReID stage.** Asking any of the others for one
raises, rather than being silently ignored. `reid="auto"` re-identifies objects from the detector's
own backbone features, so there is no second network to run; naming a model instead loads that
model as the encoder. ReID is what stops a track id being retired and re-issued each time something
is briefly occluded, so it brings id counts closer to the real number of objects.

**The detection thresholds are the tracker's own, and so is `conf`.** Ultralytics uses 0.1 in track
mode, deliberately below its 0.25 for plain prediction so the tracker has weak detections to
associate with, and each tracker's YAML carries the thresholds it was tuned with. Nothing here
overrides any of them; change one by pointing `tracker` at your own YAML.

### Holding an id through an occlusion

**`buffer_seconds` is in seconds; the tracker's own setting is in frames.** Ultralytics reads
`track_buffer` as a frame count and never scales it by the frame rate, so its default of 30 frames
means one second on a 30 fps clip and five on a 6 fps one — a five-fold difference in the thing the
setting is for. An occlusion is measured in seconds, so `track_video` takes the duration and derives
the frame count from the clip that was probed. The result is floored at 20 frames, because three
seconds at 6 fps is only 18 and a tracker given that little has very few chances to re-associate.

Raising it holds ids through longer occlusions. It is not free: a lost track's position keeps being
predicted from the last one seen, and that prediction drifts further the longer it goes uncorrected,
so a track kept alive long enough can be rebound to the wrong object.

`config_slug(tracks)` renders the configuration as a filename fragment
(`yolo26x__tracktrack__reid-auto__imgsz1280__buf3`), built from the record the run produced, so an
output file cannot be labelled with a configuration that did not make it. The buffer appears as the
seconds asked for rather than the frames they became, so one configuration keeps one name across
clips of different frame rates.

Notes:

- Weights and ReID models download on first use, so the first run needs a network.
- The ReID models are ONNX, and run through the ONNX runtime the project declares. `reid="auto"`
  needs neither the download nor the runtime, since it reuses the detector's own features.
- `device` defaults to `"mps"`. Ultralytics' automatic device selection falls through to the CPU
  on macOS unless MPS is asked for by name, so leave it alone unless you want the CPU.
- `imgsz` defaults to 640. Raise it to 960 or 1280 on 4K footage, where 640 misses small, distant
  people.
- Ultralytics is AGPL-3.0.

## Finding interactions

`find_interactions` shows the annotated clip to a vision-language model and returns the
human-vehicle interactions it reports, validated against the clip.

```python
from human_vehicle.interactions import find_interactions
from human_vehicle.vlm import GeminiBackend

backend = GeminiBackend()
run = find_interactions("notebooks/outputs/annotated/clip.mp4", backend)

for item in run.interactions:
    print(item.person_ids, item.vehicle_ids, item.interaction, item.start_time_s, item.end_time_s)
```

The input is the **annotated** clip — the mp4 `render_tracked_video` writes — and it should be one
rendered without `show_confidence`. The model reads the labels out of the pixels, so it is told the
key the overlay draws with: `P` is a person, `V` a vehicle, numbered independently from 1.

Each interaction is one person with one vehicle, described in words, placed in time with a start, an
end and the moment the model found clearest, and given a confidence. The prompt asks for uncertain
candidates below 0.5 rather than omitting them, so a low-confidence record is a candidate to look
at, not a mistake.

### The track labels an interaction carries

Besides the descriptions, an interaction carries the labels that were on the two objects:

```python
person_ids=["P3", "P7"], vehicle_ids=["V2"]
```

An ideal detector-tracker gives each object exactly one label. A real one gives none, one, or
several, so these are lists, in the order the labels were first seen:

| What the tracker did | What comes back |
| --- | --- |
| labelled the person `P3` throughout | `["P3"]` |
| lost them and re-found them as `P7` | `["P3", "P7"]` |
| never found them at all | `[]` |

**They are best effort, not ground truth.** The prompt says so: the tracker may have missed an
object altogether, may drop a label mid-interaction, and may renumber one real object several times.
Whether something is an interaction is decided from the imagery, so an unlabelled person handling a
vehicle is reported like any other — and then the descriptions are the only handle on them, which is
why they are still asked for.

The labels are what lets the same interaction, seen by two overlapping windows, be recognised as one
event: the spans overlap in time and the labels agree. **`find_interactions` itself de-duplicates
nothing** — the second sighting is corroboration, and a run's record keeps every one. Collapsing them
is a separate, pure step: see [Merging what the windows saw](#merging-what-the-windows-saw).

### What comes back

An `InteractionRun`: the interactions, the records that failed validation, the per-call detail, the
token usage and cost, and the configuration that produced all of it. It is a Pydantic model, so a
record is `run.model_dump_json(indent=2)` to disk and `InteractionRun.model_validate_json(...)` back.

Name a saved record with `run_tag(run)` — the configuration plus when the run started. `run_slug` on
its own names only the configuration, so using it as a filename means a repeat of the same settings
overwrites the earlier record. That matters here: there is no temperature in this API and `seed` is
best effort, so two runs at identical settings do differ, and comparing them is how a real
difference is told from noise.

`run.interactions` is the clip's answer, **ordered by start time**, whichever call found each one —
a windowed run gathers several calls into this one list, and which window reported what is an
artifact of the run rather than something to read the results by. Each record still carries its
`window_index` for tracing one back to the call that produced it. `run.malformed` stays in call
order instead, since a record is often there precisely because its times are not usable.

A record that violates the clip — a time past the end, a reversed span, a label nothing could have
drawn — is kept in `run.malformed` exactly as it arrived, with the reasons beside it. Nothing is
repaired and nothing is silently dropped: a confused model should stay visible.

**A failed call never raises.** It is recorded against its own call, so one bad window does not cost
a run the others; only a run where every call failed is itself marked failed. Bad arguments are the
opposite — a missing clip, or a windowing that would leave gaps — and raise before anything is
called.

### Windowing

```python
run = find_interactions(clip, backend, window_s=8.0, stride_s=4.0)
```

This tiles the clip in 8-second windows advancing 4 seconds, one call each, so every second is
examined in a short context and neighbours overlap by four. Counts are **not** comparable with a
whole-clip run: the overlap means one event is usually reported twice, and that second sighting is
corroboration rather than noise.

A windowed run assumes the backend reports times on the **clip's** clock when it is shown only a
segment. The prompt says so and every record is validated against it, but a model that quietly
reverted to segment-local times would shift every timestamp while looking plausible;
`notebooks/human_vehicle_interactions.ipynb` has a live check that verifies this for a backend, and
windowed runs there refuse to start until it passes.

### Choosing the model

The model sits behind `VlmBackend`, and the prompt, the schema, the validation and the windowing
know nothing about which one is running. There are two implementations: `GeminiBackend`, and
`QwenBackend` for a model on your own machine.

```python
from human_vehicle.vlm import GeminiBackend, QwenBackend

backend = GeminiBackend(fps=4.0, resolution="high", thinking_level="high", seed=2)
backend = QwenBackend("mlx-community/Qwen3.5-27B-4bit")  # Apple Silicon
backend = QwenBackend("Qwen/Qwen3.5-9B")  # CUDA
```

Both are shown **video**, never a sequence of stills, at 2 frames a second.

**The two are not answering quite the same question, so do not read a difference in counts as a
difference in model quality.** Gemini is shown the whole upload and told which span to look at.
`QwenBackend` cuts the window out with ffmpeg and shows the model that segment as a clip in its own
right, then puts the times back on the clip's clock itself — neither runtime accepts a time range,
and correcting the times here beats asking a 9B model to add an offset to everything it reports.

| Parameter | Default | What it is |
| --- | --- | --- |
| `model_id` | `"gemini-3.8-flash"` | the model to call |
| `fps` | `2.0` | how often the video is sampled; also the tolerance a windowed answer is allowed at a window's edge |
| `resolution` | `"low"` | 70 tokens a frame; `"medium"` is identical for video, `"high"` is 280 |
| `thinking_level` | `"medium"` | `low`, `medium` or `high`; thinking bills at the output rate |
| `seed` | `1` | recorded, but there is no temperature here and identical requests can still differ |

#### Running Qwen3.5 locally

`QwenBackend` takes the model id and nothing else that matters:

```python
run = find_interactions(clip, QwenBackend("Qwen/Qwen3.5-27B"), window_s=8.0, stride_s=4.0)
```

| Parameter | Default | What it is |
| --- | --- | --- |
| `model_id` | *required* | must match the runtime your platform uses — see below |
| `max_new_tokens` | `8192` | the answer's budget |
| `seed` | `1` | applied to the runtime's RNG |

There is no default model id, because the right one differs per machine: MLX needs a converted build
such as `mlx-community/Qwen3.5-27B-4bit`, transformers the original `Qwen/Qwen3.5-27B`. Nothing here
reasons about whether the weights will fit — a 27B model in BF16 is about 54 GB, and an id that does
not fit fails at load.

Qwen3.5 thinks by default; this backend switches that off (`enable_thinking=False`), because
reasoning would eat the token budget before the JSON was written. There is no separate instruct
build to pick instead — one hybrid model per size, with reasoning toggled at the chat template.

**Runs repeat, within limits.** Decoding is greedy and the seed is set, so the same clip answers the
same way on the same machine with the same library versions. It is not reproducible across
accelerators or across upgrades, and CUDA GEMM determinism is not claimed: that needs
`CUBLAS_WORKSPACE_CONFIG` set before torch initialises, which is your environment's business, not
this code's. `config` records the device and runtime version so a difference can be attributed.

`tolerance_s` is 1.0 s here against Gemini's 0.5 s at the same frame rate. Qwen's vision encoder
folds two consecutive frames into one temporal position (`temporal_patch_size: 2`), so that is the
finest moment it can place, and claiming the sampling interval instead would reject correct answers
near a window's seam.

**Your footage leaves the machine, and two retentions apply.** Uploaded files are kept 48 hours and
then deleted; every request sets `store=False`, which switches off the 55 days of interaction logs
the API would otherwise keep. A clip is uploaded once and reused for every call of a run — that is
the reason to use the File API at all — and whoever holds a backend owns those uploads until they
call `backend.delete_uploads()`.

`notebooks/human_vehicle_interactions.ipynb` runs all of this and shows what came back: a filmstrip
per interaction headed by its reported labels, a contact sheet of the frames no interaction claims,
and the labels laid out beside the times.

It runs whatever mp4 files are in `notebooks/outputs/annotated/`, which you fill yourself — copy in
the annotated renders you want to look at. The folder is git-ignored and created on first run, and
nothing there is filtered, so put annotated renders in it rather than source clips: an un-annotated
clip produces a run that looks perfectly normal and answers a different question, with every id list
empty.

## Merging what the windows saw

A windowed run reports one event once per window that saw it, and de-duplicates nothing on purpose.
`merge_interactions` collapses those sightings, so a clip's answer reads as a list of events.

```python
from human_vehicle.merge import merge_interactions

merged = merge_interactions(run)
for event in merged.interactions:
    print(event.person_ids, event.vehicle_ids, event.interaction, event.sightings)
```

Two records are the same event when **all three** hold:

- their spans overlap in time, touching endpoints included — `[4.0, 8.0]` and `[8.0, 11.0]` match,
  because that is what one event split at a window seam looks like;
- their `person_ids` intersect;
- their `vehicle_ids` intersect.

Intersection rather than equality is what lets a relabelled object match: `["P1", "P2"]` and `["P2"]`
are one person. Grouping is transitive, so an event crossing three windows becomes one record.

### A record without ids never merges

If either side's id list is empty there is nothing to match on, and such a record passes through
alone. This costs recall — an event whose person the tracker lost stays split across windows — and it
is still the right trade. One clip here has an unlabelled person entering a car's driver seat while a
labelled one exits its passenger side, overlapping in time at the same vehicle. Merging on the
vehicle alone would fuse them into a person who did both at once, and nothing downstream could tell.
A missed merge leaves visible duplicates; a wrong merge invents an event that never happened.

### Which record is believed

One contributor supplies every field that is not a union, so a merged record cannot contradict
itself. It is chosen in two steps:

1. **Confidence vetoes the clearly unsure.** Only records within `confidence_band` (0.3 by default)
   of the group's best can supply fields. A veto rather than a ranking, because the model uses the
   scale in coarse steps — 0.95, 0.92, 0.90 — so a 0.03 gap is noise while containment is not.
2. **Among the rest**: contained first, then the largest evidence margin, then confidence as an
   ordinary tiebreak, then two keys that only make the result deterministic.

**Contained** means the span lies strictly inside its window, so the model watched the event start
*and* finish. A span touching either edge is truncated — the model saying the action was still going
when its view ran out — and that matters most for direction: "entering" and "exiting" look alike
frame by frame. A truncated record in this footage called an entry an exit at 0.95 confidence, while
the windows that saw past the seam called it correctly.

| field | how it merges |
| --- | --- |
| `person_ids`, `vehicle_ids` | union of **every** contributor's, first occurrence kept — including records the veto excluded, since a weak record may be the only one that saw a label |
| `person_description`, `vehicle_description`, `evidence_time_s`, `confidence` | the chosen record's |
| `interaction` | the chosen record's — or, when **nothing** saw the whole event, every distinct account joined with `" | "` |
| `start_time_s`, `end_time_s` | the chosen record's — or the union of spans when nothing saw the whole event |
| `window_indexes`, `sightings` | which calls reported it, and how many records were collapsed |

The text fields differ because they answer different questions. The two descriptions name the same
object twice, so the better-observed one is the answer. `interaction` is not like that: an event
longer than a window is seen in parts, and a later window may be the only one that saw the loading or
the closing. Where something saw the event whole, its account already covers the rest; where nothing
did, every distinct account is kept.

### The merged file

`notebooks/human_vehicle_interactions.ipynb` writes `{run_tag}__merged.json` beside each windowed
run's `{run_tag}.json` — same stem, so the pair is obvious. The run is saved first: the calls were
paid for, and the merged file is derived.

**It is derived, not a source of truth.** Regenerate it from the run beside it at any time, and check
`failed_windows` and `malformed_count` before trusting it: a nonzero either way means the clip was
never fully examined, however complete the merged list looks.

One false-merge mode is accepted rather than guarded. Because grouping is transitive, a single
over-broad span can bridge two genuinely separate episodes for the same person and vehicle into one
record. What gives it away is a span much longer than the interaction it describes, and a `sightings`
count higher than the overlapping windows could account for.

## Layout

| Path | Contents |
| --- | --- |
| `src/human_vehicle/` | Reusable code, type checked in Pyright strict mode |
| `notebooks/` | Jupyter notebooks |
| `tests/` | Tests |
| `Videos/` | Local input videos, not tracked in git |

Pyright's CLI only reads `.py` files, so notebooks are checked by Ruff and by your editor.
Anything worth type checking belongs in `src/human_vehicle/`.
