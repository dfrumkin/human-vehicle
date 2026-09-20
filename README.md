# human-vehicle

Finds interactions between people and vehicles in video. A YOLO detector and tracker mark every
person and vehicle in a clip, and a vision-language model — Gemini, or Qwen3.5 on your own machine
— reads the annotated render and reports what happened. One command runs a clip, or a folder of
them, end to end.

## Requirements

- [uv](https://docs.astral.sh/uv/) — it installs Python 3.13 itself.
- ffmpeg, providing both `ffmpeg` and `ffprobe`. `brew install ffmpeg` on macOS, or your
  distribution's package elsewhere.
- For Gemini: a `GEMINI_API_KEY`, see below.
- For the local model: Apple Silicon (runs on MLX) or a CUDA machine (runs on torch). `uv sync`
  installs whichever fits your platform. Intel Macs are not supported.

## Setup

```bash
uv sync
```

For Gemini, put your key in a `.env` file at the repository root:

```
GEMINI_API_KEY=...
```

It is git-ignored, and it is found by searching upward from wherever you run the command. Running
from outside the repository means the key has to be in the environment instead.

## External assets

- **Input clips.** Put your own mp4 files anywhere; the examples below use `Videos/`, which is
  git-ignored. Nothing is bundled with the repository.
- **Model weights** (`yolo26x.pt`, `yolo26x-reid.onnx`) download from Ultralytics on first use, so
  the first run needs a network. They are git-ignored too.
- **Gemini**, if you use it, is a paid API and your footage is uploaded to it. Uploads are deleted
  when the run ends, including when it is interrupted.

## Running it

```bash
uv run human-vehicle Videos/ outputs/
uv run human-vehicle Videos/clip.mp4 outputs/ --whole-clip
uv run human-vehicle Videos/ outputs/ --vlm local --vlm-model mlx-community/Qwen3.5-9B-4bit
```

A folder input is searched for `*.mp4`, not recursively. The output folder must not be the input's
own folder, and is created if it does not exist.

Each clip produces two files, named after it:

```
outputs/<clip>.mp4     the annotated render, every person and vehicle marked
outputs/<clip>.json    the interactions the model reported
```

By default the clip is examined in overlapping 8-second windows and the sightings are merged into
one list of events. `--whole-clip` asks the model once instead.

The names carry no configuration, so a second run over a clip replaces the first. To compare two
configurations, write to two output folders.

## Options

| Flag | Default | What it is |
| --- | --- | --- |
| `input` | *required* | a video file, or a folder searched for `*.mp4` |
| `output` | *required* | where to write |
| `--weights` | `yolo26x.pt` | the detector |
| `--tracker` | `tracktrack.yaml` | the tracker, or a path to your own YAML |
| `--reid` | `yolo26x-reid.onnx` | `none`, `auto`, or a ReID model |
| `--imgsz` | `auto` | 1280 on frames 1080 px tall or more, else 640 |
| `--buffer-seconds` | `3.0` | how long a lost track stays re-findable |
| `--vlm` | `gemini` | `gemini`, or `local` for Qwen3.5 on your own machine |
| `--vlm-model` | `gemini-3.8-flash` | required under `--vlm local`, where the right id differs per machine |
| `--window-s` / `--stride-s` | `8.0` / `4.0` | the windowing |
| `--whole-clip` | off | one call over the whole clip instead |
| `--seed` | `1` | whichever backend runs |
| `--fps`, `--resolution`, `--thinking-level`, `--max-output-tokens` | `2.0`, `low`, `medium`, `8192` | Gemini only |
| `--max-new-tokens` | `8192` | local only |

`--vlm-model` has no default under `--vlm local`: MLX needs a converted build such as
`mlx-community/Qwen3.5-27B-4bit`, torch the original `Qwen/Qwen3.5-27B`. A setting belonging to the
backend that is not running is an error rather than a silent no-op.

## Reading the results

- **A clip that fails costs only itself.** It is reported, the batch goes on, and the exit code is 1
  if any clip went unanswered. A failed clip leaves its annotated mp4 and no JSON.
- **Check `failed_windows` and `malformed_count`** in a clip's JSON before trusting it. Either one
  nonzero means the clip was never fully examined, however complete the list looks.
- **Labels the clip does not support are dropped**, and a line naming them goes to stderr. That line
  is the only record of what was claimed, so a run's warnings are worth keeping.
- Interactions carry the track labels that were on the two objects — `P3` is person 3, `V2` vehicle
  2. They are best effort: the tracker may miss an object, or renumber one several times, so the
  lists can be empty or hold more than one label.

## Development

```bash
uv run pre-commit install
uv run pytest                  # tests
uv run ruff check .            # lint (add --fix to apply fixes)
uv run ruff format .           # format
uv run pyright --warnings      # type check; warnings fail too
```

| Path | Contents |
| --- | --- |
| `src/human_vehicle/` | the package: tracking, overlay, VLM backends, merging, CLI |
| `tests/` | tests |
| `Videos/` | local input videos, not tracked in git |
