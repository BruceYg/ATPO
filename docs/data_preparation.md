# Data preparation

All training and evaluation files are generated from **canonical records**: one
JSON object per video.

```json
{"id": "clip-0001", "video": "train/clip-0001.mp4", "labels": ["C2", "C3"]}
{"id": "clip-0002", "video": "train/clip-0002.mp4", "labels": []}
```

- **`labels`:** the categories present. An empty list means benign. Labels are
  category IDs of the taxonomy, or aliases it defines (the built-in taxonomies
  also accept 1-based integers, as stored in the paper's RL files).
- **`video`:** resolved against a video root. Absolute paths and URLs are used as
  given.
- **Other fields:** kept and can be copied into the training files (for example
  `subcategories`).

The same records produce the RL file (EasyR1 parquet) and the SFT file
(LLaMA-Factory JSONL), so the two cannot drift apart.

## Validate

```bash
atpo validate-data --taxonomy safewatch \
  --split train=data/train.jsonl --split val=data/val.jsonl \
  --video-root videos --check-media exists      # or: decode (slower), none
```

The report covers:

- unknown labels;
- repeated IDs and repeated videos (error, warn, or allow);
- missing or undecodable media;
- overlap between splits by ID and by video;
- label counts and label cardinality.

Nothing is deduplicated or repaired.

## Build training files

A data recipe lists the splits and the files to build. Every output gets a
`.manifest.json` with row counts, label counts, the prompt hash, and the SHA-256 of
inputs and outputs.

```yaml
# configs/data/my_dataset.yaml
variables:
  DATA: {description: "directory with the canonical files"}
  OUT: {description: "output directory"}
taxonomy: my_taxonomy.yaml          # or safewatch / xdviolence
splits: {train: "${DATA}/train.jsonl", val: "${DATA}/val.jsonl"}
video_root: ${DATA}/videos
validate: {repeated_ids: error, repeated_videos: warn, check_media: exists, split_overlap: error}
outputs:
  - kind: easyr1                    # columns id, video_path, prompt, response
    split: train
    path: ${OUT}/easyr1/train.parquet
    prompt: {render: true}          # task prompt generated from the taxonomy
  - kind: sharegpt                  # LLaMA-Factory: messages, videos, labels (+ dataset_info.json)
    split: train
    path: ${OUT}/sft/train.jsonl
    prompt: {render: true}
    target_style: json              # 'RESULT: {"K1(Knife misuse)": true, ...}'; or simple: "RESULT: K1, K2"
    dataset_name: my_sft
```

```bash
atpo prepare configs/data/my_dataset.yaml --var DATA=... --var OUT=... [--dry-run] [--summary s.json]
```

| Recipe output setting | Values |
| --- | --- |
| `prompt` | `{render: true}`, `{file: safewatch/v3.txt, strip: true, sha256: ...}` (bundled), `{path: my_prompt.txt}`, or `{text: ...}` |
| `label_format` | `id` (`["C2"]`) or `index` (`[2]`, as in the paper's RL files) |
| `path_prefix` | prepended to each video path in the output |
| `oversample` | for `easyr1` with `label_format: index`: append one extra copy of every row containing that category (the paper's C2-oversampled SafeWatch file) |
| `extra_fields` | for `sharegpt`: record fields to copy |

EasyR1 joins `data.image_dir` (the training config's `VIDEO_ROOT`) with the stored
`video_path`. Either store paths relative to the video root (no `path_prefix`), or
point `VIDEO_ROOT` at the directory that the stored paths are relative to.
`atpo train` checks that the first training videos resolve.

The RL prompt column holds the task prompt followed by `" <video>"`. The
answer-format instruction is appended at training time from the reward's
`response_format` (`atpo/resources/prompts/formats/`), as in the paper's runs.

## The paper's datasets

The adapters below read the datasets' own annotation files with the logic of the
paper's conversion scripts. Obtain the datasets from their publishers under their
licences; this repository does not redistribute videos or annotations.

| Command | Reads |
| --- | --- |
| `atpo import-data --safewatch-200k DIR` | SafeWatch-Bench-200K `main_annotation/<folder>/{full.json, full_gt.json}` |
| `atpo import-data --safewatch-bench DIR --benign-labels folder\|empty [--bench-source real\|genai]` | SafeWatch-Bench `{real,genai}/C*/<subcategory>_benchmark.json` |
| `atpo import-data --xdviolence-list FILE --video-path id\|basename` | XD-Violence `train_list.txt` / `test_list.txt`, one `<name>_label_<codes>` per line |
| `atpo filter-media` | duration and frame-count filter on decoded videos |
| `atpo select` | random subset (`--n --seed --method`) or ID manifest (`--ids-file`) |

`scripts/prepare_paper_data.sh` runs the whole chain:

```bash
SAFEWATCH_200K_ANNOTATIONS=<SafeWatch-Bench-200K>/main_annotation SAFEWATCH_BENCH=<SafeWatch-Bench> \
XD_TRAIN_LIST=<XD-Violence>/train_list.txt XD_TEST_LIST=<XD-Violence>/test_list.txt OUT=paper_data \
  scripts/prepare_paper_data.sh                       # SELECTION=filter re-runs the media filters
```

| Step | SafeWatch | XD-Violence |
| --- | --- | --- |
| Records | full videos of the annotation folders (sorted), in file order; labels from `full_gt.json` (a later duplicate wins) | train list lines; codes: `G` → B3 (Explosion), B1–B6 kept, `A` and `0` dropped |
| Training selection | ≤ 120 s (inclusive), ≥ 2 decoded frames, `.unknown_video` files dropped | ≤ 300 s |
| Validation | benchmark videos (genai then real) → `random.sample`, seed 123 → 200 | test list lines → sorted `Random(42).sample` indices → 200 (list order) |
| Recipe | `configs/data/safewatch_paper.yaml` (RL prompt v3, C2-oversampled copy, validation, SFT prompt v5) | `configs/data/xdviolence_paper.yaml` (RL prompt v1, validation, SFT prompt v1) |

The outputs are written under `$OUT/files/`:

| File | Used by |
| --- | --- |
| `safewatch/train.parquet` | SafeWatch GRPO baselines and ATPO-G configs |
| `safewatch/train_oversample_c2.parquet` | SafeWatch ATPO-C configs and the ATPO-G `_o2` config |
| `safewatch/val.parquet`, `xdviolence/val.parquet` | validation during RL |
| `xdviolence/train.parquet` | XD-Violence configs |
| `sft/safewatch_train.jsonl`, `sft/xdviolence_train.jsonl`, `sft/dataset_info.json` | SFT (datasets `safewatch_sft`, `xdviolence_sft`) |

Each RL config records the fingerprint of the file it was trained on
(`paper_data`); `atpo train` prints a note when the file you pass differs.

Video paths are stored relative to each dataset's video directory
(`train/...`, `test/...` for SafeWatch; `train/...`, `test_videos/...` for
XD-Violence). Pass that directory as `VIDEO_ROOT` when training, or rebuild with
`--var VIDEO_PATH_PREFIX=...` if your copy is laid out differently.

**Manifests.** By default the training selection uses the ID manifests in
`configs/data/manifests/`, which list the training videos of the paper in order.
Filter results depend on the local video copies and decoder, so re-running the
filters on another copy can select slightly different videos. With
`SELECTION=filter`, the filters run on your copies, and you can compare the result
with the manifest.

### Properties of these files

They are kept as they were in the paper's runs:

1. **SafeWatch validation labels.** The benign benchmark videos have no category,
   and the conversion labels them with their folder's category, so no validation
   video is labelled benign. Use `--benign-labels empty` for new work; the paper
   recipe keeps `folder`.
2. **Repeated SafeWatch training videos.** A few videos are listed twice upstream,
   with the same labels. Both copies are kept.
3. **Upstream label disagreement.** A few duplicated upstream entries disagree on
   labels; the later entry wins.
4. **Benign SafeWatch SFT target.** It is `"GUARDRAIL: "`, with a trailing space.
   The XD-Violence SFT targets use `GUARDRAIL:` although the XD prompt asks for
   `DETECTION:`.
5. **SafeWatch prompt `v3`.** It contains literal doubled braces (`GUARDRAIL = {{ … }}`).

## Custom datasets

See [custom_training.md](custom_training.md). In short: write a taxonomy file,
produce canonical records, validate them, and build the files with a recipe that
uses `prompt: {render: true}`.
