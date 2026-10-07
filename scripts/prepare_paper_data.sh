#!/usr/bin/env bash
# Build the SafeWatch and XD-Violence training and validation files used in the paper from the
# datasets' own annotation files.
#
# Inputs (environment variables; only the datasets you build are needed):
#   SAFEWATCH_200K_ANNOTATIONS  SafeWatch-Bench-200K main_annotation/ directory
#   SAFEWATCH_BENCH             SafeWatch-Bench directory (real/, genai/)
#   SAFEWATCH_VIDEOS            SafeWatch video directory with train/full/... (SELECTION=filter only)
#   XD_TRAIN_LIST, XD_TEST_LIST XD-Violence train_list.txt / test_list.txt
#   XD_VIDEOS                   XD-Violence video directory with train/... (SELECTION=filter only)
#   OUT                         output directory (default ./paper_data)
#   SELECTION                   "manifest" (default): select the training videos listed in
#                               configs/data/manifests; "filter": re-run the duration and frame
#                               filters on your video copies (results can differ with other
#                               copies or decoders)
#   DATASETS                    "safewatch xdviolence" (default) or one of them
#
# Outputs: $OUT/canonical/*.jsonl (records); $OUT/files/{safewatch,xdviolence}/*.parquet (EasyR1)
# and $OUT/files/sft/ (LLaMA-Factory JSONL + dataset_info.json), each with a .manifest.json.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFESTS="$REPO/configs/data/manifests"
OUT="${OUT:-./paper_data}"
SELECTION="${SELECTION:-manifest}"
DATASETS="${DATASETS:-safewatch xdviolence}"
CANON="$OUT/canonical"
mkdir -p "$CANON" "$OUT/files"

if [[ " $DATASETS " == *" safewatch "* ]]; then
  : "${SAFEWATCH_200K_ANNOTATIONS:?set SAFEWATCH_200K_ANNOTATIONS}" "${SAFEWATCH_BENCH:?set SAFEWATCH_BENCH}"
  atpo import-data --safewatch-200k "$SAFEWATCH_200K_ANNOTATIONS" --taxonomy safewatch \
    --video-prefix train/ --output "$CANON/safewatch_train_all.jsonl"
  if [[ "$SELECTION" == filter ]]; then
    : "${SAFEWATCH_VIDEOS:?set SAFEWATCH_VIDEOS for SELECTION=filter}"
    atpo filter-media --input "$CANON/safewatch_train_all.jsonl" --taxonomy safewatch \
      --video-root "$SAFEWATCH_VIDEOS" --max-duration 120 --min-frames 2 --on-unreadable drop \
      --output "$CANON/safewatch_train.jsonl"
  else
    atpo select --input "$CANON/safewatch_train_all.jsonl" --taxonomy safewatch \
      --ids-file "$MANIFESTS/safewatch_train.txt" --output "$CANON/safewatch_train.jsonl"
  fi
  # Validation labels as in the paper's runs: benign videos carry their folder's category.
  atpo import-data --safewatch-bench "$SAFEWATCH_BENCH" --taxonomy safewatch --benign-labels folder \
    --video-prefix test/ --output "$CANON/safewatch_test_all.jsonl"
  atpo select --input "$CANON/safewatch_test_all.jsonl" --taxonomy safewatch \
    --n 200 --seed 123 --method random_sample --output "$CANON/safewatch_val.jsonl"
  atpo prepare "$REPO/configs/data/safewatch_paper.yaml" --var CANONICAL="$CANON" --var OUT="$OUT/files"
fi

if [[ " $DATASETS " == *" xdviolence "* ]]; then
  : "${XD_TRAIN_LIST:?set XD_TRAIN_LIST}" "${XD_TEST_LIST:?set XD_TEST_LIST}"
  atpo import-data --xdviolence-list "$XD_TRAIN_LIST" --taxonomy xdviolence --video-path id \
    --video-prefix train/ --output "$CANON/xdviolence_train_all.jsonl"
  if [[ "$SELECTION" == filter ]]; then
    : "${XD_VIDEOS:?set XD_VIDEOS for SELECTION=filter}"
    atpo filter-media --input "$CANON/xdviolence_train_all.jsonl" --taxonomy xdviolence \
      --video-root "$XD_VIDEOS" --max-duration 300 --on-unreadable drop \
      --output "$CANON/xdviolence_train.jsonl"
  else
    atpo select --input "$CANON/xdviolence_train_all.jsonl" --taxonomy xdviolence \
      --ids-file "$MANIFESTS/xdviolence_train.txt" --output "$CANON/xdviolence_train.jsonl"
  fi
  atpo import-data --xdviolence-list "$XD_TEST_LIST" --taxonomy xdviolence --video-path basename \
    --video-prefix test_videos/ --output "$CANON/xdviolence_test_all.jsonl"
  atpo select --input "$CANON/xdviolence_test_all.jsonl" --taxonomy xdviolence \
    --n 200 --seed 42 --method sorted_index_sample --output "$CANON/xdviolence_val.jsonl"
  atpo prepare "$REPO/configs/data/xdviolence_paper.yaml" --var CANONICAL="$CANON" --var OUT="$OUT/files"
fi
echo "done: $OUT"
