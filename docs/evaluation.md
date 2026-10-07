# Evaluation

```bash
atpo evaluate --predictions preds.jsonl --ground-truth data/test.jsonl \
  --taxonomy safewatch --invalid-policy negative --output metrics.json
```

## Coverage is explicit

Every evaluation states how incomplete predictions were scored:

| Option | Applies to | Choices |
| --- | --- | --- |
| `--invalid-policy` (required) | records with status `parse_error`, `video_error`, `generation_error` | `negative`: predicted no category (the paper's convention); `exclude`: left out; `error`: stop |
| `--missing-policy` (default `error`) | ground-truth videos without a prediction record | as above |
| `--label-source` | predictions | `labels` (release parser) or `historical` (`parse.historical_labels`: missing keys false, as the paper's parser) |

`partial` records are scored with the categories they flagged and counted
separately. The report states, for the samples scored:

- the number of invalid, partial, and missing samples;
- which policy each category of sample was scored under.

## Metrics

The definitions reproduce the evaluation script that produced the paper's numbers,
including scikit-learn's `zero_division=0` conventions (`tests/test_evaluation.py`
compares with scikit-learn).

| Metric | Definition |
| --- | --- |
| `jaccard_similarity` | mean over samples of \|P ∩ G\| / \|P ∪ G\|; both empty scores 1, exactly one empty scores 0 |
| `exact_match_ratio` | share of samples with P = G |
| `hamming_accuracy` | share of correct (sample, category) decisions |
| `micro_precision/recall/f1` | pooled over samples and categories; the paper's "P/R" |
| `macro_*` | unweighted mean over all categories (a never-predicted category contributes 0) |
| `sample_*`, `weighted_*` | scikit-learn `average="samples"` / `"weighted"` |
| `per_category` | TP, FP, FN, TN, precision, recall, F1, specificity, support |
| `binary` | unsafe = any category: accuracy, precision, recall, F1, `false_refusal_rate` (benign videos flagged), `violation_leakage_rate` (unsafe videos missed) |

## The paper's evaluation sets

SafeWatch-Real (the real videos of SafeWatch-Bench, benign videos without a
category):

```bash
atpo import-data --safewatch-bench <SafeWatch-Bench> --bench-source real --benign-labels empty \
  --taxonomy safewatch --video-prefix test/ --output safewatch_real.jsonl
```

XD-Violence (the test list):

```bash
atpo import-data --xdviolence-list <XD-Violence>/test_list.txt --taxonomy xdviolence \
  --video-path basename --video-prefix test_videos/ --output xdviolence_test.jsonl
```

To score as in the paper, use `--invalid-policy negative --label-source historical`.
