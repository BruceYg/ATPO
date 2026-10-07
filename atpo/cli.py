"""Command-line interface: ``atpo <command> --help`` for details."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence


def _parse_overrides(items: Optional[Sequence[str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


# ---------------------------------------------------------------------- presets
def _add_presets(sub) -> None:
    p = sub.add_parser("presets", help="list bundled inference presets")
    p.set_defaults(func=_cmd_presets)


def _cmd_presets(args) -> int:
    from .inference.presets import list_presets

    for name, description in list_presets().items():
        print(f"{name}\n    {description}")
    return 0


# ---------------------------------------------------------------------- models
def _add_models(sub) -> None:
    p = sub.add_parser("models", help="list named checkpoints and where they resolve to")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_models)


def _cmd_models(args) -> int:
    from .inference.model_index import env_var_name, list_models

    entries = list_models()
    if args.json:
        print(json.dumps([e.to_dict() for e in entries], indent=2))
        return 0
    for e in entries:
        where = f"{e.location}" + (f"@{e.revision}" if e.revision else "") if e.location else \
            f"not configured (set {env_var_name(e.name)})"
        print(f"{e.name}\n    {e.description}\n    status: {e.status}; location: {where}; "
              f"fallback preset: {e.inference_preset}")
    return 0


# ----------------------------------------------------------------- show-config
def _add_show_config(sub) -> None:
    p = sub.add_parser("show-config", help="print the resolved inference configuration")
    p.add_argument("--model", help="model name (atpo models), local directory, or repository ID")
    p.add_argument("--preset", help="bundled preset name")
    p.add_argument("--config", help="inference config JSON file")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override, e.g. video.max_pixels=50176")
    p.add_argument("--training-run", help="atpo train output dir: derive the config from the training prompt")
    p.add_argument("--prompt", action="store_true", help="print only the exact prompt text")
    p.add_argument("--output", help="write the resolved config (atpo_config.json format) to this file")
    p.set_defaults(func=_cmd_show_config)


def _cmd_show_config(args) -> int:
    from .inference.config import InferenceConfig
    from .inference.model import resolve_inference_config
    from .inference.presets import load_preset

    overrides = _parse_overrides(args.set)
    if args.model:
        from .inference.model import PROCESSOR_PATTERNS, resolve_model

        model_path, _, fallback = resolve_model(args.model, allow_patterns=PROCESSOR_PATTERNS)
        config, info = resolve_inference_config(model_path, preset=args.preset, config=args.config,
                                                overrides=overrides, fallback_preset=fallback)
    elif args.preset:
        config, info = load_preset(args.preset).with_overrides(overrides), {"source": f"preset:{args.preset}"}
    elif args.config:
        config, info = InferenceConfig.from_file(args.config).with_overrides(overrides), {"source": "explicit_config"}
    elif args.training_run:
        import yaml

        from .export.exporter import inference_config_from_training

        run = Path(args.training_run)
        easyr1_config = yaml.safe_load((run / "easyr1_config.yaml" if run.is_dir() else run).read_text())
        config = inference_config_from_training(
            easyr1_config, base_model=str(easyr1_config["worker"]["actor"]["model"]["model_path"])
        ).with_overrides(overrides)
        info = {"source": "training_run", "path": str(run)}
    else:
        raise SystemExit("give --model, --preset, --config, or --training-run")
    if args.output:
        config.save(args.output)
    if args.prompt:
        sys.stdout.write(config.prompt)
        return 0
    print(json.dumps({"resolution": info, "fingerprint": config.fingerprint(), "config": config.to_dict()},
                     indent=2, ensure_ascii=False))
    return 0


# --------------------------------------------------------------------- predict
def _add_predict(sub) -> None:
    p = sub.add_parser("predict", help="classify videos with a local or Hugging Face checkpoint")
    p.add_argument("--model", required=True, help="model name (atpo models), local directory, or repository ID")
    p.add_argument("--revision", help="Hub revision (branch, tag, or commit)")
    p.add_argument("--preset", help="inference preset (required if the checkpoint has no atpo_config.json)")
    p.add_argument("--config", help="inference config JSON (overrides the checkpoint's)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="config override, e.g. video.max_pixels=50176 or generation.max_new_tokens=512")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", action="append", help="video path or URL (repeatable)")
    src.add_argument("--input", help="JSONL with 'video' (and optional 'id') per line")
    p.add_argument("--video-root", help="base directory for relative paths in --input (default: its directory)")
    p.add_argument("--output", help="output JSONL (default: print to stdout for --video)")
    p.add_argument("--resume", action="store_true", help="skip ids already present in --output")
    p.add_argument("--overwrite", action="store_true", help="replace an existing --output")
    p.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--processor", help="processor source (Hub ID or path) overriding the config")
    p.add_argument("--processor-revision")
    p.add_argument("--cache-dir")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--max-model-len", type=int, help="vLLM: maximum model length (default from config)")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM")
    p.add_argument("--device-map", default="auto", help="transformers")
    p.add_argument("--attn-implementation", help="transformers, e.g. sdpa or flash_attention_2")
    p.add_argument("--no-raw", action="store_true", help="omit raw_response from the output")
    p.add_argument("--no-explanation", action="store_true", help="omit the reasoning text from the output")
    p.set_defaults(func=_cmd_predict)


def _cmd_predict(args) -> int:
    from .inference.model import VideoSafetyModel
    from .inference.runner import load_inputs, run_predictions

    if args.output is None and args.input is not None:
        raise SystemExit("--output is required with --input")
    backend_kwargs: dict[str, Any] = {}
    if args.backend == "vllm":
        backend_kwargs.update(tensor_parallel_size=args.tensor_parallel_size,
                              gpu_memory_utilization=args.gpu_memory_utilization)
        if args.max_model_len is not None:
            backend_kwargs["max_model_len"] = args.max_model_len
    else:
        backend_kwargs["device_map"] = args.device_map
        if args.attn_implementation:
            backend_kwargs["attn_implementation"] = args.attn_implementation
    ids, videos, input_info = load_inputs(videos=args.video, input_path=args.input, video_root=args.video_root)
    model = VideoSafetyModel.from_pretrained(
        args.model, revision=args.revision, preset=args.preset, config=args.config,
        overrides=_parse_overrides(args.set), backend=args.backend, processor=args.processor,
        processor_revision=args.processor_revision, cache_dir=args.cache_dir,
        local_files_only=args.local_files_only, trust_remote_code=args.trust_remote_code, dtype=args.dtype,
        **backend_kwargs,
    )
    if args.output is None:
        for prediction in model.predict_batch(videos, ids, batch_size=args.batch_size):
            print(json.dumps(prediction.to_dict(include_raw=not args.no_raw,
                                                include_explanation=not args.no_explanation), ensure_ascii=False))
        return 0
    meta = run_predictions(
        model, ids, videos, args.output, input_info=input_info, batch_size=args.batch_size, resume=args.resume,
        overwrite=args.overwrite, include_raw=not args.no_raw, include_explanation=not args.no_explanation,
    )
    print(json.dumps({"output": args.output, "status": meta["status"], "counts": meta["counts"]}, indent=2))
    return 0 if meta["status"] == "complete" else 1


# -------------------------------------------------------------------- evaluate
def _add_evaluate(sub) -> None:
    p = sub.add_parser("evaluate", help="compute multi-label metrics with explicit coverage handling")
    p.add_argument("--predictions", required=True, help="JSONL written by `atpo predict`")
    p.add_argument("--ground-truth", required=True, help="canonical data JSONL with id and labels")
    p.add_argument("--taxonomy", required=True, help="builtin name (safewatch, xdviolence) or taxonomy file")
    p.add_argument("--invalid-policy", required=True, choices=["negative", "exclude", "error"],
                   help="how to score invalid predictions; the paper counted parse failures as 'negative'")
    p.add_argument("--missing-policy", default="error", choices=["negative", "exclude", "error"],
                   help="how to score ground-truth samples without a prediction record")
    p.add_argument("--label-source", default="labels", choices=["labels", "historical"],
                   help="release labels, or parse.historical_labels (missing decisions counted as false, "
                        "as the paper's evaluation did)")
    p.add_argument("--output", help="write the full result JSON here")
    p.set_defaults(func=_cmd_evaluate)


def _cmd_evaluate(args) -> int:
    from .evaluation import evaluate_samples, ground_truth_from_records, load_jsonl, samples_from_predictions, summary_table
    from .taxonomy import load_taxonomy

    taxonomy = load_taxonomy(args.taxonomy)
    gt = ground_truth_from_records(load_jsonl(args.ground_truth), taxonomy)
    samples, extra = samples_from_predictions(load_jsonl(args.predictions), gt, taxonomy,
                                              label_source=args.label_source)
    result = evaluate_samples(samples, taxonomy, invalid_policy=args.invalid_policy,
                              missing_policy=args.missing_policy, extra=extra)
    result["inputs"] = {"predictions": args.predictions, "ground_truth": args.ground_truth,
                        "label_source": args.label_source}
    print(summary_table(result, taxonomy))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


# --------------------------------------------------------------------- data
def _add_prepare(sub) -> None:
    p = sub.add_parser("prepare", help="build SFT/RL training files from canonical records with a data recipe")
    p.add_argument("recipe", help="data recipe YAML (see configs/data/)")
    p.add_argument("--var", action="append", metavar="NAME=VALUE", help="value for ${NAME} in the recipe")
    p.add_argument("--dry-run", action="store_true", help="validate and report without writing files")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--summary", help="write the full summary (validation reports, outputs) to this JSON file")
    p.add_argument("--verbose", action="store_true", help="print the full summary")
    p.set_defaults(func=_cmd_prepare)


def _parse_vars(items) -> dict[str, str]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--var expects NAME=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key] = value
    return out


def _cmd_prepare(args) -> int:
    from .data.recipe import run_recipe

    summary = run_recipe(args.recipe, variables=_parse_vars(args.var), dry_run=args.dry_run, overwrite=args.overwrite)
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.verbose:
        print(json.dumps(summary, indent=2))
        return 0
    for name, report in summary["validation"].items():
        print(f"{name}: {report['num_records']} records, {report['num_benign']} benign, labels {report['label_counts']}")
        for warning in report["warnings"]:
            print(f"  warning: {warning[:160]}")
    for output in summary["outputs"]:
        print(f"wrote {output['path']} ({output['rows']} rows{', dry run' if args.dry_run else ''})")
    return 0


def _add_validate_data(sub) -> None:
    p = sub.add_parser("validate-data", help="validate canonical JSONL records")
    p.add_argument("--taxonomy", required=True)
    p.add_argument("--split", action="append", required=True, metavar="NAME=PATH",
                   help="split name and JSONL path (repeatable); overlap between splits is reported")
    p.add_argument("--video-root")
    p.add_argument("--check-media", choices=["none", "exists", "decode"], default="none")
    p.add_argument("--repeated-ids", choices=["allow", "warn", "error"], default="error")
    p.add_argument("--repeated-videos", choices=["allow", "warn", "error"], default="warn")
    p.add_argument("--require-all-categories", action="store_true")
    p.set_defaults(func=_cmd_validate_data)


def _cmd_validate_data(args) -> int:
    from .data.records import check_split_overlap, load_records, validate_records
    from .taxonomy import load_taxonomy

    taxonomy = load_taxonomy(args.taxonomy)
    splits = {name: load_records(path, taxonomy) for name, path in _parse_vars(args.split).items()}
    result: dict[str, Any] = {}
    ok = True
    for name, records in splits.items():
        report = validate_records(records, taxonomy, video_root=args.video_root, repeated_ids=args.repeated_ids,
                                  repeated_videos=args.repeated_videos, check_media=args.check_media,
                                  require_all_categories=args.require_all_categories)
        result[name] = report.to_dict()
        ok = ok and report.ok
    overlap = check_split_overlap(splits, video_root=args.video_root)
    result["split_overlap"] = overlap
    print(json.dumps(result, indent=2))
    return 0 if ok and not overlap else 1


def _add_import_data(sub) -> None:
    p = sub.add_parser("import-data", help="create canonical records from dataset annotations or existing files")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--safewatch-200k", metavar="DIR", help="SafeWatch-Bench-200K main_annotation directory")
    src.add_argument("--safewatch-bench", metavar="DIR", help="SafeWatch-Bench directory with real/ and genai/")
    src.add_argument("--xdviolence-list", metavar="FILE", help="XD-Violence list file (train_list.txt, test_list.txt)")
    src.add_argument("--easyr1", help="EasyR1 parquet with id, video_path, prompt, response")
    src.add_argument("--sharegpt", help="LLaMA-Factory ShareGPT JSONL with id, videos, labels")
    p.add_argument("--taxonomy", required=True)
    p.add_argument("--benign-labels", choices=["folder", "empty"],
                   help="--safewatch-bench (required): historical 'folder' labels benign videos with their folder's "
                        "category; 'empty' labels them benign")
    p.add_argument("--bench-source", choices=["all", "real", "genai"], default="all",
                   help="--safewatch-bench: which videos (the SafeWatch-Real evaluation of the paper uses 'real')")
    p.add_argument("--video-path", choices=["id", "basename"],
                   help="--xdviolence-list (required): store videos under the full ID or the file name")
    p.add_argument("--video-type", choices=["full", "clip", "all"], default="full", help="--safewatch-200k videos")
    p.add_argument("--video-prefix", default="", help="prepend to each video path, e.g. train/")
    p.add_argument("--path-prefix", help="--easyr1/--sharegpt: prefix to strip from stored video paths")
    p.add_argument("--keep-field", action="append", default=[], help="--sharegpt: field to keep (e.g. subcategories)")
    p.add_argument("--output", required=True)
    p.set_defaults(func=_cmd_import_data)


def _write_records_with_manifest(records, output: str, info: dict) -> dict:
    import hashlib
    from collections import Counter

    from .data.records import write_records

    write_records(records, output)
    digest = hashlib.sha256(Path(output).read_bytes()).hexdigest()
    counts = Counter(label for r in records for label in r.labels)
    manifest = {"output": output, "records": len(records), "benign": sum(1 for r in records if not r.labels),
                "label_counts": dict(sorted(counts.items())), "unique_ids": len({r.id for r in records}),
                "output_sha256": digest, **info}
    Path(output + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _cmd_import_data(args) -> int:
    from .data import sources
    from .data.adapters import records_from_easyr1_parquet, records_from_sharegpt
    from .taxonomy import load_taxonomy

    taxonomy = load_taxonomy(args.taxonomy)
    options = {"video_prefix": args.video_prefix}
    if args.safewatch_200k:
        kind, path = "safewatch-200k", args.safewatch_200k
        options["video_type"] = args.video_type
        records = sources.safewatch_annotation_records(path, taxonomy, video_type=args.video_type,
                                                       video_prefix=args.video_prefix)
    elif args.safewatch_bench:
        if args.benign_labels is None:
            raise SystemExit("--safewatch-bench needs --benign-labels folder|empty (see docs/data_preparation.md)")
        kind, path = "safewatch-bench", args.safewatch_bench
        bench_sources = ("real", "genai") if args.bench_source == "all" else (args.bench_source,)
        options["benign_labels"] = args.benign_labels
        options["sources"] = list(bench_sources)
        records = sources.safewatch_benchmark_records(path, taxonomy, benign_labels=args.benign_labels,
                                                      sources=bench_sources, video_prefix=args.video_prefix)
    elif args.xdviolence_list:
        if args.video_path is None:
            raise SystemExit("--xdviolence-list needs --video-path id|basename")
        kind, path = "xdviolence-list", args.xdviolence_list
        options["video_path"] = args.video_path
        records = sources.xdviolence_list_records(path, taxonomy, video_path=args.video_path,
                                                  video_prefix=args.video_prefix)
    elif args.easyr1:
        kind, path, options = "easyr1", args.easyr1, {"path_prefix": args.path_prefix}
        records = records_from_easyr1_parquet(path, taxonomy, path_prefix=args.path_prefix)
    else:
        kind, path, options = "sharegpt", args.sharegpt, {"path_prefix": args.path_prefix, "keep_field": args.keep_field}
        records = records_from_sharegpt(path, taxonomy, path_prefix=args.path_prefix,
                                        extra_fields=tuple(args.keep_field))
    manifest = _write_records_with_manifest(records, args.output, {"source": {"kind": kind, "path": str(path),
                                                                              "options": options},
                                                                   "taxonomy": taxonomy.name})
    print(json.dumps({k: manifest[k] for k in ("output", "records", "unique_ids", "benign", "label_counts")}))
    return 0


# ---------------------------------------------------------------- filter-media
def _add_filter_media(sub) -> None:
    p = sub.add_parser("filter-media", help="drop records by video duration / frame count (historical filters)")
    p.add_argument("--input", required=True)
    p.add_argument("--taxonomy", required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--max-duration", type=float, help="seconds, inclusive (SafeWatch 120, XD-Violence 300)")
    p.add_argument("--min-duration", type=float)
    p.add_argument("--min-frames", type=int, help="minimum decoded frames (SafeWatch 2)")
    p.add_argument("--drop-suffix", action="append", help="drop videos with this suffix (default .unknown_video)")
    p.add_argument("--on-unreadable", choices=["drop", "error"], default="error",
                   help="historical scripts dropped missing/unreadable videos silently ('drop', reported here)")
    p.add_argument("--backend", choices=["decord", "opencv"], default="decord")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--probe-cache", help="JSON cache of probe results for repeated runs")
    p.add_argument("--output", required=True)
    p.add_argument("--report", help="report path (default <output>.media_report.json)")
    p.set_defaults(func=_cmd_filter_media)


def _cmd_filter_media(args) -> int:
    from .data.records import load_records
    from .data.sources import filter_by_media
    from .taxonomy import load_taxonomy

    taxonomy = load_taxonomy(args.taxonomy)
    records = load_records(args.input, taxonomy)
    kept, report = filter_by_media(
        records, args.video_root, max_duration=args.max_duration, min_duration=args.min_duration,
        min_frames=args.min_frames, drop_suffixes=tuple(args.drop_suffix or [".unknown_video"]),
        on_unreadable=args.on_unreadable, backend=args.backend, workers=args.workers, cache_file=args.probe_cache)
    report_path = args.report or args.output + ".media_report.json"
    Path(report_path).write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    _write_records_with_manifest(kept, args.output, {"source": {"kind": "filter-media", "input": args.input,
                                                                "report": report_path, **report.criteria}})
    summary = report.to_dict()
    print(json.dumps({"output": args.output, "kept": summary["kept"], "dropped": summary["dropped"],
                      "dropped_by_reason": summary["dropped_by_reason"], "report": report_path}))
    return 0


# ---------------------------------------------------------------------- select
def _add_select(sub) -> None:
    p = sub.add_parser("select", help="take a subset of records: random sample or an ID manifest")
    p.add_argument("--input", required=True)
    p.add_argument("--taxonomy", required=True)
    how = p.add_mutually_exclusive_group(required=True)
    how.add_argument("--n", type=int, help="sample size")
    how.add_argument("--ids-file", help="ID manifest (one ID per line, in order; see configs/data/manifests)")
    p.add_argument("--seed", type=int, help="--n: random seed (required)")
    p.add_argument("--method", choices=["random_sample", "sorted_index_sample"],
                   help="--n: random_sample keeps sampling order (SafeWatch), sorted_index_sample keeps input order "
                        "(XD-Violence)")
    p.add_argument("--output", required=True)
    p.add_argument("--write-manifest", help="also write the selected IDs as a manifest file")
    p.set_defaults(func=_cmd_select)


def _cmd_select(args) -> int:
    from .data.records import load_records
    from .data.sources import read_id_manifest, select_by_manifest, select_records, write_id_manifest
    from .taxonomy import load_taxonomy

    taxonomy = load_taxonomy(args.taxonomy)
    records = load_records(args.input, taxonomy)
    if args.ids_file:
        selected = select_by_manifest(records, read_id_manifest(args.ids_file))
        info = {"kind": "select", "input": args.input, "ids_file": args.ids_file}
    else:
        if args.seed is None or args.method is None:
            raise SystemExit("--n needs --seed and --method")
        selected = select_records(records, args.n, seed=args.seed, method=args.method)
        info = {"kind": "select", "input": args.input, "n": args.n, "seed": args.seed, "method": args.method}
    manifest = _write_records_with_manifest(selected, args.output, {"source": info})
    if args.write_manifest:
        write_id_manifest(selected, args.write_manifest, header=[f"{len(selected)} IDs selected from {args.input}"])
    print(json.dumps({k: manifest[k] for k in ("output", "records", "unique_ids", "benign", "label_counts")}))
    return 0


# -------------------------------------------------------------------- training
def _add_train(sub) -> None:
    p = sub.add_parser("train", help="launch SFT (LLaMA-Factory) or GRPO/ATPO (EasyR1) training from a config")
    p.add_argument("--config", required=True, help="training config (configs/sft, configs/grpo, configs/atpo, ...)")
    p.add_argument("--var", action="append", metavar="NAME=VALUE", help="value for a config variable")
    p.add_argument("--output-dir", required=True, help="run directory (config, logs, checkpoints)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override a backend setting, e.g. trainer.n_gpus_per_node=8 (recorded in the run file)")
    p.add_argument("--validation", choices=["frozen", "historical"],
                   help="EasyR1: validation reward mode (default from the config, normally 'frozen')")
    p.add_argument("--resume", action="store_true", help="EasyR1: continue from the last checkpoint in --output-dir")
    p.add_argument("--logger", action="append", help="loggers, e.g. --logger file --logger wandb")
    p.add_argument("--nproc-per-node", type=int, help="LLaMA-Factory: number of GPUs (torchrun)")
    p.add_argument("--skip-checks", action="store_true", help="skip preflight checks of data, model, and paths")
    p.add_argument("--dry-run", action="store_true", help="write the backend config and print the command only")
    p.set_defaults(func=_cmd_train)


def _cmd_train(args) -> int:
    from .configuration import parse_scalar
    from .training import compose_easyr1_config, launch_easyr1, load_training_config
    from .training.llamafactory import compose_llamafactory_config, launch_llamafactory
    from .training.preflight import preflight_easyr1, preflight_llamafactory

    config = load_training_config(args.config)
    variables = _parse_vars(args.var)
    overrides = {k: parse_scalar(v) for k, v in _parse_vars(args.set).items()}
    if config.kind == "easyr1":
        tree = compose_easyr1_config(config, variables=variables, output_dir=args.output_dir, overrides=overrides,
                                     validation=args.validation, resume=args.resume, loggers=args.logger)
        if not args.skip_checks:
            preflight_easyr1(config, tree, output_dir=args.output_dir)
        return launch_easyr1(config, tree, output_dir=args.output_dir, variables=variables, dry_run=args.dry_run)
    tree = compose_llamafactory_config(config, variables=variables, output_dir=args.output_dir, overrides=overrides,
                                       loggers=args.logger)
    if not args.skip_checks:
        preflight_llamafactory(config, tree)
    return launch_llamafactory("train", tree, output_dir=args.output_dir, config=config, variables=variables,
                               nproc_per_node=args.nproc_per_node, dry_run=args.dry_run)


def _add_merge_lora(sub) -> None:
    p = sub.add_parser("merge-lora", help="merge an SFT LoRA adapter into its base model (llamafactory-cli export)")
    p.add_argument("--config", required=True, help="the SFT config used for training (its 'export' section)")
    p.add_argument("--var", action="append", metavar="NAME=VALUE")
    p.add_argument("--adapter", required=True, help="LoRA output directory (or a checkpoint inside it)")
    p.add_argument("--output-dir", required=True, help="directory for the merged model")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_cmd_merge_lora)


def _cmd_merge_lora(args) -> int:
    from .training import load_training_config
    from .training.llamafactory import compose_merge_config, launch_llamafactory

    config = load_training_config(args.config)
    variables = _parse_vars(args.var)
    tree = compose_merge_config(config, variables=variables, adapter=args.adapter, export_dir=args.output_dir)
    run_dir = Path(args.output_dir).resolve().parent / (Path(args.output_dir).name + "_merge_run")
    return launch_llamafactory("export", tree, output_dir=run_dir, config=config, variables=variables,
                               dry_run=args.dry_run)


def _add_inspect_state(sub) -> None:
    p = sub.add_parser("inspect-state", help="show the ATPO controller state saved with an EasyR1 checkpoint")
    p.add_argument("path", help="global_step_* directory or a reward_state.json file")
    p.add_argument("--json", action="store_true", help="print the raw state")
    p.set_defaults(func=_cmd_inspect_state)


def _cmd_inspect_state(args) -> int:
    path = Path(args.path)
    files = [path] if path.is_file() else [p for p in (path / "reward_state.json", path / "val_reward_state.json")
                                            if p.exists()]
    if not files:
        raise SystemExit(f"no reward_state.json under {path}")
    for file in files:
        state = json.loads(file.read_text())
        if args.json:
            print(json.dumps(state, indent=2))
            continue
        print(f"{file.name}: reward={state.get('reward')} taxonomy={state.get('taxonomy')} "
              f"batches_scored={state.get('batches_scored')}")
        ctrl = state.get("controller")
        if not ctrl:
            print("  (static reward, no controller)")
            continue
        readable = ctrl["readable"]
        ids = ctrl["category_ids"] if ctrl["mode"] == "category" else ["global"]
        print(f"  mode={ctrl['mode']} updates={ctrl['num_updates']} initialized={ctrl['initialized']}")
        print(f"  {'category':<10}{'alpha':>10}{'beta':>10}{'FP_ema':>12}{'FN_ema':>12}{'logit_u':>10}")
        for i, cid in enumerate(ids):
            print(f"  {cid:<10}{readable['alpha'][i]:>10.4f}{readable['beta'][i]:>10.4f}"
                  f"{readable['fp_ema'][i]:>12.4f}{readable['fn_ema'][i]:>12.4f}{readable['logit_u'][i]:>10.4f}")
    return 0


# ---------------------------------------------------------------------- export
def _add_export(sub) -> None:
    p = sub.add_parser("export", help="package a checkpoint for inference (atpo_config.json + checksums)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="EasyR1 global_step_* directory (FSDP shards are merged locally)")
    src.add_argument("--hf-model", help="directory with Hugging Face weights (e.g. a merged SFT model)")
    p.add_argument("--output-dir", required=True)
    cfg = p.add_mutually_exclusive_group()
    cfg.add_argument("--preset", help="inference preset (use the historical preset for paper checkpoints)")
    cfg.add_argument("--inference-config", help="inference config JSON")
    cfg.add_argument("--training-run", help="atpo train output dir: derive the config from the training prompt")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override inference settings")
    p.add_argument("--processor-source", help="--hf-model: copy missing processor files from this model")
    p.add_argument("--remerge", action="store_true", help="--checkpoint: rerun the merger even if weights exist")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_cmd_export)


def _cmd_export(args) -> int:
    from .export import export_easyr1_checkpoint, export_hf_model

    overrides = _parse_overrides(args.set)
    if args.checkpoint:
        result = export_easyr1_checkpoint(args.checkpoint, args.output_dir, training_run=args.training_run,
                                          preset=args.preset, inference_config=args.inference_config,
                                          overrides=overrides, remerge=args.remerge, dry_run=args.dry_run)
    else:
        if args.training_run:
            raise SystemExit("--training-run applies to --checkpoint exports")
        result = export_hf_model(args.hf_model, args.output_dir, preset=args.preset,
                                 inference_config=args.inference_config, overrides=overrides,
                                 processor_source=args.processor_source)
    print(json.dumps(result, indent=2, default=str))
    return 0


COMMANDS = [_add_presets, _add_models, _add_show_config, _add_predict, _add_evaluate, _add_prepare, _add_validate_data,
            _add_import_data, _add_filter_media, _add_select, _add_train, _add_merge_lora, _add_inspect_state, _add_export]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atpo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for add in COMMANDS:
        add(sub)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
