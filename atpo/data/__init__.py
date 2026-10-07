"""Data preparation: canonical records, validation, prompts, and training-file builders."""

from .adapters import records_from_easyr1_parquet, records_from_sharegpt
from .builders import (build_easyr1_rows, build_sharegpt_rows, llamafactory_dataset_info, oversample_rows,
                       write_easyr1_parquet, write_jsonl)
from .prompts import apply_format_template, render_task_prompt, sft_target, with_video_token
from .records import (DataValidationError, Record, ValidationReport, check_split_overlap, load_records,
                      parse_record, validate_records, write_records)
from .recipe import RecipeError, run_recipe

__all__ = [
    "DataValidationError",
    "Record",
    "RecipeError",
    "ValidationReport",
    "apply_format_template",
    "build_easyr1_rows",
    "build_sharegpt_rows",
    "check_split_overlap",
    "llamafactory_dataset_info",
    "load_records",
    "oversample_rows",
    "parse_record",
    "records_from_easyr1_parquet",
    "records_from_sharegpt",
    "render_task_prompt",
    "run_recipe",
    "sft_target",
    "validate_records",
    "with_video_token",
    "write_easyr1_parquet",
    "write_jsonl",
    "write_records",
]
