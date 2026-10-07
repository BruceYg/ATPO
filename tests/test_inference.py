"""Inference API tests that run without a GPU (fake processor, vision function and backend)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atpo.cli import main as cli_main
from atpo.inference import InferenceConfig, VideoSafetyModel, build_messages, list_presets, load_preset
from atpo.inference.backends import Backend, GenerationOutput
from atpo.inference.model import ConfigResolutionError, resolve_inference_config
from atpo.inference.runner import load_inputs, run_predictions

from helpers import REPO_ROOT

GOOD = ('<think>A person is punched.</think>\n<answer>GUARDRAIL: {"C1(Sexual Content)": false, '
        '"C2(Harassment & Bullying)": true, "C3(Threats, Violence & Harm)": true, '
        '"C4(False & Deceptive Information)": false, "C5(Illegal/Regulated Activities)": false, '
        '"C6(Hateful Content & Extremism)": false}</answer>')
BENIGN = GOOD.replace("true", "false")
PARTIAL = '<think>x</think><answer>GUARDRAIL: {"C1(Sexual Content)": false, "C3(Threats, Violence & Harm)": false}</answer>'


class FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return json.dumps(messages, sort_keys=True)


class FakeBackend(Backend):
    name = "fake"

    def __init__(self, responses):
        self.responses = responses
        self.seen = []

    def generate(self, requests, generation):
        out = []
        for request in requests:
            messages = json.loads(request.prompt)
            video = next(i["video"] for i in messages[-1]["content"] if i.get("type") == "video")
            self.seen.append(video)
            response = self.responses[Path(video).name]
            if isinstance(response, Exception):
                out.append(GenerationOutput(text=None, error=str(response)))
            else:
                out.append(GenerationOutput(text=response, info={"finish_reason": "stop"}))
        return out


def fake_vision(messages):
    video = next(i for i in messages[-1]["content"] if i.get("type") == "video")
    if Path(video["video"]).name.startswith("corrupt"):
        raise RuntimeError("cannot decode")
    return None, [("frames", {"fps": 1.0})], {"do_sample_frames": False}


@pytest.fixture
def videos(tmp_path):
    names = ["unsafe.mp4", "benign.mp4", "partial.mp4", "garbage.mp4", "fail.mp4", "corrupt.mp4"]
    for name in names:
        (tmp_path / name).write_bytes(b"\x00")
    return tmp_path, names


def make_model(responses, preset="safewatch-qwen2.5vl-rl"):
    return VideoSafetyModel(FakeBackend(responses), FakeProcessor(), load_preset(preset), vision_fn=fake_vision)


RESPONSES = {"unsafe.mp4": GOOD, "benign.mp4": BENIGN, "partial.mp4": PARTIAL, "garbage.mp4": "I am not sure.",
             "fail.mp4": RuntimeError("prompt too long"), "corrupt.mp4": GOOD}


def test_presets_load_with_the_paper_prompts():
    names = list_presets()
    assert {"safewatch-qwen2.5vl-rl", "safewatch-qwen3vl-rl", "xdviolence-qwen2.5vl-rl",
            "xdviolence-qwen3vl-rl"} <= set(names)
    prompts = REPO_ROOT / "atpo" / "resources" / "prompts"
    # The default SafeWatch query is used byte for byte (including its trailing newline).
    default = (prompts / "safewatch" / "inference_default_rl.txt").read_text(encoding="utf-8")
    assert load_preset("safewatch-qwen2.5vl-rl").prompt == default
    assert default.endswith("\n")
    # Query files are read and stripped.
    assert load_preset("xdviolence-qwen2.5vl-rl").prompt == (prompts / "xdviolence" / "v1reason.txt").read_text().strip()
    assert load_preset("safewatch-qwen3vl-rl").prompt == (prompts / "safewatch" / "v3_qwen3.txt").read_text().strip()


def test_messages_match_the_paper_layout():
    config = load_preset("safewatch-qwen2.5vl-rl")
    messages = build_messages(config, "videos/v.mp4")
    # as in the paper's evaluation (video after the text)
    assert messages == [
        {"role": "system", "content": "You are a helpful video content moderation assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": config.prompt},
            {"type": "video", "video": "videos/v.mp4", "max_pixels": 200704, "fps": 1.0, "min_pixels": 12544},
        ]},
    ]
    placeholder = config.with_overrides({"video.position": "placeholder", "system_prompt": None,
                                         "prompt": "Task text <video> Format text."})
    content = build_messages(placeholder, "v.mp4")
    assert [m["role"] for m in content] == ["user"]
    assert [i["type"] for i in content[0]["content"]] == ["text", "video", "text"]


def test_predict_batch_statuses_and_order(videos):
    root, names = videos
    model = make_model(RESPONSES)
    paths = [str(root / n) for n in names] + [str(root / "missing.mp4")]
    preds = model.predict_batch(paths, batch_size=4)
    by = {Path(p.video).name: p for p in preds}
    assert [Path(p.video).name for p in preds] == names + ["missing.mp4"]
    assert by["unsafe.mp4"].status == "ok" and by["unsafe.mp4"].labels == ["C2", "C3"]
    assert by["unsafe.mp4"].label_names == ["Harassment & Bullying", "Threats, Violence & Harm"]
    assert by["unsafe.mp4"].unsafe is True and by["unsafe.mp4"].explanation == "A person is punched."
    assert by["benign.mp4"].status == "ok" and by["benign.mp4"].labels == [] and by["benign.mp4"].unsafe is False
    partial = by["partial.mp4"]
    assert partial.status == "partial" and partial.unsafe is None
    assert partial.missing_categories == ["C2", "C4", "C5", "C6"]
    for name, status in (("garbage.mp4", "parse_error"), ("fail.mp4", "generation_error"),
                         ("corrupt.mp4", "video_error"), ("missing.mp4", "video_error")):
        assert by[name].status == status
        assert by[name].labels is None and by[name].unsafe is None and by[name].decisions is None
    assert by["garbage.mp4"].parse["historical_labels"] == []
    assert "missing.mp4" not in model.backend.seen and "corrupt.mp4" not in [Path(v).name for v in model.backend.seen]


def test_runner_resume_and_metadata(videos, tmp_path):
    root, names = videos
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps({"id": f"id{i}", "video": n, "labels": []}) + "\n" for i, n in enumerate(names)))
    ids, paths, info = load_inputs(input_path=str(data))
    assert paths[0] == str(root / names[0]) and info["count"] == len(names)
    out = tmp_path / "out" / "preds.jsonl"
    model = make_model(RESPONSES)
    meta = run_predictions(model, ids[:3], paths[:3], out, input_info=info, batch_size=2)
    assert meta["status"] == "complete" and meta["num_records"] == 3
    with pytest.raises(FileExistsError):
        run_predictions(model, ids, paths, out)
    model2 = make_model(RESPONSES)
    meta = run_predictions(model2, ids, paths, out, resume=True)
    assert [Path(v).name for v in model2.backend.seen] == names[3:5]  # only the remaining (corrupt never generates)
    assert meta["num_records"] == len(names) and meta["resumed_from"] == 3
    assert meta["counts"] == {"ok": 2, "partial": 1, "parse_error": 1, "generation_error": 1, "video_error": 1}
    sidecar = json.loads((out.parent / "preds.jsonl.meta.json").read_text())
    assert sidecar["config"]["prompt"]["sha256"] == load_preset("safewatch-qwen2.5vl-rl").prompt_sha256
    assert sidecar["config_fingerprint"] == model.config.fingerprint()
    records = [json.loads(l) for l in out.read_text().splitlines()]
    assert len({r["id"] for r in records}) == len(names)


def test_evaluate_cli_on_release_predictions(videos, tmp_path, capsys):
    root, names = videos
    gt = tmp_path / "gt.jsonl"
    labels = {"unsafe.mp4": ["C2", "C3"], "benign.mp4": [], "partial.mp4": ["C1"], "garbage.mp4": [],
              "fail.mp4": ["C4"], "corrupt.mp4": []}
    gt.write_text("".join(json.dumps({"id": n, "video": n, "labels": labels[n]}) + "\n" for n in names))
    out = tmp_path / "preds.jsonl"
    model = make_model(RESPONSES)
    ids, paths, info = load_inputs(input_path=str(gt))
    run_predictions(model, ids, paths, out, input_info=info)
    result_path = tmp_path / "metrics.json"
    assert cli_main(["evaluate", "--predictions", str(out), "--ground-truth", str(gt), "--taxonomy", "safewatch",
                     "--invalid-policy", "exclude", "--output", str(result_path)]) == 0
    result = json.loads(result_path.read_text())
    assert result["coverage"]["scored_samples"] == 3 and result["coverage"]["invalid"] == 3
    with pytest.raises(Exception):
        cli_main(["evaluate", "--predictions", str(out), "--ground-truth", str(gt), "--taxonomy", "safewatch",
                  "--invalid-policy", "error"])
    # Historical convention: parse failures scored with historical labels, runtime errors excluded.
    assert cli_main(["evaluate", "--predictions", str(out), "--ground-truth", str(gt), "--taxonomy", "safewatch",
                     "--invalid-policy", "exclude", "--label-source", "historical", "--output", str(result_path)]) == 0
    result = json.loads(result_path.read_text())
    assert result["coverage"]["scored_samples"] == 4
    assert result["coverage"]["parse_errors_scored_with_historical_labels"] == 1


def test_config_roundtrip_and_validation(tmp_path):
    config = load_preset("xdviolence-qwen3vl-rl")
    again = InferenceConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert again.fingerprint() == config.fingerprint() and again.prompt == config.prompt
    data = config.to_dict()
    data["prompt"]["text"] += " "
    with pytest.raises(ValueError, match="sha256"):
        InferenceConfig.from_dict(data)
    with pytest.raises(ValueError, match="placeholder"):
        config.with_overrides({"prompt": "has <video> inside"})
    smaller = config.with_overrides({"video.max_pixels": 50176, "generation.max_new_tokens": 256})
    assert smaller.video.max_pixels == 50176 and smaller.fingerprint() != config.fingerprint()
    with pytest.raises(ValueError):
        config.with_overrides({"video.nonexistent": 1})


def test_config_resolution_precedence(tmp_path):
    with pytest.raises(ConfigResolutionError, match="atpo_config.json"):
        resolve_inference_config(str(tmp_path))
    load_preset("safewatch-qwen3vl-rl").save(tmp_path)
    config, info = resolve_inference_config(str(tmp_path))
    assert info == {"source": "checkpoint"} and config.preset == "safewatch-qwen3vl-rl"
    with pytest.warns(UserWarning, match="differs"):
        config, info = resolve_inference_config(str(tmp_path), preset="safewatch-qwen2.5vl-rl")
    assert info["checkpoint_config_overridden"] is True


def test_show_config_prints_exact_prompt(capsys):
    assert cli_main(["show-config", "--preset", "safewatch-qwen2.5vl-rl", "--prompt"]) == 0
    assert capsys.readouterr().out == load_preset("safewatch-qwen2.5vl-rl").prompt
