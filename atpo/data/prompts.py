"""Task prompts, answer-format instructions, and supervised targets.

Historical prompts are used verbatim from ``atpo/resources/prompts`` (see
``prompts/README.md`` for their provenance). For a custom taxonomy,
:func:`render_task_prompt` produces a prompt with the same structure as the
historical SafeWatch/XD-Violence prompts: category list, definitions, and the
``BLOCK = { "ID(Name)": boolean, ... }`` output format that the parser expects.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Iterable, Literal, Optional

from ..taxonomy import Taxonomy

TargetStyle = Literal["simple", "json"]

# Answer-format instructions appended by the EasyR1 format templates
# (atpo/resources/prompts/formats/*.jinja), rendered as "{{ content | trim }}<suffix>".
FORMAT_TEMPLATES = {
    "think_answer": "safewatch.jinja",
    "triple_newline": "qwen3_2.jinja",
}


def read_resource_prompt(relpath: str, *, strip: bool = False) -> str:
    node = resources.files("atpo.resources.prompts")
    for part in relpath.split("/"):
        node = node / part
    text = node.read_bytes().decode("utf-8")
    return text.strip() if strip else text


def format_template_path(response_format: str) -> str:
    """Filesystem path of the EasyR1 format template for a response format."""
    if response_format not in FORMAT_TEMPLATES:
        raise ValueError(f"no format template for response_format {response_format!r}")
    return str(resources.files("atpo.resources.prompts") / "formats" / FORMAT_TEMPLATES[response_format])


def apply_format_template(content: str, response_format: str) -> str:
    """Render a format template exactly as EasyR1's dataset does (Jinja2, template stripped)."""
    from jinja2 import Template

    if response_format == "plain":
        return content
    template = read_resource_prompt(f"formats/{FORMAT_TEMPLATES[response_format]}")
    return Template(template.strip()).render(content=content)


def render_task_prompt(
    taxonomy: Taxonomy,
    *,
    intro: Optional[str] = None,
    subject: str = "categories",
    positive_rule: Optional[str] = None,
) -> str:
    """Generate a task prompt for any taxonomy, following the historical prompt structure."""
    block = taxonomy.block_name
    upper = subject.upper()
    intro = intro or (
        taxonomy.description
        or "You are a video classification model. Your task is to analyze the video and decide which of the "
        "following categories it contains."
    )
    listing = " ".join(f"{c.id}: {c.name};" for c in taxonomy.categories)
    boxes = []
    for c in taxonomy.categories:
        body = (c.description or c.name).strip()
        boxes.append(f"<box>\n{c.id}: {c.name}:\n{body}\n</box>")
    keys = ",\n".join(f'"{taxonomy.output_key(c)}": boolean' for c in taxonomy.categories)
    positive_rule = positive_rule or (
        "**Set the value to true ONLY for the categories present in the video. "
        "If none of them is present, set all values to false.**"
    )
    return (
        f"{intro.strip()}\n\n"
        f"**Analyze the video for the presence of the following {subject}:**\n\n"
        f"<BEGIN {upper}>\n{listing}\n<END {upper}>\n\n"
        f"**Refer to the detailed explanations of each category below:**\n\n"
        f"<BEGIN {upper} EXPLANATIONS>\n" + "\n".join(boxes) + f"\n<END {upper} EXPLANATIONS>\n\n"
        f"**Provide the result in the following JSON format:**\n\n"
        f"{block} = {{\n{keys}\n}}\n\n"
        f"{positive_rule}\n\n"
        f'**The result must be the JSON format of "{block}", beginning with "{block}:"**'
    )


def with_video_token(prompt: str, *, placement: Literal["post", "pre"] = "post", token: str = "<video>") -> str:
    """Attach the video placeholder the way the paper's data conversion did.

    ``post``: ``prompt.strip() + " <video>"`` (all SafeWatch/XD-Violence training
    files of the paper). ``pre``: ``"<video> " + prompt.strip()``.
    """
    if token in prompt:
        raise ValueError(f"prompt already contains {token!r}")
    if placement == "post":
        return f"{prompt.strip()} {token}"
    if placement == "pre":
        return f"{token} {prompt.strip()}"
    raise ValueError("placement must be 'post' or 'pre'")


def sft_target(
    labels: Iterable[str],
    taxonomy: Taxonomy,
    *,
    style: TargetStyle,
    block_name: Optional[str] = None,
) -> str:
    """Supervised target text.

    ``simple``: ``"GUARDRAIL: C2, C3"`` (benign: ``"GUARDRAIL: "``), as in the SafeWatch
    SFT data (target template sv5). ``json``: ``"GUARDRAIL: " + json.dumps(decisions,
    indent=4)`` with full keys, as in the XD-Violence SFT data (sv1). ``block_name``
    defaults to the taxonomy's; the historical XD-Violence data used ``GUARDRAIL``.
    """
    block = block_name or taxonomy.block_name
    present = set(taxonomy.resolve_labels(list(labels)))
    if style == "simple":
        return f"{block}: " + ", ".join(cid for cid in taxonomy.ids if cid in present)
    if style == "json":
        decisions = {taxonomy.output_key(c): c.id in present for c in taxonomy.categories}
        return f"{block}: " + json.dumps(decisions, indent=4, ensure_ascii=False)
    raise ValueError("style must be 'simple' or 'json'")
