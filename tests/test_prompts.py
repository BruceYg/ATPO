"""Bundled prompts and format templates match their checksums."""

from __future__ import annotations

import hashlib

from helpers import REPO_ROOT

PROMPTS = REPO_ROOT / "atpo" / "resources" / "prompts"


def test_checksums_cover_every_bundled_file():
    listed = {}
    for line in (PROMPTS / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split()
        listed[rel] = digest
    bundled = {p.relative_to(PROMPTS).as_posix() for p in PROMPTS.rglob("*") if p.suffix in (".txt", ".jinja")}
    assert set(listed) == bundled
    for rel, digest in listed.items():
        assert hashlib.sha256((PROMPTS / rel).read_bytes()).hexdigest() == digest, rel
