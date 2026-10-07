"""Picklable stand-ins used by the EasyR1 integration tests (imported inside Ray workers)."""

from __future__ import annotations

import os


class TableTokenizer:
    """Decodes a one-token response ``[i]`` to ``responses[i]``."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:  # noqa: ARG002 - EasyR1 signature
        return self.responses[int(ids[0])]


class FakeWorkerGroup:
    """Records checkpoint calls instead of saving model shards."""

    def __init__(self):
        self.saved: list[str] = []
        self.loaded: list[str] = []

    def save_checkpoint(self, path: str, save_model_only: bool = False) -> None:
        # The FSDP checkpoint manager creates the actor folder (and its parent).
        os.makedirs(path, exist_ok=True)
        self.saved.append(path)

    def load_checkpoint(self, path: str) -> None:
        self.loaded.append(path)


class FakeDataLoader:
    def __init__(self):
        self.state = {"epoch": 0, "consumed": 0}

    def state_dict(self) -> dict:
        return dict(self.state)

    def load_state_dict(self, state: dict) -> None:
        self.state = dict(state)
