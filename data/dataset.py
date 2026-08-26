"""NPZ dataset access shared by all baseline models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_archive(path: str | Path) -> dict[str, np.ndarray | dict[str, object]]:
    """Load an archive without enabling pickle and decode its JSON metadata."""

    with np.load(path, allow_pickle=False) as source:
        archive: dict[str, np.ndarray | dict[str, object]] = {
            key: source[key].copy() for key in source.files if key != "metadata_json"
        }
        archive["metadata"] = json.loads(str(source["metadata_json"].item()))
    return archive


class PowerFlowDataset(Dataset[torch.Tensor]):
    """Return native model states from one fixed train/validation/test split."""

    SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}

    def __init__(
        self,
        archive: dict[str, np.ndarray | dict[str, object]],
        representation: str,
        split: str,
    ) -> None:
        if representation not in {"common", "wang"}:
            raise ValueError("representation must be 'common' or 'wang'.")
        if split not in self.SPLIT_CODE:
            raise ValueError(f"Unknown split: {split}")
        key = "state_common" if representation == "common" else "state_wang"
        states = np.asarray(archive[key], dtype=np.float32)
        split_code = np.asarray(archive["split"], dtype=np.int8)
        self.states = torch.from_numpy(states[split_code == self.SPLIT_CODE[split]])

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.states[index]

