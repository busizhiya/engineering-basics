from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def split_by_files(files: Sequence[Path], ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Dict[str, List[Path]]:
    files = sorted(files)
    n = len(files)
    if n < 3:
        raise ValueError("文件数过少，至少需要 3 个已清洗样本")

    n_train = max(1, int(n * ratios[0]))
    n_val = max(1, int(n * ratios[1]))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = max(1, n_train - 1)

    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    if not test_files:
        test_files = [val_files[-1]]
        val_files = val_files[:-1]

    return {
        "train": list(train_files),
        "val": list(val_files),
        "test": list(test_files),
    }


def save_loss_curve(train_loss: List[float], val_loss: List[float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_json(data: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
