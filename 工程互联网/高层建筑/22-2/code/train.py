from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_utils import DRIVER_COLS, RESPONSE_COLS, load_cleaned_csv_data, sliding_windows
from src.models import ConvBiLSTMDualHead
from src.train_utils import (
    SequenceDataset,
    save_json,
    save_loss_curve,
    set_seed,
    split_by_files,
)

TASK_CONFIG = {
    "forward": {
        "x_key": "x_forward",
        "y_key": "y_forward",
        "input_names": DRIVER_COLS,
        "output_names": RESPONSE_COLS,
        "head_split": (15, 2),
    },
    "inverse": {
        "x_key": "x_inverse",
        "y_key": "y_inverse",
        "input_names": RESPONSE_COLS,
        "output_names": DRIVER_COLS,
        "head_split": (2, 1),
    },
}


def collect_windows(files: List[Path], x_key: str, y_key: str, seq_len: int, stride: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    all_x = []
    all_y = []
    used_files = []
    for f in files:
        data = load_cleaned_csv_data(f)
        x = data[x_key].astype(np.float32)
        y = data[y_key].astype(np.float32)
        wx, wy = sliding_windows(x, y, seq_len=seq_len, stride=stride)
        if len(wx) == 0:
            continue
        all_x.append(wx)
        all_y.append(wy)
        used_files.append(f.name)

    if not all_x:
        raise ValueError("没有可用于训练的窗口，请减小 seq_len 或检查预处理输出")

    return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0), used_files


def evaluate_rmse(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, np.ndarray]:
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yp = model(xb)
            preds.append(yp.detach().cpu().numpy())
            trues.append(yb.detach().cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)

    err = pred - true
    rmse_ch = np.sqrt(np.mean(err**2, axis=(0, 1)))
    rmse_all = float(np.sqrt(np.mean(err**2)))
    return rmse_all, rmse_ch


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 Conv1D+BiLSTM+双头MLP 序列模型")
    parser.add_argument("--task", type=str, choices=["forward", "inverse"], required=True)
    parser.add_argument("--data_dir", type=str, default="processed/cleaned")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    cfg = TASK_CONFIG[args.task]
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("*.csv"))
    if len(csv_files) < 3:
        raise ValueError("清洗后的 csv 文件少于 3 个，无法稳定划分 train/val/test")

    split = split_by_files(csv_files, ratios=(0.7, 0.15, 0.15))

    x_train, y_train, train_used = collect_windows(split["train"], cfg["x_key"], cfg["y_key"], args.seq_len, args.stride)
    x_val, y_val, val_used = collect_windows(split["val"], cfg["x_key"], cfg["y_key"], args.seq_len, args.stride)
    x_test, y_test, test_used = collect_windows(split["test"], cfg["x_key"], cfg["y_key"], args.seq_len, args.stride)

    train_ds = SequenceDataset(x_train, y_train)
    val_ds = SequenceDataset(x_val, y_val)
    test_ds = SequenceDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    input_dim = x_train.shape[-1]
    output_dim = y_train.shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvBiLSTMDualHead(
        input_dim=input_dim,
        output_dim=output_dim,
        head_split=cfg["head_split"],
        conv_channels=64,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val = math.inf
    best_epoch = -1
    bad_epochs = 0
    train_loss_hist: List[float] = []
    val_loss_hist: List[float] = []

    ckpt_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            yp = model(xb)
            loss = criterion(yp, yb)
            loss.backward()
            optimizer.step()

            tr_loss += float(loss.item()) * xb.size(0)

        tr_loss /= len(train_ds)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                yp = model(xb)
                loss = criterion(yp, yb)
                va_loss += float(loss.item()) * xb.size(0)
        va_loss /= len(val_ds)

        train_loss_hist.append(tr_loss)
        val_loss_hist.append(va_loss)
        print(f"[Epoch {epoch:03d}] train={tr_loss:.6f} val={va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "task": args.task,
                    "model_state": model.state_dict(),
                    "model_cfg": {
                        "input_dim": input_dim,
                        "output_dim": output_dim,
                        "head_split": cfg["head_split"],
                        "conv_channels": 64,
                        "lstm_hidden": 128,
                        "lstm_layers": 2,
                        "dropout": 0.2,
                    },
                    "normalize": False,
                    "seq_len": args.seq_len,
                    "stride": args.stride,
                    "input_names": cfg["input_names"],
                    "output_names": cfg["output_names"],
                    "split_files": {
                        "train": train_used,
                        "val": val_used,
                        "test": test_used,
                    },
                    "train_args": vars(args),
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("触发早停")
                break

    save_loss_curve(train_loss_hist, val_loss_hist, out_dir / "loss_curve.png")

    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state"])

    rmse_all, rmse_ch = evaluate_rmse(model, test_loader, device=device)
    metrics = {
        "task": args.task,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_rmse_all": rmse_all,
        "test_rmse_by_channel": {
            name: float(v) for name, v in zip(cfg["output_names"], rmse_ch)
        },
        "windows": {
            "train": int(len(train_ds)),
            "val": int(len(val_ds)),
            "test": int(len(test_ds)),
        },
        "checkpoint": str(ckpt_path),
        "loss_curve": str(out_dir / "loss_curve.png"),
    }
    save_json(metrics, out_dir / "metrics.json")

    print("训练完成")
    print(f"最佳轮次: {best_epoch}")
    print(f"测试集整体 RMSE: {rmse_all:.6f}")
    print(f"模型保存: {ckpt_path}")


if __name__ == "__main__":
    main()
