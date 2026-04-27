from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import detrend

from src.data_utils import bandpass_filter, estimate_fs_from_time
from src.models import ConvBiLSTMDualHead


def load_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ckpt["model_cfg"]
    model = ConvBiLSTMDualHead(
        input_dim=model_cfg["input_dim"],
        output_dim=model_cfg["output_dim"],
        head_split=tuple(model_cfg["head_split"]),
        conv_channels=model_cfg["conv_channels"],
        lstm_hidden=model_cfg["lstm_hidden"],
        lstm_layers=model_cfg["lstm_layers"],
        dropout=model_cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return ckpt, model


def _parse_time_value(v: object) -> pd.Timestamp:
    if pd.isna(v):
        return pd.NaT
    s = str(v).strip()
    if s.startswith('="') and s.endswith('"'):
        s = s[2:-1]
    elif s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    return pd.to_datetime(s, errors="coerce")


def load_infer_csv(
    csv_path: Path,
    input_names: list,
    output_names: list,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    df = pd.read_csv(csv_path)

    missing_in = [c for c in input_names if c not in df.columns]
    if missing_in:
        raise ValueError(f"{csv_path.name} 缺少输入列: {missing_in}")

    has_output = all(c in df.columns for c in output_names)

    if "time_s" in df.columns:
        use_cols = ["time_s"] + input_names + (output_names if has_output else [])
        df = df[use_cols].dropna().reset_index(drop=True)
        if df.empty:
            raise ValueError(f"{csv_path.name} 清洗后无有效数据")

        time_s = df["time_s"].to_numpy(dtype=np.float32)
        x = df[input_names].to_numpy(dtype=np.float32)
        y_true = df[output_names].to_numpy(dtype=np.float32) if has_output else None
        return x, y_true, time_s

    if "time" in df.columns:
        use_cols = ["time"] + input_names + (output_names if has_output else [])
        df = df[use_cols].copy()
        df["time"] = df["time"].map(_parse_time_value)
        df = df.dropna(subset=["time"] + input_names + (output_names if has_output else [])).reset_index(drop=True)
        if df.empty:
            raise ValueError(f"{csv_path.name} 清洗后无有效数据")

        fs = estimate_fs_from_time(df["time"])
        t0 = df["time"].iloc[0]
        time_s = (df["time"] - t0).dt.total_seconds().to_numpy(dtype=np.float32)

        x = df[input_names].to_numpy(dtype=np.float64)
        x = np.column_stack([detrend(x[:, i], type="linear") for i in range(x.shape[1])])
        x = bandpass_filter(x, fs=fs, low_hz=0.1, high_hz=20.0, order=4).astype(np.float32)

        if has_output:
            y = df[output_names].to_numpy(dtype=np.float64)
            y = np.column_stack([detrend(y[:, i], type="linear") for i in range(y.shape[1])])
            y_true = bandpass_filter(y, fs=fs, low_hz=0.1, high_hz=20.0, order=4).astype(np.float32)
        else:
            y_true = None

        return x, y_true, time_s

    raise ValueError(f"{csv_path.name} 缺少 time_s 或 time 列")


def predict_full_sequence(model, x: np.ndarray, seq_len: int, stride: int, device: torch.device) -> np.ndarray:
    n, in_dim = x.shape
    if n <= 0:
        raise ValueError("输入序列为空")

    windows = []
    starts = []
    if n < seq_len:
        pad = np.repeat(x[-1:, :], seq_len - n, axis=0)
        w = np.concatenate([x, pad], axis=0)
        windows.append(w)
        starts.append(0)
    else:
        for st in range(0, n - seq_len + 1, stride):
            windows.append(x[st : st + seq_len])
            starts.append(st)
        if starts[-1] + seq_len < n:
            st = n - seq_len
            windows.append(x[st : st + seq_len])
            starts.append(st)

    windows_np = np.stack(windows).astype(np.float32)

    with torch.no_grad():
        xb = torch.from_numpy(windows_np).to(device)
        yb = model(xb).cpu().numpy()

    out_dim = yb.shape[-1]
    pred_sum = np.zeros((n, out_dim), dtype=np.float64)
    pred_cnt = np.zeros((n, 1), dtype=np.float64)

    for k, st in enumerate(starts):
        ed = min(st + seq_len, n)
        valid_len = ed - st
        pred_sum[st:ed] += yb[k, :valid_len]
        pred_cnt[st:ed] += 1.0

    pred = pred_sum / np.maximum(pred_cnt, 1.0)
    return pred.astype(np.float32)


def save_compare_plot(time_s: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, ch_names, out_path: Path) -> None:
    n_ch = y_true.shape[1]
    fig_h = max(2 * n_ch, 8)
    fig, axes = plt.subplots(n_ch, 1, figsize=(14, fig_h), sharex=True)
    if n_ch == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(time_s, y_true[:, i], label="true", linewidth=1.0)
        ax.plot(time_s, y_pred[:, i], label="pred", linewidth=1.0, alpha=0.85)
        ax.set_ylabel(ch_names[i], fontsize=8)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Prediction vs Ground Truth")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用已训练模型进行推理并输出对比图和 RMS")
    parser.add_argument("--checkpoint", type=str, required=True, help="train.py 输出的 best.pt")
    parser.add_argument("--csv", type=str, default="", help="指定单个 cleaned csv 文件，默认使用测试集第一个")
    parser.add_argument("--out_dir", type=str, default="outputs/infer")
    parser.add_argument("--stride", type=int, default=50, help="推理滑窗步长")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt, model = load_checkpoint(ckpt_path, device=device)
    task = ckpt["task"]
    input_names = ckpt["input_names"]
    output_names = ckpt["output_names"]

    if args.csv:
        csv_path = Path(args.csv)
    else:
        test_files = ckpt["split_files"].get("test", [])
        if not test_files:
            raise ValueError("checkpoint 中没有 test 文件信息，请手动指定 --csv")
        candidate = Path(ckpt["train_args"]["data_dir"]) / test_files[0]
        csv_path = candidate

    x, y_true, time_s = load_infer_csv(csv_path, input_names=input_names, output_names=output_names)

    seq_len = int(ckpt["seq_len"])
    y_pred = predict_full_sequence(model, x, seq_len=seq_len, stride=args.stride, device=device)

    out_root = Path(args.out_dir) / task / csv_path.stem
    out_root.mkdir(parents=True, exist_ok=True)

    ch_names = output_names

    if y_true is not None:
        err = y_pred - y_true
        rmse_ch = np.sqrt(np.mean(err**2, axis=0))
        rmse_all = float(np.sqrt(np.mean(err**2)))

        save_compare_plot(time_s, y_true, y_pred, ch_names, out_root / "prediction_vs_true.png")

        rms_df = pd.DataFrame({"channel": ch_names, "rmse": rmse_ch})
        rms_df.loc[len(rms_df)] = ["ALL", rmse_all]
        rms_df.to_csv(out_root / "rms_report.csv", index=False, encoding="utf-8-sig")
    else:
        rmse_ch = None
        rmse_all = None

    pred_dict = {"time_s": time_s}
    for i, name in enumerate(ch_names):
        pred_dict[f"pred_{name}"] = y_pred[:, i]

    if y_true is not None:
        for i, name in enumerate(ch_names):
            pred_dict[f"true_{name}"] = y_true[:, i]

    pred_df = pd.DataFrame(pred_dict)
    pred_df.to_csv(out_root / "prediction_full.csv", index=False, encoding="utf-8-sig")

    summary = {
        "task": task,
        "csv": str(csv_path),
        "checkpoint": str(ckpt_path),
        "has_ground_truth": y_true is not None,
        "rmse_all": rmse_all,
        "rmse_by_channel": ({k: float(v) for k, v in zip(ch_names, rmse_ch)} if rmse_ch is not None else None),
        "plot": (str(out_root / "prediction_vs_true.png") if y_true is not None else None),
        "rms_csv": (str(out_root / "rms_report.csv") if y_true is not None else None),
        "prediction_csv": str(out_root / "prediction_full.csv"),
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"推理完成: {csv_path.name}")
    if rmse_all is not None:
        print(f"整体 RMSE: {rmse_all:.6f}")
        print(f"图像输出: {out_root / 'prediction_vs_true.png'}")
    else:
        print("输入文件未包含真实输出列，已跳过 RMS 与真值对比图，仅导出 prediction_full.csv")


if __name__ == "__main__":
    main()
