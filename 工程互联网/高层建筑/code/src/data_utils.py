from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, filtfilt

DEFAULT_FS = 50.0

DRIVER_COLS = [
    "WSMS00012.AccX",
    "WSMS00012.AccY",
    "WSMS00012.AccZ",
]

RESP_ACCEL_COLS = [
    "WSMS00007.AccX",
    "WSMS00007.AccY",
    "WSMS00007.AccZ",
    "WSMS00008.AccX",
    "WSMS00008.AccY",
    "WSMS00008.AccZ",
    "WSMS00009.AccX",
    "WSMS00009.AccY",
    "WSMS00009.AccZ",
    "WSMS00010.AccX",
    "WSMS00010.AccY",
    "WSMS00010.AccZ",
    "WSMS00011.AccX",
    "WSMS00011.AccY",
    "WSMS00011.AccZ",
]

STRAIN_COLS = [
    "WSGD00003.chdata",
    "WSGD00004.chdata",
]

RESPONSE_COLS = RESP_ACCEL_COLS + STRAIN_COLS
ALL_SIGNAL_COLS = DRIVER_COLS + RESPONSE_COLS


@dataclass
class CleanedSequence:
    name: str
    time_s: np.ndarray
    fs: float
    x_forward: np.ndarray
    y_forward: np.ndarray
    x_inverse: np.ndarray
    y_inverse: np.ndarray
    raw_driver_z: np.ndarray
    filtered_driver_z: np.ndarray


def _parse_time_value(v: object) -> pd.Timestamp:
    if pd.isna(v):
        return pd.NaT
    s = str(v).strip()
    if s.startswith('="') and s.endswith('"'):
        s = s[2:-1]
    elif s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    return pd.to_datetime(s, errors="coerce")


def estimate_fs_from_time(time_series: pd.Series, default_fs: float = DEFAULT_FS) -> float:
    dt = time_series.diff().dt.total_seconds().dropna()
    dt = dt[dt > 0]
    if dt.empty:
        return default_fs
    median_dt = float(dt.median())
    if median_dt <= 0:
        return default_fs
    fs = 1.0 / median_dt
    if not np.isfinite(fs) or fs <= 0:
        return default_fs
    return fs


def bandpass_filter(data: np.ndarray, fs: float, low_hz: float = 0.1, high_hz: float = 20.0, order: int = 4) -> np.ndarray:
    nyq = fs * 0.5
    low = max(low_hz / nyq, 1e-5)
    high = min(high_hz / nyq, 0.999)
    if not (0 < low < high < 1):
        return data.copy()

    b, a = butter(order, [low, high], btype="bandpass")
    min_len = max(3 * max(len(a), len(b)), 16)

    out = np.empty_like(data)
    for i in range(data.shape[1]):
        sig = data[:, i]
        if len(sig) <= min_len:
            out[:, i] = sig
        else:
            out[:, i] = filtfilt(b, a, sig)
    return out


def load_and_clean_csv(csv_path: Path) -> CleanedSequence:
    df = pd.read_csv(csv_path)

    missing = [c for c in ALL_SIGNAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} 缺少列: {missing}")
    if "time" not in df.columns:
        raise ValueError(f"{csv_path.name} 缺少 time 列")

    df = df[["time"] + ALL_SIGNAL_COLS].copy()
    df["time"] = df["time"].map(_parse_time_value)

    df = df.dropna(subset=["time"] + ALL_SIGNAL_COLS)
    if df.empty:
        raise ValueError(f"{csv_path.name} 清洗后无有效数据")

    df = df.sort_values("time").reset_index(drop=True)
    fs = estimate_fs_from_time(df["time"])

    signal = df[ALL_SIGNAL_COLS].to_numpy(dtype=np.float64)
    raw_driver_z = signal[:, DRIVER_COLS.index("WSMS00012.AccZ")].copy()

    detrended = np.empty_like(signal)
    for i in range(signal.shape[1]):
        detrended[:, i] = detrend(signal[:, i], type="linear")

    filtered = bandpass_filter(detrended, fs=fs, low_hz=0.1, high_hz=20.0, order=4)
    filtered_driver_z = filtered[:, DRIVER_COLS.index("WSMS00012.AccZ")].copy()

    t0 = df["time"].iloc[0]
    time_s = (df["time"] - t0).dt.total_seconds().to_numpy(dtype=np.float64)

    x_forward = filtered[:, : len(DRIVER_COLS)]
    y_forward = filtered[:, len(DRIVER_COLS) :]

    x_inverse = y_forward
    y_inverse = x_forward

    return CleanedSequence(
        name=csv_path.stem,
        time_s=time_s,
        fs=float(fs),
        x_forward=x_forward.astype(np.float32),
        y_forward=y_forward.astype(np.float32),
        x_inverse=x_inverse.astype(np.float32),
        y_inverse=y_inverse.astype(np.float32),
        raw_driver_z=raw_driver_z.astype(np.float32),
        filtered_driver_z=filtered_driver_z.astype(np.float32),
    )


def save_cleaned_sequence(seq: CleanedSequence, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_filtered = np.concatenate([seq.x_forward, seq.y_forward], axis=1)
    df = pd.DataFrame(all_filtered, columns=ALL_SIGNAL_COLS)
    df.insert(0, "fs_hz", np.full((len(df),), seq.fs, dtype=np.float32))
    df.insert(0, "time_s", seq.time_s.astype(np.float32))
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def save_meta(meta: Dict, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cleaned_csv_data(csv_path: Path) -> Dict[str, np.ndarray]:
    df = pd.read_csv(csv_path)

    # Case 1: cleaned csv produced by preprocess.py
    cleaned_required = ["time_s"] + ALL_SIGNAL_COLS
    if all(c in df.columns for c in cleaned_required):
        time_s = df["time_s"].to_numpy(dtype=np.float32)
        all_sig = df[ALL_SIGNAL_COLS].to_numpy(dtype=np.float32)

        x_forward = all_sig[:, : len(DRIVER_COLS)]
        y_forward = all_sig[:, len(DRIVER_COLS) :]

        x_inverse = y_forward
        y_inverse = x_forward

        if "fs_hz" in df.columns:
            fs = float(df["fs_hz"].median())
        else:
            fs = DEFAULT_FS

        return {
            "time_s": time_s,
            "fs": np.array([fs], dtype=np.float32),
            "x_forward": x_forward,
            "y_forward": y_forward,
            "x_inverse": x_inverse,
            "y_inverse": y_inverse,
        }

    # Case 2: raw csv from acquisition system, clean on the fly
    raw_required = ["time"] + ALL_SIGNAL_COLS
    if all(c in df.columns for c in raw_required):
        seq = load_and_clean_csv(csv_path)
        return {
            "time_s": seq.time_s.astype(np.float32),
            "fs": np.array([seq.fs], dtype=np.float32),
            "x_forward": seq.x_forward,
            "y_forward": seq.y_forward,
            "x_inverse": seq.x_inverse,
            "y_inverse": seq.y_inverse,
        }

    expected_cols = sorted(set(cleaned_required + raw_required))
    missing = [c for c in expected_cols if c not in df.columns]
    raise ValueError(f"{csv_path.name} 既不是 cleaned csv 也不是 raw csv，缺少列: {missing}")


def sliding_windows(x: np.ndarray, y: np.ndarray, seq_len: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    if x.shape[0] != y.shape[0]:
        raise ValueError("x 和 y 长度不一致")
    n = x.shape[0]
    if n < seq_len:
        return np.empty((0, seq_len, x.shape[1]), dtype=np.float32), np.empty((0, seq_len, y.shape[1]), dtype=np.float32)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for st in range(0, n - seq_len + 1, stride):
        ed = st + seq_len
        xs.append(x[st:ed])
        ys.append(y[st:ed])

    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)
