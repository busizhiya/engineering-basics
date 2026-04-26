from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

MASS_CODE_TO_VALUE = {
    0: 4.0,
    1: 5.159,
    2: 6.386,
    3: 7.597,
    4: 8.749,
    5: 9.867,
    6: 11.045,
    7: 11.58,
    8: 12.082,
    9: 12.592,
    10: 13.175,
}

SENSOR_CODE_TO_CM = {1: 38.0, 2: 68.0, 3: 98.0}
LOAD_CODE_TO_CM = {1: 38.0, 2: 68.0, 3: 98.0, 4: 128.0, 5: 158.0}
DEFAULT_SIGNAL_COLUMNS = ["dist_01.dist", "s0017-1.chdata", "s0017-2.chdata", "s0017-3.chdata", "s0015-2.chdata", "s0015-3.chdata", "s0013-1.chdata", "s0013-2.chdata", "s0013-3.chdata"]

C_FEATURE_NAMES = ["1", "sensor", "load", "sensor*load", "sensor^2", "load^2"]
K_FEATURE_NAMES = [
    "1",
    "sensor",
    "load",
    "sensor*load",
    "sensor^2",
    "load^2",
    "rbf_delta_15",
    "rbf_delta_30",
    "rbf_delta_50",
    "sensor_norm*rbf_delta_15",
    "sensor_norm*rbf_delta_30",
]


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train multi-signal continuous bidirectional mass-displacement models.")
    parser.add_argument("--data-csv", type=Path, default=base / "2026.3.30_test_wide.csv")
    parser.add_argument("--annotations-csv", type=Path, default=base / "annotations_sorted_renamed.csv")
    parser.add_argument("--signal-columns", type=str, default=",".join(DEFAULT_SIGNAL_COLUMNS))
    parser.add_argument("--out-dir", type=Path, default=base / "outputs" / "multi_signal_models")
    parser.add_argument("--time-origin", type=float, default=None, help="Optional absolute time origin (seconds) added to relative time columns")
    parser.add_argument("--time-origin-anchor-code", type=str, default="s0005-2", help="Sensor code used to find absolute time origin from rec/long CSV")
    return parser.parse_args()


def robust_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def parse_time_value(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    if text.startswith('="') and text.endswith('"'):
        text = text[2:-1]
    try:
        return float(text)
    except ValueError:
        return None


def parse_label(raw: Any) -> tuple[int | None, int | None, int | None]:
    text = str(raw).strip()
    if not text:
        return None, None, None
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit() or len(text) < 3:
        return None, None, None
    return int(text[0]), int(text[1]), int(text[2:])


def parse_ref_index(raw: Any) -> int | None:
    text = str(raw).strip().lower()
    if not text.startswith("ref"):
        return None
    suffix = text[3:]
    if not suffix.isdigit():
        return None
    return int(suffix)


def resolve_annotation_time_scale(data_time: pd.Series, ann_time: pd.Series) -> pd.Series:
    data_max = float(data_time.max())
    ann_max = float(ann_time.max())
    if data_max > 1e9 and ann_max < 1e6:
        return ann_time + float(data_time.min())
    return ann_time


def _try_read_min_time(path: Path) -> float | None:
    if not path.exists():
        return None
    for enc in ("utf-8-sig", "gbk"):
        try:
            tmp = pd.read_csv(path, usecols=["time"], nrows=5000, encoding=enc)
            t = tmp["time"].map(parse_time_value).dropna()
            if t.empty:
                return None
            return float(t.min())
        except Exception:
            continue
    return None


def normalize_sensor_code(raw_code: str) -> str:
    text = str(raw_code).strip().lower()
    m = re.match(r"^s0*(\d+)-(\d+)$", text)
    if not m:
        return text
    return f"s{int(m.group(1)):05d}-{int(m.group(2))}"


def _try_read_first_time_by_code(path: Path, target_code: str) -> float | None:
    if not path.exists():
        return None
    target = normalize_sensor_code(target_code)
    for enc in ("utf-8-sig", "gbk"):
        try:
            for chunk in pd.read_csv(path, usecols=["code", "time"], chunksize=200000, encoding=enc):
                codes = chunk["code"].astype(str).map(normalize_sensor_code)
                sel = chunk[codes == target]
                if sel.empty:
                    continue
                for raw_t in sel["time"].tolist():
                    t = parse_time_value(raw_t)
                    if t is not None:
                        return float(t)
        except Exception:
            continue
    return None


def infer_time_origin_seconds(data_csv_path: Path, data_time: pd.Series, explicit_origin: float | None, anchor_code: str) -> tuple[float, str]:
    if explicit_origin is not None:
        return float(explicit_origin), "explicit"

    tmin = float(data_time.min())
    tmax = float(data_time.max())
    if tmax > 1e9:
        return 0.0, "already_absolute"

    # Relative-time wide CSV usually starts near 0 and spans small values.
    if not (-10.0 <= tmin <= 10.0 and tmax < 1e7):
        return 0.0, "no_origin_needed"

    rec_files = sorted(data_csv_path.parent.glob("rec_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for rec in rec_files[:5]:
        anchor_first = _try_read_first_time_by_code(rec, anchor_code)
        if anchor_first is not None and anchor_first > 1e9:
            return anchor_first, f"rec:{rec.name}:{normalize_sensor_code(anchor_code)}"

    sidecar = data_csv_path.with_name(data_csv_path.stem.replace("_wide", "") + data_csv_path.suffix)
    if sidecar != data_csv_path:
        sidecar_first = _try_read_first_time_by_code(sidecar, anchor_code)
        if sidecar_first is not None and sidecar_first > 1e9:
            return sidecar_first, f"sidecar:{sidecar.name}:{normalize_sensor_code(anchor_code)}"

    return 0.0, f"not_found_anchor:{normalize_sensor_code(anchor_code)}"


def infer_sensor_code_from_signal_column(signal_column: str, reference_anchor_code: str) -> str:
    col = str(signal_column).strip().lower()
    if col.endswith(".chdata"):
        code = col.split(".", 1)[0]
        return normalize_sensor_code(code)
    if col.startswith("dist_") and col.endswith(".dist"):
        return normalize_sensor_code(reference_anchor_code)
    return normalize_sensor_code(reference_anchor_code)


def infer_signal_first_time_from_sources(
    data_csv_path: Path,
    signal_column: str,
    reference_anchor_code: str,
    fallback_time: float,
) -> tuple[float, str]:
    code = infer_sensor_code_from_signal_column(signal_column, reference_anchor_code)

    rec_files = sorted(data_csv_path.parent.glob("rec_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for rec in rec_files[:5]:
        t0 = _try_read_first_time_by_code(rec, code)
        if t0 is not None and t0 > 1e9:
            return float(t0), f"rec:{rec.name}:{code}"

    sidecar = data_csv_path.with_name(data_csv_path.stem.replace("_wide", "") + data_csv_path.suffix)
    if sidecar != data_csv_path:
        t0 = _try_read_first_time_by_code(sidecar, code)
        if t0 is not None and t0 > 1e9:
            return float(t0), f"sidecar:{sidecar.name}:{code}"

    return float(fallback_time), f"fallback_wide:{signal_column}"


def nearest_indices(sample_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sample_times, query_times, side="left")
    idx = np.clip(idx, 0, len(sample_times) - 1)
    left = np.clip(idx - 1, 0, len(sample_times) - 1)
    use_left = np.abs(query_times - sample_times[left]) < np.abs(query_times - sample_times[idx])
    return np.where(use_left, left, idx)


def fit_disp_vs_mass(mass: np.ndarray, disp: np.ndarray) -> tuple[float, float, float, int]:
    n = int(len(mass))
    if n < 2 or len(np.unique(mass)) < 2:
        return 0.0, float(np.mean(disp)), 0.0, n
    k, c = np.polyfit(mass, disp, deg=1)
    pred = k * mass + c
    ss_res = float(np.sum((disp - pred) ** 2))
    ss_tot = float(np.sum((disp - float(np.mean(disp))) ** 2))
    r2 = 0.0 if ss_tot <= 1e-12 else max(0.0, 1.0 - ss_res / ss_tot)
    return float(k), float(c), float(r2), n


def build_c_feature_matrix(sensor_cm: np.ndarray, load_cm: np.ndarray) -> np.ndarray:
    s = sensor_cm.astype(float)
    l = load_cm.astype(float)
    return np.column_stack([np.ones_like(s), s, l, s * l, s**2, l**2])


def build_k_feature_matrix(sensor_cm: np.ndarray, load_cm: np.ndarray) -> np.ndarray:
    s = sensor_cm.astype(float)
    l = load_cm.astype(float)
    delta = l - s
    sensor_norm = (s - 68.0) / 30.0
    g15 = np.exp(-0.5 * (delta / 15.0) ** 2)
    g30 = np.exp(-0.5 * (delta / 30.0) ** 2)
    g50 = np.exp(-0.5 * (delta / 50.0) ** 2)
    return np.column_stack([
        np.ones_like(s),
        s,
        l,
        s * l,
        s**2,
        l**2,
        g15,
        g30,
        g50,
        sensor_norm * g15,
        sensor_norm * g30,
    ])


def solve_surface(x: np.ndarray, y: np.ndarray, l2_reg: float) -> np.ndarray:
    xtx = x.T @ x
    xty = x.T @ y
    reg = l2_reg * np.eye(xtx.shape[0], dtype=float)
    return np.linalg.solve(xtx + reg, xty).astype(float)


def predict_k_surface(sensor_cm: np.ndarray, load_cm: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return build_k_feature_matrix(sensor_cm, load_cm) @ coef


def predict_c_surface(sensor_cm: np.ndarray, load_cm: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return build_c_feature_matrix(sensor_cm, load_cm) @ coef


def signal_slug(signal_column: str) -> str:
    return signal_column.replace(".", "_").replace("-", "_")


def resolve_signal_columns(data_df: pd.DataFrame, requested: str) -> list[str]:
    raw = [x.strip() for x in requested.split(",") if x.strip()]
    resolved: list[str] = []
    for col in raw:
        if col in data_df.columns:
            resolved.append(col)
            continue
        if col == "dist_01.dist":
            fallback = [c for c in data_df.columns if c.startswith("dist_") and c.endswith(".dist")]
            if fallback:
                resolved.append(sorted(fallback)[0])
                continue
    uniq = []
    seen = set()
    for c in resolved:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def resolve_reference_signal_column(data_df: pd.DataFrame) -> str:
    if "dist_01.dist" in data_df.columns:
        return "dist_01.dist"
    cols = [c for c in data_df.columns if c.startswith("dist_") and c.endswith(".dist")]
    if not cols:
        raise ValueError("No displacement reference signal found (dist_*.dist)")
    return sorted(cols)[0]


def estimate_first_frame_time(data_df: pd.DataFrame, signal_column: str) -> float:
    s = pd.to_numeric(data_df[signal_column], errors="coerce")
    valid = ~s.isna()
    if int(valid.sum()) == 0:
        raise ValueError(f"Signal has no valid numeric data: {signal_column}")

    t = data_df.loc[valid, "time"].to_numpy(dtype=float)
    v = s.loc[valid].to_numpy(dtype=float)
    v0 = float(v[0])
    scale = float(np.nanstd(v)) if len(v) > 1 else abs(v0)
    eps = max(1e-6, 0.01 * (scale if np.isfinite(scale) and scale > 0 else 1.0))

    diff = np.abs(v - v0)
    nz = np.abs(v)
    idx = np.where((diff > eps) | (nz > eps))[0]
    first_idx = int(idx[0]) if len(idx) else 0
    return float(t[first_idx])


def plot_training_scatter(ann_df: pd.DataFrame, models_df: pd.DataFrame, signal_column: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    load_colors = {1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e", 4: "#9467bd", 5: "#d62728"}
    sensor_markers = {1: "o", 2: "s", 3: "^"}

    for sensor_code in sorted(SENSOR_CODE_TO_CM.keys()):
        for load_code in sorted(LOAD_CODE_TO_CM.keys()):
            sub = ann_df[(ann_df["sensor_code"] == sensor_code) & (ann_df["load_code"] == load_code)]
            if sub.empty:
                continue
            ax.scatter(sub["mass_value"], sub["displacement_rel"], c=load_colors[load_code], marker=sensor_markers[sensor_code], s=42, alpha=0.75)

            row = models_df[(models_df["sensor_code"] == sensor_code) & (models_df["load_code"] == load_code)]
            if row.empty:
                continue
            k = float(row.iloc[0]["k_disp_vs_mass"])
            c = float(row.iloc[0]["c_disp_vs_mass"])
            x_line = np.linspace(float(sub["mass_value"].min()), float(sub["mass_value"].max()), 120)
            ax.plot(x_line, k * x_line + c, color=load_colors[load_code], linewidth=1.2, alpha=0.85)

    load_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=load_colors[l], markersize=8, label=f"load={l} ({LOAD_CODE_TO_CM[l]:g}cm)") for l in sorted(LOAD_CODE_TO_CM.keys())]
    sensor_handles = [Line2D([0], [0], marker=sensor_markers[s], color="#111827", linestyle="None", markersize=8, label=f"sensor={s} ({SENSOR_CODE_TO_CM[s]:g}cm)") for s in sorted(SENSOR_CODE_TO_CM.keys())]
    fit_handle = Line2D([0], [0], color="#111827", linewidth=1.3, label="linear fit")
    leg1 = ax.legend(handles=load_handles, loc="upper left", title="Color = load position")
    ax.add_artist(leg1)
    ax.legend(handles=sensor_handles + [fit_handle], loc="lower right", title="Marker = sensor position")

    ax.set_title(f"Training Scatter: Mass vs Relative Displacement ({signal_column})")
    ax.set_xlabel("mass")
    ax.set_ylabel("relative displacement")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_model_params(models_df: pd.DataFrame, signal_column: str, out_path: Path) -> None:
    x = np.arange(len(models_df), dtype=float)
    labels = models_df["model"].tolist()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(x, models_df["k_disp_vs_mass"], marker="o", linewidth=1.8, color="#2563eb")
    axes[0].set_ylabel("k")
    axes[0].set_title(f"15 linear model parameters ({signal_column})")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, models_df["c_disp_vs_mass"], marker="s", linewidth=1.8, color="#16a34a")
    axes[1].set_ylabel("c")
    axes[1].grid(alpha=0.25)

    axes[2].bar(x, models_df["r2"], color="#f59e0b", alpha=0.85)
    axes[2].set_ylabel("R2")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_xlabel("model")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_surface_fit(models_df: pd.DataFrame, k_coef: np.ndarray, c_coef: np.ndarray, signal_column: str, out_path: Path) -> None:
    s = models_df["sensor_cm"].to_numpy(dtype=float)
    l = models_df["load_cm"].to_numpy(dtype=float)
    k_true = models_df["k_disp_vs_mass"].to_numpy(dtype=float)
    c_true = models_df["c_disp_vs_mass"].to_numpy(dtype=float)
    k_hat = predict_k_surface(s, l, k_coef)
    c_hat = predict_c_surface(s, l, c_coef)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(k_true, k_hat, c="#2563eb", s=55, label="15 model points")
    mn, mx = float(min(k_true.min(), k_hat.min())), float(max(k_true.max(), k_hat.max()))
    axes[0].plot([mn, mx], [mn, mx], "k--", linewidth=1.0, label="y=x")
    axes[0].set_title(f"k surface fit ({signal_column})")
    axes[0].set_xlabel("true k")
    axes[0].set_ylabel("predicted k")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].scatter(c_true, c_hat, c="#16a34a", s=55, label="15 model points")
    mn, mx = float(min(c_true.min(), c_hat.min())), float(max(c_true.max(), c_hat.max()))
    axes[1].plot([mn, mx], [mn, mx], "k--", linewidth=1.0, label="y=x")
    axes[1].set_title(f"c surface fit ({signal_column})")
    axes[1].set_xlabel("true c")
    axes[1].set_ylabel("predicted c")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_surface_3d(models_df: pd.DataFrame, k_coef: np.ndarray, c_coef: np.ndarray, signal_column: str, out_path: Path) -> None:
    sensor_vals = np.linspace(float(models_df["sensor_cm"].min()), float(models_df["sensor_cm"].max()), 50)
    load_vals = np.linspace(float(models_df["load_cm"].min()), float(models_df["load_cm"].max()), 60)
    s_grid, l_grid = np.meshgrid(sensor_vals, load_vals)
    k_grid = predict_k_surface(s_grid.ravel(), l_grid.ravel(), k_coef).reshape(s_grid.shape)
    c_grid = predict_c_surface(s_grid.ravel(), l_grid.ravel(), c_coef).reshape(s_grid.shape)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf1 = ax1.plot_surface(s_grid, l_grid, k_grid, cmap="viridis", alpha=0.78, linewidth=0)
    ax1.scatter(models_df["sensor_cm"], models_df["load_cm"], models_df["k_disp_vs_mass"], c="#111827", s=35, depthshade=False, label="15 fitted models")
    ax1.set_title(f"k(sensor,load) ({signal_column})")
    ax1.set_xlabel("sensor cm")
    ax1.set_ylabel("load cm")
    ax1.set_zlabel("k")
    ax1.legend(loc="best")
    fig.colorbar(surf1, ax=ax1, shrink=0.65, pad=0.08)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf2 = ax2.plot_surface(s_grid, l_grid, c_grid, cmap="plasma", alpha=0.78, linewidth=0)
    ax2.scatter(models_df["sensor_cm"], models_df["load_cm"], models_df["c_disp_vs_mass"], c="#111827", s=35, depthshade=False, label="15 fitted models")
    ax2.set_title(f"c(sensor,load) ({signal_column})")
    ax2.set_xlabel("sensor cm")
    ax2.set_ylabel("load cm")
    ax2.set_zlabel("c")
    ax2.legend(loc="best")
    fig.colorbar(surf2, ax=ax2, shrink=0.65, pad=0.08)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_x0_height_aligned_overlay(data_df: pd.DataFrame, ann_df: pd.DataFrame, signal_column: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    cmap = plt.get_cmap("viridis")
    sensor_positions = sorted(ann_df["sensor_cm"].unique())
    norm = plt.Normalize(vmin=float(min(sensor_positions)), vmax=float(max(sensor_positions)))

    t_col = "signal_time" if "signal_time" in data_df.columns else "time"
    t = data_df[t_col].to_numpy(dtype=float)
    d = data_df["displacement"].to_numpy(dtype=float)
    baseline_map = ann_df.groupby("ref_code", as_index=True)["baseline"].mean().to_dict()

    seg_count = 0
    ann_sorted = ann_df.sort_values("time").reset_index(drop=True)
    for sensor_code in sorted(ann_sorted["sensor_code"].unique()):
        for load_code in sorted(ann_sorted["load_code"].unique()):
            g = ann_sorted[(ann_sorted["sensor_code"] == sensor_code) & (ann_sorted["load_code"] == load_code)].sort_values("time")
            if g.empty:
                continue
            t0_rows = g[g["mass_code"] == 0]
            t10_rows = g[g["mass_code"] == 10]
            if t0_rows.empty or t10_rows.empty:
                continue
            for t0 in t0_rows["time"].to_numpy(dtype=float):
                t10_candidates = t10_rows[t10_rows["time"] >= t0]["time"].to_numpy(dtype=float)
                if len(t10_candidates) == 0:
                    continue
                t10 = float(t10_candidates[0])
                if t10 <= t0:
                    continue
                mask = (t >= t0) & (t <= t10)
                if int(mask.sum()) < 3:
                    continue
                t_seg = t[mask]
                d_seg = d[mask]
                baseline = float(baseline_map.get(int(sensor_code), 0.0))
                rel_seg = d_seg - baseline
                x_norm = 10.0 * (t_seg - t0) / (t10 - t0)
                y_aligned = rel_seg - float(rel_seg[0])
                sensor_cm = float(SENSOR_CODE_TO_CM[int(sensor_code)])
                ax.plot(x_norm, y_aligned, color=cmap(norm(sensor_cm)), alpha=0.65, linewidth=1.4)
                seg_count += 1

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, pad=0.01)
    cbar.set_label("sensor position (cm)")

    ax.set_xlim(0.0, 10.0)
    ax.set_xlabel("normalized stage (x0 -> x10)")
    ax.set_ylabel("relative displacement (x0-height aligned)")
    ax.set_title(f"x0-height aligned x0-x10 overlay ({signal_column}, count={seg_count})")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_time_alignment_debug(
    data_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    signal_column: str,
    ref_start_time: float,
    signal_start_time: float,
    shift_sec: float,
    out_path: Path,
) -> None:
    raw_t = data_df["time"].to_numpy(dtype=float)
    raw_v = data_df["displacement"].to_numpy(dtype=float)
    aligned_t = data_df["signal_time"].to_numpy(dtype=float)

    ann_times = ann_df["time"].to_numpy(dtype=float)
    ann_idx = nearest_indices(aligned_t, ann_times)
    ann_y = raw_v[ann_idx]
    ann_labels = ann_df["mass_code"].astype(int).astype(str).to_numpy()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(raw_t, raw_v, color="#9ca3af", linewidth=1.0, alpha=0.85, label="raw signal (original time)")
    ax.plot(aligned_t, raw_v, color="#2563eb", linewidth=1.15, alpha=0.95, label="aligned signal (shifted time)")
    ax.scatter(ann_times, ann_y, c="#dc2626", s=12, alpha=0.75, label="annotation positions")

    # Keep labels sparse so the figure remains readable.
    max_labels = min(24, len(ann_times))
    if max_labels > 0:
        label_ids = np.linspace(0, len(ann_times) - 1, num=max_labels, dtype=int)
        used = set()
        for i in label_ids:
            key = int(i)
            if key in used:
                continue
            used.add(key)
            ax.text(ann_times[key], ann_y[key], ann_labels[key], fontsize=7, color="#7f1d1d", alpha=0.85)

    ax.axvline(ref_start_time, color="#16a34a", linestyle="--", linewidth=1.2, label="ref first frame")
    ax.axvline(signal_start_time, color="#f59e0b", linestyle="--", linewidth=1.2, label="signal first frame")

    text = (
        f"signal={signal_column}\n"
        f"ref_start={ref_start_time:.6f}s\n"
        f"signal_start={signal_start_time:.6f}s\n"
        f"shift_applied={shift_sec:.6f}s"
    )
    ax.text(
        0.01,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#d1d5db"},
    )

    ax.set_title(f"Time alignment debug ({signal_column})")
    ax.set_xlabel("time")
    ax.set_ylabel("displacement")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def train_one_signal(
    signal_column: str,
    data_parsed: pd.DataFrame,
    ann_template: pd.DataFrame,
    ref_template: pd.DataFrame,
    ref_start_time: float,
    signal_start_time: float,
    signal_start_time_source: str,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    data_df = data_parsed.copy()
    data_df["displacement"] = pd.to_numeric(data_df[signal_column], errors="coerce")
    data_df = data_df.dropna(subset=["displacement"]).copy().reset_index(drop=True)

    time_shift_sec = -(signal_start_time - float(ref_start_time))
    data_df["signal_time"] = data_df["time"] - time_shift_sec

    ann_df = ann_template.copy()
    ref_df = ref_template.copy()

    t_data = data_df["signal_time"].to_numpy(dtype=float)
    ref_idx = nearest_indices(t_data, ref_df["time"].to_numpy(dtype=float))
    ref_df["displacement"] = data_df["displacement"].to_numpy(dtype=float)[ref_idx]
    baseline_map = ref_df.groupby("ref_code", as_index=True)["displacement"].mean().to_dict()

    idx = nearest_indices(t_data, ann_df["time"].to_numpy(dtype=float))
    ann_df["displacement"] = data_df["displacement"].to_numpy(dtype=float)[idx]
    ann_df["ref_code"] = ann_df["sensor_code"]
    ann_df["baseline"] = ann_df["ref_code"].map(baseline_map)
    ann_df = ann_df.dropna(subset=["baseline"]).copy()
    ann_df["displacement_rel"] = ann_df["displacement"] - ann_df["baseline"]

    rows: list[dict[str, Any]] = []
    for sensor_code in sorted(SENSOR_CODE_TO_CM.keys()):
        for load_code in sorted(LOAD_CODE_TO_CM.keys()):
            sub = ann_df[(ann_df["sensor_code"] == sensor_code) & (ann_df["load_code"] == load_code)]
            if sub.empty:
                continue
            k, c, r2, n = fit_disp_vs_mass(sub["mass_value"].to_numpy(dtype=float), sub["displacement_rel"].to_numpy(dtype=float))
            rows.append(
                {
                    "model": f"s{sensor_code}_l{load_code}",
                    "sensor_code": int(sensor_code),
                    "load_code": int(load_code),
                    "sensor_cm": float(SENSOR_CODE_TO_CM[sensor_code]),
                    "load_cm": float(LOAD_CODE_TO_CM[load_code]),
                    "k_disp_vs_mass": k,
                    "c_disp_vs_mass": c,
                    "r2": r2,
                    "n": n,
                }
            )

    if not rows:
        raise ValueError(f"No valid model groups found for signal {signal_column}")

    models_df = pd.DataFrame(rows).sort_values(["sensor_code", "load_code"]).reset_index(drop=True)
    s_fit = models_df["sensor_cm"].to_numpy(dtype=float)
    l_fit = models_df["load_cm"].to_numpy(dtype=float)

    k_coef = solve_surface(build_k_feature_matrix(s_fit, l_fit), models_df["k_disp_vs_mass"].to_numpy(dtype=float), l2_reg=1e-3)
    c_coef = solve_surface(build_c_feature_matrix(s_fit, l_fit), models_df["c_disp_vs_mass"].to_numpy(dtype=float), l2_reg=1e-6)

    models_df["k_surface_pred"] = predict_k_surface(s_fit, l_fit, k_coef)
    models_df["c_surface_pred"] = predict_c_surface(s_fit, l_fit, c_coef)

    slug = signal_slug(signal_column)
    params_csv = out_dir / "models_15_params.csv"
    scatter_plot = out_dir / "train_mass_displacement_scatter.png"
    params_plot = out_dir / "train_15_linear_model_params.png"
    surface_fit_plot = out_dir / "train_continuous_surface_fit.png"
    surface_3d_plot = out_dir / "train_continuous_surface_3d.png"
    aligned_overlay_plot = out_dir / "train_aligned_x0_height_overlay.png"
    alignment_debug_plot = out_dir / "train_time_alignment_debug.png"
    model_json = out_dir / "continuous_mass_disp_model.json"

    models_df.to_csv(params_csv, index=False, encoding="utf-8-sig")
    plot_training_scatter(ann_df, models_df, signal_column, scatter_plot)
    plot_model_params(models_df, signal_column, params_plot)
    plot_surface_fit(models_df, k_coef, c_coef, signal_column, surface_fit_plot)
    plot_surface_3d(models_df, k_coef, c_coef, signal_column, surface_3d_plot)
    plot_x0_height_aligned_overlay(data_df, ann_df, signal_column, aligned_overlay_plot)
    plot_time_alignment_debug(data_df, ann_df, signal_column, ref_start_time, signal_start_time, time_shift_sec, alignment_debug_plot)

    model_obj = {
        "signal_column": signal_column,
        "signal_slug": slug,
        "reference_signal_column": "dist_01.dist",
        "reference_first_frame_time": float(ref_start_time),
        "signal_first_frame_time_source": signal_start_time_source,
        "signal_first_frame_time": float(signal_start_time),
        "time_shift_applied_sec": float(time_shift_sec),
        "displacement_definition": "relative_to_ref_by_sensor",
        "sensor_code_to_cm": SENSOR_CODE_TO_CM,
        "load_code_to_cm": LOAD_CODE_TO_CM,
        "k_feature_type": "poly2_diag_rbf",
        "c_feature_type": "poly2",
        "k_feature_names": K_FEATURE_NAMES,
        "c_feature_names": C_FEATURE_NAMES,
        "k_coef": k_coef.tolist(),
        "c_coef": c_coef.tolist(),
        "reference_baseline_by_ref": {str(k): float(v) for k, v in baseline_map.items()},
        "parameter_count": 17,
    }
    with model_json.open("w", encoding="utf-8") as f:
        json.dump(model_obj, f, ensure_ascii=False, indent=2)

    k_rmse = float(np.sqrt(np.mean((models_df["k_disp_vs_mass"] - models_df["k_surface_pred"]) ** 2)))
    c_rmse = float(np.sqrt(np.mean((models_df["c_disp_vs_mass"] - models_df["c_surface_pred"]) ** 2)))

    summary = {
        "signal_column": signal_column,
        "signal_slug": slug,
        "reference_first_frame_time": float(ref_start_time),
        "signal_first_frame_time_source": signal_start_time_source,
        "signal_first_frame_time": float(signal_start_time),
        "time_shift_applied_sec": float(time_shift_sec),
        "rows_annotations_used": int(len(ann_df)),
        "linear_model_count": int(len(models_df)),
        "continuous_parameter_count": 17,
        "k_surface_rmse": k_rmse,
        "c_surface_rmse": c_rmse,
        "artifacts": {
            "params_csv": str(params_csv),
            "scatter_plot": str(scatter_plot),
            "params_plot": str(params_plot),
            "surface_fit_plot": str(surface_fit_plot),
            "surface_3d_plot": str(surface_3d_plot),
            "aligned_x0_height_overlay": str(aligned_overlay_plot),
            "alignment_debug_plot": str(alignment_debug_plot),
            "continuous_model_json": str(model_json),
        },
    }
    with (out_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_df = robust_read_csv(args.data_csv)
    ann_df = robust_read_csv(args.annotations_csv)

    data_df["time"] = data_df["time"].map(parse_time_value)
    data_df = data_df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    time_origin_sec, time_origin_source = infer_time_origin_seconds(args.data_csv, data_df["time"], args.time_origin, args.time_origin_anchor_code)
    if abs(time_origin_sec) > 1e-12:
        data_df["time"] = data_df["time"] + float(time_origin_sec)

    ann_df["time"] = ann_df["time"].map(parse_time_value)
    ann_df = ann_df.dropna(subset=["time"]).copy()
    ann_df["time"] = resolve_annotation_time_scale(data_df["time"], ann_df["time"])

    ann_raw = ann_df.copy()

    parsed = ann_df["label"].apply(parse_label)
    ann_df["sensor_code"] = parsed.map(lambda x: x[0])
    ann_df["load_code"] = parsed.map(lambda x: x[1])
    ann_df["mass_code"] = parsed.map(lambda x: x[2])
    ann_df = ann_df.dropna(subset=["sensor_code", "load_code", "mass_code"]).copy()
    ann_df["sensor_code"] = ann_df["sensor_code"].astype(int)
    ann_df["load_code"] = ann_df["load_code"].astype(int)
    ann_df["mass_code"] = ann_df["mass_code"].astype(int)
    ann_df = ann_df[ann_df["sensor_code"].isin(SENSOR_CODE_TO_CM.keys())].copy()
    ann_df = ann_df[ann_df["load_code"].isin(LOAD_CODE_TO_CM.keys())].copy()
    ann_df = ann_df[ann_df["mass_code"].isin(MASS_CODE_TO_VALUE.keys())].copy()
    ann_df["sensor_cm"] = ann_df["sensor_code"].map(SENSOR_CODE_TO_CM)
    ann_df["load_cm"] = ann_df["load_code"].map(LOAD_CODE_TO_CM)
    ann_df["mass_value"] = ann_df["mass_code"].map(MASS_CODE_TO_VALUE)
    ann_df = ann_df.sort_values("time").reset_index(drop=True)

    ref_df = ann_raw.copy()
    ref_df["ref_code"] = ref_df["label"].map(parse_ref_index)
    ref_df = ref_df.dropna(subset=["ref_code"]).copy()
    ref_df["ref_code"] = ref_df["ref_code"].astype(int)
    ref_df = ref_df[ref_df["ref_code"].isin(SENSOR_CODE_TO_CM.keys())].copy()
    if ref_df.empty:
        raise ValueError("No ref labels found (expected ref1/ref2/ref3).")

    signal_columns = resolve_signal_columns(data_df, args.signal_columns)
    if not signal_columns:
        raise ValueError("No valid signal columns resolved.")

    reference_signal = resolve_reference_signal_column(data_df)
    ref_fallback = estimate_first_frame_time(data_df, reference_signal)
    ref_start_time, ref_start_source = infer_signal_first_time_from_sources(
        args.data_csv,
        reference_signal,
        args.time_origin_anchor_code,
        ref_fallback,
    )

    summaries: list[dict[str, Any]] = []
    for signal_column in signal_columns:
        sig_dir = args.out_dir / signal_slug(signal_column)
        sig_fallback = estimate_first_frame_time(data_df, signal_column)
        sig_start_time, sig_start_source = infer_signal_first_time_from_sources(
            args.data_csv,
            signal_column,
            args.time_origin_anchor_code,
            sig_fallback,
        )
        summaries.append(train_one_signal(signal_column, data_df, ann_df, ref_df, ref_start_time, sig_start_time, sig_start_source, sig_dir))

    merged = pd.DataFrame(
        [
            {
                "signal_column": s["signal_column"],
                "signal_slug": s["signal_slug"],
                "k_surface_rmse": s["k_surface_rmse"],
                "c_surface_rmse": s["c_surface_rmse"],
                "model_json": s["artifacts"]["continuous_model_json"],
            }
            for s in summaries
        ]
    )
    merged_csv = args.out_dir / "multi_signal_train_metrics.csv"
    merged.to_csv(merged_csv, index=False, encoding="utf-8-sig")

    summary = {
        "data_csv": str(args.data_csv),
        "annotations_csv": str(args.annotations_csv),
        "time_origin_applied_sec": float(time_origin_sec),
        "time_origin_source": time_origin_source,
        "time_origin_anchor_code": normalize_sensor_code(args.time_origin_anchor_code),
        "reference_signal_column": reference_signal,
        "reference_first_frame_time_source": ref_start_source,
        "reference_first_frame_time": float(ref_start_time),
        "rows_data": int(len(data_df)),
        "rows_annotations_used": int(len(ann_df)),
        "signal_columns": signal_columns,
        "signal_count": int(len(signal_columns)),
        "multi_signal_metrics_csv": str(merged_csv),
        "per_signal_summaries": summaries,
    }

    with (args.out_dir / "train_summary_multi_signal.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
