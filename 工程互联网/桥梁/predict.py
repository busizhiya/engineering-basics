from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SIGNAL_COLUMNS = ["dist_01.dist", "s0017-1.chdata", "s0017-2.chdata", "s0017-3.chdata"]


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Multi-signal bidirectional prediction for continuous sensor/load positions.")
    parser.add_argument("--model-root", type=Path, default=base / "outputs" / "multi_signal_models")
    parser.add_argument("--signal-columns", type=str, default=",".join(DEFAULT_SIGNAL_COLUMNS))
    parser.add_argument("--sensor-cm", type=float, required=True)
    parser.add_argument("--load-cm", type=float, required=True)
    parser.add_argument("--mass", type=float, default=None, help="If set, directly predict relative displacement")
    parser.add_argument("--disp-rel", type=float, default=None, help="If set, directly predict mass from relative displacement")

    parser.add_argument("--data-csv", type=Path, default=base / "2026.3.30_test_wide.csv")
    parser.add_argument("--annotations-csv", type=Path, default=base / "annotations_sorted_renamed.csv")
    parser.add_argument("--sx-label", type=str, default="sx")
    parser.add_argument("--tx-label", type=str, default="tx")
    parser.add_argument("--out-dir", type=Path, default=base / "outputs" / "predict_multi_signal")
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


def signal_slug(signal_column: str) -> str:
    return signal_column.replace(".", "_").replace("-", "_")


def parse_signal_columns(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


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


def feature_row_poly2(sensor_cm: float, load_cm: float) -> np.ndarray:
    s = float(sensor_cm)
    l = float(load_cm)
    return np.array([1.0, s, l, s * l, s**2, l**2], dtype=float)


def feature_row_poly2_diag_rbf(sensor_cm: float, load_cm: float) -> np.ndarray:
    s = float(sensor_cm)
    l = float(load_cm)
    delta = l - s
    sensor_norm = (s - 68.0) / 30.0
    g15 = np.exp(-0.5 * (delta / 15.0) ** 2)
    g30 = np.exp(-0.5 * (delta / 30.0) ** 2)
    g50 = np.exp(-0.5 * (delta / 50.0) ** 2)
    return np.array([1.0, s, l, s * l, s**2, l**2, g15, g30, g50, sensor_norm * g15, sensor_norm * g30], dtype=float)


def feature_row(sensor_cm: float, load_cm: float, feature_type: str) -> np.ndarray:
    if feature_type == "poly2_diag_rbf":
        return feature_row_poly2_diag_rbf(sensor_cm, load_cm)
    return feature_row_poly2(sensor_cm, load_cm)


def parse_model(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        model = json.load(f)
    if "k_coef" not in model or "c_coef" not in model:
        raise ValueError(f"Model JSON missing k_coef/c_coef: {path}")
    return model


def predict_disp_from_mass(mass: float, sensor_cm: float, load_cm: float, model: dict[str, Any]) -> tuple[float, float, float]:
    k_coef = np.array(model["k_coef"], dtype=float)
    c_coef = np.array(model["c_coef"], dtype=float)
    k_feature_type = str(model.get("k_feature_type", "poly2"))
    c_feature_type = str(model.get("c_feature_type", "poly2"))
    k = float(feature_row(sensor_cm, load_cm, k_feature_type) @ k_coef)
    c = float(feature_row(sensor_cm, load_cm, c_feature_type) @ c_coef)
    return float(k * mass + c), k, c


def predict_mass_from_disp(disp_rel: float, sensor_cm: float, load_cm: float, model: dict[str, Any]) -> tuple[float, float, float]:
    k_coef = np.array(model["k_coef"], dtype=float)
    c_coef = np.array(model["c_coef"], dtype=float)
    k_feature_type = str(model.get("k_feature_type", "poly2"))
    c_feature_type = str(model.get("c_feature_type", "poly2"))
    k = float(feature_row(sensor_cm, load_cm, k_feature_type) @ k_coef)
    c = float(feature_row(sensor_cm, load_cm, c_feature_type) @ c_coef)
    if abs(k) < 1e-12:
        raise ValueError("Predicted slope k is near zero")
    return float((disp_rel - c) / k), k, c


def nearest_value_at_time(times: np.ndarray, values: np.ndarray, t_query: float) -> float:
    idx = int(np.searchsorted(times, t_query, side="left"))
    idx = max(0, min(idx, len(times) - 1))
    left = max(0, idx - 1)
    if abs(t_query - times[left]) < abs(t_query - times[idx]):
        idx = left
    return float(values[idx])


def find_mark_time(ann_df: pd.DataFrame, target_label: str, fallback_prefix: str) -> float:
    labels = ann_df["label"].astype(str).str.strip().str.lower()
    exact = ann_df[labels == target_label.strip().lower()]
    if not exact.empty:
        return float(exact.iloc[0]["time"])
    regex = rf"^{re.escape(fallback_prefix)}\d*$"
    cand = ann_df[labels.str.match(regex)].sort_values("time")
    if not cand.empty:
        return float(cand.iloc[0]["time"])
    raise ValueError(f"Could not find label '{target_label}' or fallback '{fallback_prefix}*'")


def plot_file_mode(t: np.ndarray, rel: np.ndarray, sx_time: float, tx_time: float, rel_tx: float, signal_column: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(t, rel, color="#1f77b4", linewidth=1.2, label="relative displacement (sx ref)")
    ax.axvline(sx_time, color="#16a34a", linestyle="--", linewidth=1.2, label="sx")
    ax.axvline(tx_time, color="#dc2626", linestyle="--", linewidth=1.2, label="tx")
    ax.scatter([sx_time, tx_time], [0.0, rel_tx], c=["#16a34a", "#dc2626"], s=55, zorder=3)
    ax.set_title(f"Relative displacement with sx as reference ({signal_column})")
    ax.set_xlabel("time")
    ax.set_ylabel("relative displacement")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_time_alignment_debug(
    raw_t: np.ndarray,
    aligned_t: np.ndarray,
    values: np.ndarray,
    ann_times: np.ndarray,
    ann_labels: np.ndarray,
    ref_start_time: float,
    signal_start_time: float,
    shift_sec: float,
    signal_column: str,
    out_path: Path,
) -> None:
    ann_idx = np.searchsorted(aligned_t, ann_times, side="left")
    ann_idx = np.clip(ann_idx, 0, len(aligned_t) - 1)
    left = np.clip(ann_idx - 1, 0, len(aligned_t) - 1)
    use_left = np.abs(ann_times - aligned_t[left]) < np.abs(ann_times - aligned_t[ann_idx])
    ann_idx = np.where(use_left, left, ann_idx)
    ann_y = values[ann_idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(raw_t, values, color="#9ca3af", linewidth=1.0, alpha=0.85, label="raw signal (original time)")
    ax.plot(aligned_t, values, color="#2563eb", linewidth=1.15, alpha=0.95, label="aligned signal (shifted time)")
    ax.scatter(ann_times, ann_y, c="#dc2626", s=12, alpha=0.75, label="annotation positions")

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


def plot_mass_mode(sensor_cm: float, load_cm: float, input_mass: float, k: float, c: float, signal_column: str, out_path: Path) -> None:
    m_grid = np.linspace(max(0.0, input_mass - 4.0), input_mass + 4.0, 120)
    d_grid = k * m_grid + c
    d_point = k * input_mass + c

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(m_grid, d_grid, color="#2563eb", linewidth=1.5)
    ax.scatter([input_mass], [d_point], color="#dc2626", s=65, zorder=3)
    ax.set_title(f"Relative displacement curve ({signal_column}) @ sensor={sensor_cm:g}cm, load={load_cm:g}cm")
    ax.set_xlabel("mass")
    ax.set_ylabel("relative displacement")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_disp_mode(sensor_cm: float, load_cm: float, input_disp: float, k: float, c: float, signal_column: str, out_path: Path) -> None:
    span = max(1.0, 0.5 * abs(float(input_disp)) + 1.0)
    d_grid = np.linspace(float(input_disp) - span, float(input_disp) + span, 120)
    m_grid = (d_grid - c) / k
    m_point = (float(input_disp) - c) / k

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(d_grid, m_grid, color="#2563eb", linewidth=1.5)
    ax.scatter([input_disp], [m_point], color="#dc2626", s=65, zorder=3)
    ax.set_title(f"Mass curve from relative displacement ({signal_column}) @ sensor={sensor_cm:g}cm, load={load_cm:g}cm")
    ax.set_xlabel("relative displacement")
    ax.set_ylabel("mass")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_multi_signal_bars(df: pd.DataFrame, metric_col: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["signal_column"], df[metric_col], color="#2563eb", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("signal")
    ax.set_ylabel(metric_col)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mass is not None and args.disp_rel is not None:
        raise ValueError("Use only one of --mass or --disp-rel")

    requested_signals = parse_signal_columns(args.signal_columns)
    results: list[dict[str, Any]] = []

    if args.mass is None and args.disp_rel is None:
        data_df = robust_read_csv(args.data_csv)
        ann_df = robust_read_csv(args.annotations_csv)
        data_df["time"] = data_df["time"].map(parse_time_value)
        data_df = data_df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

        time_origin_sec, time_origin_source = infer_time_origin_seconds(args.data_csv, data_df["time"], args.time_origin, args.time_origin_anchor_code)
        if abs(time_origin_sec) > 1e-12:
            data_df["time"] = data_df["time"] + float(time_origin_sec)

        ann_df["time"] = ann_df["time"].map(parse_time_value)
        ann_df = ann_df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        ann_df["time"] = resolve_annotation_time_scale(data_df["time"], ann_df["time"])
        sx_time = find_mark_time(ann_df, args.sx_label, "s")
        tx_time = find_mark_time(ann_df, args.tx_label, "t")
        if tx_time < sx_time:
            sx_time, tx_time = tx_time, sx_time

        reference_signal = resolve_reference_signal_column(data_df)
        ref_fallback = estimate_first_frame_time(data_df, reference_signal)
        ref_start_time, ref_start_source = infer_signal_first_time_from_sources(
            args.data_csv,
            reference_signal,
            args.time_origin_anchor_code,
            ref_fallback,
        )

    for signal_column in requested_signals:
        slug = signal_slug(signal_column)
        model_json = args.model_root / slug / "continuous_mass_disp_model.json"
        if not model_json.exists():
            continue
        model = parse_model(model_json)

        signal_out = args.out_dir / slug
        signal_out.mkdir(parents=True, exist_ok=True)

        if args.mass is not None:
            pred_rel, k, c = predict_disp_from_mass(args.mass, args.sensor_cm, args.load_cm, model)
            out = {
                "mode": "mass_to_displacement",
                "signal_column": signal_column,
                "sensor_cm": float(args.sensor_cm),
                "load_cm": float(args.load_cm),
                "mass": float(args.mass),
                "pred_relative_displacement": float(pred_rel),
                "k_disp_vs_mass": float(k),
                "c_disp_vs_mass": float(c),
            }
            pd.DataFrame([out]).to_csv(signal_out / "predict_mass_to_disp.csv", index=False, encoding="utf-8-sig")
            plot_mass_mode(args.sensor_cm, args.load_cm, float(args.mass), k, c, signal_column, signal_out / "predict_mass_to_disp_curve.png")
            with (signal_out / "predict_summary.json").open("w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            results.append(out)
            continue

        if args.disp_rel is not None:
            pred_mass, k, c = predict_mass_from_disp(args.disp_rel, args.sensor_cm, args.load_cm, model)
            out = {
                "mode": "displacement_to_mass",
                "signal_column": signal_column,
                "sensor_cm": float(args.sensor_cm),
                "load_cm": float(args.load_cm),
                "input_relative_displacement": float(args.disp_rel),
                "pred_mass": float(pred_mass),
                "k_disp_vs_mass": float(k),
                "c_disp_vs_mass": float(c),
            }
            pd.DataFrame([out]).to_csv(signal_out / "predict_disp_to_mass.csv", index=False, encoding="utf-8-sig")
            plot_disp_mode(args.sensor_cm, args.load_cm, float(args.disp_rel), k, c, signal_column, signal_out / "predict_disp_to_mass_curve.png")
            with (signal_out / "predict_summary.json").open("w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            results.append(out)
            continue

        model_signal = str(model.get("signal_column", signal_column))
        if model_signal not in data_df.columns:
            continue

        d = pd.to_numeric(data_df[model_signal], errors="coerce")
        valid = ~d.isna()
        if int(valid.sum()) < 3:
            continue

        sig_fallback = estimate_first_frame_time(data_df, model_signal)
        sig_start_time, sig_start_source = infer_signal_first_time_from_sources(
            args.data_csv,
            model_signal,
            args.time_origin_anchor_code,
            sig_fallback,
        )
        shift_sec = -(sig_start_time - ref_start_time)
        t_aligned = data_df["time"] - shift_sec

        t = t_aligned.loc[valid].to_numpy(dtype=float)
        raw_t = data_df.loc[valid, "time"].to_numpy(dtype=float)
        dv = d.loc[valid].to_numpy(dtype=float)
        sx_disp = nearest_value_at_time(t, dv, sx_time)
        tx_disp = nearest_value_at_time(t, dv, tx_time)
        rel_tx = float(tx_disp - sx_disp)
        rel_series = dv - sx_disp

        ann_times = ann_df["time"].to_numpy(dtype=float)
        ann_labels = ann_df["label"].astype(str).to_numpy()
        align_plot = signal_out / "predict_time_alignment_debug.png"
        plot_time_alignment_debug(raw_t, t, dv, ann_times, ann_labels, ref_start_time=ref_start_time, signal_start_time=sig_start_time, shift_sec=shift_sec, signal_column=signal_column, out_path=align_plot)

        mass_sx, k, c = predict_mass_from_disp(0.0, args.sensor_cm, args.load_cm, model)
        mass_tx, _, _ = predict_mass_from_disp(rel_tx, args.sensor_cm, args.load_cm, model)

        row_df = pd.DataFrame(
            [
                {"point": "sx", "time": float(sx_time), "absolute_displacement": float(sx_disp), "relative_displacement": 0.0, "pred_mass": float(mass_sx)},
                {"point": "tx", "time": float(tx_time), "absolute_displacement": float(tx_disp), "relative_displacement": float(rel_tx), "pred_mass": float(mass_tx)},
            ]
        )
        row_df.to_csv(signal_out / "predict_sx_tx_points.csv", index=False, encoding="utf-8-sig")

        out = {
            "mode": "sx_tx_mass_delta_from_data",
            "signal_column": signal_column,
            "sensor_cm": float(args.sensor_cm),
            "load_cm": float(args.load_cm),
            "time_origin_applied_sec": float(time_origin_sec),
            "time_origin_source": time_origin_source,
            "time_origin_anchor_code": normalize_sensor_code(args.time_origin_anchor_code),
            "sx_label": args.sx_label,
            "tx_label": args.tx_label,
            "sx_time": float(sx_time),
            "tx_time": float(tx_time),
            "reference_signal_column": reference_signal,
            "reference_first_frame_time_source": ref_start_source,
            "reference_first_frame_time": float(ref_start_time),
            "signal_first_frame_time_source": sig_start_source,
            "signal_first_frame_time": float(sig_start_time),
            "time_shift_applied_sec": float(shift_sec),
            "sx_displacement_abs": float(sx_disp),
            "tx_displacement_abs": float(tx_disp),
            "sx_displacement_rel": 0.0,
            "tx_displacement_rel": float(rel_tx),
            "pred_mass_sx": float(mass_sx),
            "pred_mass_tx": float(mass_tx),
            "pred_mass_delta_tx_minus_sx": float(mass_tx - mass_sx),
            "k_disp_vs_mass": float(k),
            "c_disp_vs_mass": float(c),
            "output_csv": str(signal_out / "predict_sx_tx_points.csv"),
            "output_plot": str(signal_out / "predict_sx_tx_plot.png"),
            "alignment_debug_plot": str(align_plot),
        }

        plot_file_mode(t, rel_series, sx_time, tx_time, rel_tx, signal_column, signal_out / "predict_sx_tx_plot.png")
        with (signal_out / "predict_summary.json").open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        results.append(out)

    if not results:
        raise ValueError("No valid signals processed. Check --signal-columns and model files.")

    merged_df = pd.DataFrame(results)
    merged_csv = args.out_dir / "predict_multi_signal_summary.csv"
    merged_df.to_csv(merged_csv, index=False, encoding="utf-8-sig")

    if args.mass is None and "pred_mass_delta_tx_minus_sx" in merged_df.columns:
        plot_multi_signal_bars(
            merged_df,
            "pred_mass_delta_tx_minus_sx",
            "Predicted mass delta (tx-sx) by signal",
            args.out_dir / "predict_multi_signal_mass_delta.png",
        )
        plot_multi_signal_bars(
            merged_df,
            "tx_displacement_rel",
            "Relative displacement at tx by signal",
            args.out_dir / "predict_multi_signal_rel_disp_tx.png",
        )
    elif args.mass is not None and "pred_relative_displacement" in merged_df.columns:
        plot_multi_signal_bars(
            merged_df,
            "pred_relative_displacement",
            "Predicted relative displacement by signal",
            args.out_dir / "predict_multi_signal_rel_disp_from_mass.png",
        )
    elif args.disp_rel is not None and "pred_mass" in merged_df.columns:
        plot_multi_signal_bars(
            merged_df,
            "pred_mass",
            "Predicted mass by signal",
            args.out_dir / "predict_multi_signal_mass_from_rel_disp.png",
        )

    summary = {
        "signal_columns": requested_signals,
        "processed_signal_count": int(len(results)),
        "output_summary_csv": str(merged_csv),
        "results": results,
    }

    with (args.out_dir / "predict_multi_signal_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Keep CLI output compact: displacement/mass only. Full details remain in files.
    compact_rows: list[dict[str, Any]] = []
    for r in results:
        mode = str(r.get("mode", ""))
        signal = str(r.get("signal_column", ""))
        if mode == "mass_to_displacement":
            compact_rows.append({
                "signal_column": signal,
                "mass": float(r["mass"]),
                "pred_relative_displacement": float(r["pred_relative_displacement"]),
            })
        elif mode == "displacement_to_mass":
            compact_rows.append({
                "signal_column": signal,
                "input_relative_displacement": float(r["input_relative_displacement"]),
                "pred_mass": float(r["pred_mass"]),
            })
        else:
            compact_rows.append({
                "signal_column": signal,
                "sx_displacement_rel": float(r["sx_displacement_rel"]),
                "tx_displacement_rel": float(r["tx_displacement_rel"]),
                "pred_mass_sx": float(r["pred_mass_sx"]),
                "pred_mass_tx": float(r["pred_mass_tx"]),
            })

    cli_output = {
        "mode": str(results[0].get("mode", "")),
        "processed_signal_count": int(len(results)),
        "results": compact_rows,
    }
    print(json.dumps(cli_output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
