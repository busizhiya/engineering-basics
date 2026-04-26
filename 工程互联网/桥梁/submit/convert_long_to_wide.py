from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plot_sensors import parse_optional_float, parse_time_value


@dataclass
class LongSeries:
    code: str
    sensor_type: str
    times: np.ndarray
    values: np.ndarray


def _scalar_suffix(sensor_type: str) -> str:
    if sensor_type == "sgd":
        return "chdata"
    if sensor_type == "input":
        return "dist"
    return "value"

    
def _format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def load_long_series(csv_path: Path) -> list[LongSeries]:
    grouped: dict[tuple[str, str], list[tuple[float, tuple[float, ...]]]] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"code", "type", "time", "value1"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Input CSV is not in old long format: missing required columns code/type/time/value1")

        for row in reader:
            code = (row.get("code") or "").strip()
            sensor_type = (row.get("type") or "").strip()
            if not code or not sensor_type:
                continue

            time_value = parse_time_value(row.get("time"))
            if time_value is None:
                continue

            if sensor_type == "vibr":
                x_val = parse_optional_float(row.get("value1"))
                y_val = parse_optional_float(row.get("value2"))
                z_val = parse_optional_float(row.get("value3"))
                if x_val is None or y_val is None or z_val is None:
                    continue
                values = (x_val, y_val, z_val)
            else:
                scalar = parse_optional_float(row.get("value1"))
                if scalar is None:
                    continue
                values = (scalar,)

            grouped.setdefault((code, sensor_type), []).append((time_value, values))

    series_list: list[LongSeries] = []
    for (code, sensor_type), samples in grouped.items():
        samples.sort(key=lambda item: item[0])
        times = np.array([item[0] for item in samples], dtype=float)
        if sensor_type == "vibr":
            values = np.array([item[1] for item in samples], dtype=float)
        else:
            values = np.array([item[1][0] for item in samples], dtype=float)[:, np.newaxis]
        series_list.append(LongSeries(code=code, sensor_type=sensor_type, times=times, values=values))

    series_list.sort(key=lambda item: (item.sensor_type, item.code))
    if not series_list:
        raise ValueError("No valid sensor samples found in input CSV")
    return series_list


def estimate_dt(series_list: list[LongSeries]) -> float:
    diffs: list[float] = []
    for series in series_list:
        if len(series.times) < 2:
            continue
        current = np.diff(series.times)
        positive = current[current > 1e-9]
        if positive.size > 0:
            diffs.append(float(np.median(positive)))

    if not diffs:
        return 0.02
    return max(1e-6, float(np.median(np.array(diffs, dtype=float))))


def align_start_times(series_list: list[LongSeries]) -> list[LongSeries]:
    aligned: list[LongSeries] = []
    for series in series_list:
        base = float(series.times[0])
        aligned_times = series.times - base
        aligned.append(LongSeries(code=series.code, sensor_type=series.sensor_type, times=aligned_times, values=series.values))
    return aligned


def build_timeline(series_list: list[LongSeries], dt: float, mode: str) -> np.ndarray:
    starts = [float(series.times[0]) for series in series_list]
    ends = [float(series.times[-1]) for series in series_list]

    if mode == "intersection":
        start = max(starts)
        end = min(ends)
        if end <= start:
            print("No overlapping time range across sensors; falling back to union timeline.")
            start = min(starts)
            end = max(ends)
    else:
        start = min(starts)
        end = max(ends)

    count = int(np.floor((end - start) / dt)) + 1
    count = max(count, 2)
    return start + np.arange(count, dtype=float) * dt


def interpolate_series_at(series: LongSeries, timeline: np.ndarray) -> np.ndarray:
    result = np.full((timeline.shape[0], series.values.shape[1]), np.nan, dtype=float)

    left = float(series.times[0])
    right = float(series.times[-1])
    mask = (timeline >= left) & (timeline <= right)
    if not np.any(mask):
        return result

    valid_t = timeline[mask]
    for dim in range(series.values.shape[1]):
        result[mask, dim] = np.interp(valid_t, series.times, series.values[:, dim])
    return result


def convert_long_to_wide(input_csv: Path, output_csv: Path, dt: float | None, timeline_mode: str, align_start: bool) -> None:
    series_list = load_long_series(input_csv)
    if align_start:
        series_list = align_start_times(series_list)

    resolved_dt = estimate_dt(series_list) if dt is None else max(1e-6, dt)
    timeline = build_timeline(series_list, resolved_dt, timeline_mode)

    interpolated: dict[tuple[str, str], np.ndarray] = {}
    for series in series_list:
        interpolated[(series.code, series.sensor_type)] = interpolate_series_at(series, timeline)

    fieldnames = ["time"]
    for series in series_list:
        if series.sensor_type == "vibr":
            fieldnames.extend([f"{series.code}.AccX", f"{series.code}.AccY", f"{series.code}.AccZ"])
        else:
            fieldnames.append(f"{series.code}.{_scalar_suffix(series.sensor_type)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for idx, time_value in enumerate(timeline):
            row: dict[str, str] = {"time": _format_float(float(time_value))}
            for series in series_list:
                values = interpolated[(series.code, series.sensor_type)][idx]
                if series.sensor_type == "vibr":
                    row[f"{series.code}.AccX"] = "" if np.isnan(values[0]) else _format_float(float(values[0]))
                    row[f"{series.code}.AccY"] = "" if np.isnan(values[1]) else _format_float(float(values[1]))
                    row[f"{series.code}.AccZ"] = "" if np.isnan(values[2]) else _format_float(float(values[2]))
                else:
                    col = f"{series.code}.{_scalar_suffix(series.sensor_type)}"
                    row[col] = "" if np.isnan(values[0]) else _format_float(float(values[0]))
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert old long sensor CSV format to synchronized wide format.")
    parser.add_argument("--input", default="2026.3.30_test.csv", help="Old-format CSV path (one row per sensor sample)")
    parser.add_argument("--output", default="2026.3.30_test_wide.csv", help="Output wide-format CSV path")
    parser.add_argument("--dt", type=float, default=None, help="Target sampling interval in seconds; default auto-estimate")
    parser.add_argument(
        "--timeline",
        choices=["intersection", "union"],
        default="intersection",
        help="intersection keeps only overlapping time range; union keeps full range with blanks outside sensor coverage",
    )
    parser.add_argument(
        "--no-align-start",
        action="store_true",
        help="Disable per-sensor start-time alignment (default is enabled)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    convert_long_to_wide(input_path, output_path, args.dt, args.timeline, align_start=not args.no_align_start)
    print(f"Converted to wide format: {output_path}")
