from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from manim import *

    MANIM_AVAILABLE = True
except ModuleNotFoundError as import_error:
    MANIM_AVAILABLE = False
    MANIM_IMPORT_ERROR = import_error

import numpy as np


DEFAULT_CSV = Path(__file__).with_name("vibrate_wide.csv")
DEFAULT_SENSOR_DEF = Path(__file__).with_name("sensor_def.json")
DEFAULT_OUTPUT_TITLE = "3.30数据"
DEFAULT_TARGET_SAMPLES = 18000
DEFAULT_MAX_RUNTIME_SECONDS = 0.0
DEFAULT_TIME_SCALE = 1.0


@dataclass
class SensorSeries:
    code: str
    sensor_type: str
    samples: list[tuple[float, tuple[float, ...]]]

    @property
    def first_time(self) -> float:
        return self.samples[0][0]

    @property
    def last_time(self) -> float:
        return self.samples[-1][0]

    @property
    def duration(self) -> float:
        return max(self.last_time - self.first_time, 0.0)


@dataclass
class SensorMeta:
    code: str
    position: np.ndarray
    color: str | None
    name: str | None


PALETTE = [
    "#58a6ff",
    "#7ee787",
    "#f2cc60",
    "#2dd4bf",
    "#ffb86b",
    "#ff7b72",
    "#c792ea",
    "#ffd166",
]


def parse_optional_float(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    raw_value = raw_value.strip()
    if not raw_value:
        return None
    if "|" in raw_value:
        raw_value = raw_value.split("|", 1)[0].strip()
    if raw_value.startswith('="') and raw_value.endswith('"'):
        raw_value = raw_value[2:-1]
    try:
        return float(raw_value)
    except ValueError:
        return None


def parse_time_value(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    text = raw_value.strip()
    if not text:
        return None

    if text.startswith('="') and text.endswith('"'):
        text = text[2:-1]

    try:
        return float(text)
    except ValueError:
        pass

    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return None


def load_sensor_series_long_format(rows: list[dict[str, str]]) -> list[SensorSeries]:
    grouped: dict[tuple[str, str], list[tuple[float, tuple[float, ...]]]] = {}

    for row in rows:
        code = (row.get("code") or "").strip()
        sensor_type = (row.get("type") or "").strip()
        if not code or not sensor_type:
            continue

        time_value = parse_time_value(row.get("time"))
        if time_value is None:
            continue

        if sensor_type == "vibr":
            values = (
                parse_optional_float(row.get("value1")),
                parse_optional_float(row.get("value2")),
                parse_optional_float(row.get("value3")),
            )
            if any(component is None for component in values):
                continue
            sample = (time_value, tuple(float(component) for component in values))
        else:
            value = parse_optional_float(row.get("value1"))
            if value is None:
                continue
            sample = (time_value, (float(value),))

        grouped.setdefault((code, sensor_type), []).append(sample)

    series_list = [
        SensorSeries(code=code, sensor_type=sensor_type, samples=sorted(samples, key=lambda item: item[0]))
        for (code, sensor_type), samples in grouped.items()
        if samples
    ]
    series_list.sort(key=lambda item: (item.first_time, item.sensor_type, item.code))
    return series_list


def load_sensor_series_wide_format(rows: list[dict[str, str]], fieldnames: list[str]) -> list[SensorSeries]:
    vibr_columns: dict[str, dict[str, str]] = {}
    scalar_columns: list[tuple[str, str, str]] = []

    for column in fieldnames:
        if column == "time" or "." not in column:
            continue
        code, suffix = column.split(".", 1)
        if suffix in {"AccX", "AccY", "AccZ"}:
            vibr_columns.setdefault(code, {})[suffix] = column
        elif suffix == "chdata":
            scalar_columns.append((code, column, "sgd"))
        else:
            scalar_columns.append((code, column, "input"))

    grouped: dict[tuple[str, str], list[tuple[float, tuple[float, ...]]]] = {}

    for row in rows:
        time_value = parse_time_value(row.get("time"))
        if time_value is None:
            continue

        for code, axis_map in vibr_columns.items():
            col_x = axis_map.get("AccX")
            col_y = axis_map.get("AccY")
            col_z = axis_map.get("AccZ")
            if not col_x or not col_y or not col_z:
                continue

            values = (
                parse_optional_float(row.get(col_x)),
                parse_optional_float(row.get(col_y)),
                parse_optional_float(row.get(col_z)),
            )
            if any(component is None for component in values):
                continue
            grouped.setdefault((code, "vibr"), []).append((time_value, (float(values[0]), float(values[1]), float(values[2]))))

        for code, column, sensor_type in scalar_columns:
            value = parse_optional_float(row.get(column))
            if value is None:
                continue
            grouped.setdefault((code, sensor_type), []).append((time_value, (float(value),)))

    series_list = [
        SensorSeries(code=code, sensor_type=sensor_type, samples=sorted(samples, key=lambda item: item[0]))
        for (code, sensor_type), samples in grouped.items()
        if samples
    ]
    series_list.sort(key=lambda item: (item.first_time, item.sensor_type, item.code))
    return series_list


def load_sensor_series(csv_path: Path) -> list[SensorSeries]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if {"code", "type", "time", "value1"}.issubset(set(fieldnames)):
        return load_sensor_series_long_format(rows)
    return load_sensor_series_wide_format(rows, fieldnames)


def load_sensor_definitions(sensor_def_path: Path) -> dict[str, SensorMeta]:
    if not sensor_def_path.exists():
        return {}

    with sensor_def_path.open("r", encoding="utf-8") as handle:
        raw_items = json.load(handle)

    mapping: dict[str, SensorMeta] = {}
    if not isinstance(raw_items, list):
        return mapping

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        code = str(item.get("code") or "").strip()
        position = item.get("position")
        if not code or not isinstance(position, list) or len(position) < 3:
            continue

        try:
            pos = np.array([float(position[0]), float(position[1]), float(position[2])], dtype=float)
        except (TypeError, ValueError):
            continue

        color = item.get("color")
        name = item.get("name")
        mapping[code] = SensorMeta(code=code, position=pos, color=color if isinstance(color, str) else None, name=name if isinstance(name, str) else None)

    return mapping


def downsample_samples(samples: list[tuple[float, tuple[float, ...]]], target_count: int) -> list[tuple[float, tuple[float, ...]]]:
    if len(samples) <= target_count:
        return samples
    if target_count <= 1:
        return [samples[0]]

    step = (len(samples) - 1) / (target_count - 1)
    selected: list[tuple[float, tuple[float, ...]]] = []
    last_index = -1
    for index in range(target_count):
        current_index = min(len(samples) - 1, round(index * step))
        if current_index != last_index:
            selected.append(samples[current_index])
            last_index = current_index
    if selected[-1] != samples[-1]:
        selected.append(samples[-1])
    return selected


def nice_step(span: float, target_ticks: int = 5) -> float:
    if span <= 0:
        return 1.0
    raw_step = span / target_ticks
    power = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / power
    if normalized < 1.5:
        nice = 1.0
    elif normalized < 3.5:
        nice = 2.0
    elif normalized < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * power


def get_vector_limit(series: SensorSeries) -> float:
    peak = 0.0
    for _, values in series.samples:
        for value in values:
            peak = max(peak, abs(value))
    return max(peak, 1.0)


def interpolate_sample(samples: list[tuple[float, tuple[float, ...]]], position: float) -> tuple[float, ...]:
    if position <= 0:
        return samples[0][1]
    if position >= len(samples) - 1:
        return samples[-1][1]

    left_index = int(math.floor(position))
    right_index = min(left_index + 1, len(samples) - 1)
    blend = position - left_index
    left_values = samples[left_index][1]
    right_values = samples[right_index][1]
    return tuple(left + (right - left) * blend for left, right in zip(left_values, right_values))


def interpolate_sample_point(samples: list[tuple[float, tuple[float, ...]]], position: float) -> tuple[float, tuple[float, ...]]:
    if position <= 0:
        return samples[0]
    if position >= len(samples) - 1:
        return samples[-1]

    left_index = int(math.floor(position))
    right_index = min(left_index + 1, len(samples) - 1)
    blend = position - left_index

    left_time, left_values = samples[left_index]
    right_time, right_values = samples[right_index]
    current_time = left_time + (right_time - left_time) * blend
    current_values = tuple(left + (right - left) * blend for left, right in zip(left_values, right_values))
    return current_time, current_values


def interpolate_values_by_elapsed_time(samples: list[tuple[float, tuple[float, ...]]], elapsed_time: float) -> tuple[float, ...]:
    """Interpolate sample values using per-sensor elapsed time (t - t0)."""
    if len(samples) == 1:
        return samples[0][1]

    first_time = samples[0][0]
    target_time = first_time + max(0.0, elapsed_time)

    if target_time <= samples[0][0]:
        return samples[0][1]
    if target_time >= samples[-1][0]:
        return samples[-1][1]

    for index in range(1, len(samples)):
        right_time, right_values = samples[index]
        left_time, left_values = samples[index - 1]
        if target_time <= right_time:
            span = right_time - left_time
            if span <= 0:
                return right_values
            blend = (target_time - left_time) / span
            return tuple(left + (right - left) * blend for left, right in zip(left_values, right_values))

    return samples[-1][1]


if MANIM_AVAILABLE:

    class SensorVisualizationScene(ThreeDScene):
        def construct(self) -> None:
            csv_path = Path(os.environ.get("SENSOR_CSV", str(DEFAULT_CSV))).expanduser().resolve()
            if not csv_path.exists():
                message = Text(f"CSV not found: {csv_path}", font_size=28)
                self.add_fixed_in_frame_mobjects(message)
                self.wait(2)
                return

            self.camera.background_color = "#0b1020"
            self.target_samples = max(12, int(os.environ.get("SENSOR_TARGET_SAMPLES", str(DEFAULT_TARGET_SAMPLES))))
            self.max_runtime_seconds = float(os.environ.get("SENSOR_MAX_RUNTIME_SECONDS", str(DEFAULT_MAX_RUNTIME_SECONDS)))
            self.time_scale = max(0.05, float(os.environ.get("SENSOR_TIME_SCALE", str(DEFAULT_TIME_SCALE))))
            self.show_labels = os.environ.get("SENSOR_SHOW_LABELS", "0") == "1"
            series_list = load_sensor_series(csv_path)
            sensor_defs = load_sensor_definitions(Path(os.environ.get("SENSOR_DEF_PATH", str(DEFAULT_SENSOR_DEF))).expanduser().resolve())
            vibr_count = len([series for series in series_list if series.sensor_type == "vibr"])
            scalar_count = len(series_list) - vibr_count
            self.timeline_duration_seconds = max((series.duration for series in series_list), default=4.0)
            self._show_title(csv_path, len(series_list), vibr_count, scalar_count)

            origins = self._resolve_sensor_origins(series_list, sensor_defs)
            axes = self._build_world_axes(origins)
            self.add(axes)

            scene_radius = max((np.linalg.norm(origin[:2]) for origin in origins), default=3.0)
            zoom = min(1.15, max(0.38, 5.2 / max(scene_radius, 1.0)))
            self.set_camera_orientation(phi=72 * DEGREES, theta=-55 * DEGREES, zoom=zoom)

            tracker = ValueTracker(0.0)
            sensor_groups = VGroup()
            for index, (series, origin) in enumerate(zip(series_list, origins)):
                fallback_color = PALETTE[index % len(PALETTE)]
                sensor_meta = sensor_defs.get(series.code)
                color = sensor_meta.color if sensor_meta and sensor_meta.color else fallback_color
                sensor_groups.add(self._build_sensor_glyph(series, origin, color, tracker, sensor_meta))

            self.add(sensor_groups)
            self.begin_ambient_camera_rotation(rate=0.14, about="theta")

            run_time = max(4.0, self.timeline_duration_seconds * self.time_scale)
            if self.max_runtime_seconds > 0:
                run_time = min(run_time, self.max_runtime_seconds)
            self.play(tracker.animate.set_value(1.0), run_time=run_time, rate_func=linear)
            self.wait(0.5)
            self.stop_ambient_camera_rotation()

            outro = Text("End of render", font_size=30, color="#9aa4b2")
            outro.to_edge(DOWN)
            self.add_fixed_in_frame_mobjects(outro)
            self.play(FadeIn(outro), run_time=0.5)
            self.wait(1.0)

        def _resolve_sensor_origins(self, series_list: list[SensorSeries], sensor_defs: dict[str, SensorMeta]) -> list[np.ndarray]:
            fallback_positions = self._compute_origin_positions(len(series_list))
            origins: list[np.ndarray] = []
            for index, series in enumerate(series_list):
                meta = sensor_defs.get(series.code)
                if meta is not None:
                    origins.append(meta.position)
                else:
                    origins.append(fallback_positions[index])
            return origins

        def _compute_origin_positions(self, count: int) -> list[np.ndarray]:
            if count <= 0:
                return []

            cols = max(3, math.ceil(math.sqrt(count * 1.2)))
            rows = math.ceil(count / cols)

            max_span_x = config.frame_width * 0.65
            max_span_y = config.frame_height * 0.55
            spacing_x = max_span_x / max(cols - 1, 1)
            spacing_y = max_span_y / max(rows - 1, 1)
            spacing = min(1.9, spacing_x, spacing_y)

            x_offset = (cols - 1) * spacing / 2
            y_offset = (rows - 1) * spacing / 2

            positions: list[np.ndarray] = []
            for index in range(count):
                row = index // cols
                col = index % cols
                x_coord = col * spacing - x_offset
                y_coord = y_offset - row * spacing
                positions.append(np.array([x_coord, y_coord, 0.0]))
            return positions

        def _build_world_axes(self, origins: list[np.ndarray]) -> ThreeDAxes:
            if origins:
                max_x = max(abs(origin[0]) for origin in origins) + 2.3
                max_y = max(abs(origin[1]) for origin in origins) + 2.3
            else:
                max_x = 4.0
                max_y = 4.0

            return ThreeDAxes(
                x_range=[-max_x, max_x, nice_step(max_x / 3.0)],
                y_range=[-max_y, max_y, nice_step(max_y / 3.0)],
                z_range=[-3.0, 3.0, 1.0],
                x_length=8.4,
                y_length=8.4,
                z_length=4.0,
            )

        def _build_sensor_glyph(
            self,
            series: SensorSeries,
            origin: np.ndarray,
            color: ManimColor,
            tracker: ValueTracker,
            sensor_meta: SensorMeta | None,
        ) -> VGroup:
            samples = downsample_samples(series.samples, target_count=self.target_samples)
            scale = self._series_scale(series)
            origin_dot = Dot(point=origin, color=color, radius=0.035)

            initial_end = self._current_end_point(series, samples, origin, scale, tracker)
            shaft = Line(start=origin, end=initial_end, color=color, stroke_width=3.0)
            tip = Dot(point=initial_end, color=color, radius=0.028)

            def update_vector(_: Mobject) -> None:
                end_point = self._current_end_point(series, samples, origin, scale, tracker)
                shaft.put_start_and_end_on(origin, end_point)
                tip.move_to(end_point)

            shaft.add_updater(update_vector)
            tip.add_updater(update_vector)

            glyph = VGroup(origin_dot, shaft, tip)
            if self.show_labels:
                label_text = sensor_meta.name if sensor_meta and sensor_meta.name else series.code
                label = Text(label_text, font_size=14, color=color)
                label.next_to(origin_dot, UP, buff=0.12)
                glyph.add(label)

            return glyph

        def _series_scale(self, series: SensorSeries) -> float:
            if series.sensor_type == "vibr":
                limit = get_vector_limit(series)
            else:
                limit = max((abs(values[0]) for _, values in series.samples), default=1.0)
            return 1.05 / max(limit, 1e-6)

        def _current_end_point(
            self,
            series: SensorSeries,
            samples: list[tuple[float, tuple[float, ...]]],
            origin: np.ndarray,
            scale: float,
            tracker: ValueTracker,
        ) -> np.ndarray:
            elapsed_time = tracker.get_value() * self.timeline_duration_seconds
            values = interpolate_values_by_elapsed_time(samples, elapsed_time)

            if series.sensor_type == "vibr":
                end_point = origin + np.array(values) * scale
            else:
                end_point = origin + np.array([0.0, 0.0, values[0] * scale])

            if np.linalg.norm(end_point - origin) < 1e-5:
                end_point = origin + np.array([0.0, 0.0, 1e-3])
            return end_point

        def _show_title(self, csv_path: Path, total_series: int, vibr_count: int, scalar_count: int) -> None:
            title = Text(DEFAULT_OUTPUT_TITLE, font_size=40)
            title.to_edge(UP).shift(LEFT * 2.8)
            subtitle = Text(
                f"{csv_path.name} | groups={total_series} | vibr={vibr_count} | scalar={scalar_count}",
                font_size=24,
                color="#9aa4b2",
            )
            subtitle.next_to(title, DOWN, buff=0.18).align_to(title, LEFT)
            self.add_fixed_in_frame_mobjects(title, subtitle)
            self.play(FadeIn(title, shift=DOWN * 0.2), FadeIn(subtitle, shift=DOWN * 0.2), run_time=0.9)
            self.wait(0.2)

else:
    class SensorVisualizationScene:  # pragma: no cover
        pass


def print_usage() -> None:
    print(
        "Render with: python -m manim plot_sensors.py SensorVisualizationScene -qm\n"
        f"CSV override: set SENSOR_CSV={DEFAULT_CSV.name} or point it to another file\n"
        "Runtime: SENSOR_TIME_SCALE=1.0 follows real duration; set SENSOR_MAX_RUNTIME_SECONDS>0 to cap\n"
        "Speed/clarity: SENSOR_TARGET_SAMPLES controls per-sensor downsampling"
    )


if __name__ == "__main__":
    if not MANIM_AVAILABLE:
        print(f"Manim is not installed: {MANIM_IMPORT_ERROR}")
        print_usage()
        sys.exit(1)

    if os.environ.get("MANIM_SKIP_AUTORUN"):
        print_usage()
        sys.exit(0)

    scene_name = "SensorVisualizationScene"
    command = [
        sys.executable,
        "-m",
        "manim",
        str(Path(__file__).resolve()),
        scene_name,
        "-qm",
    ]
    raise SystemExit(subprocess.call(command))
