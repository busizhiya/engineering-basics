from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import font_manager

from plot_sensors import DEFAULT_SENSOR_DEF, SensorMeta, load_sensor_definitions


DEFAULT_OUTPUT = Path(__file__).with_name("outputs") / "sensor_positions_3d.png"


def configure_matplotlib_fonts() -> None:
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "PingFang SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_fonts = [font_name for font_name in preferred_fonts if font_name in available_fonts]

    if selected_fonts:
        plt.rcParams["font.sans-serif"] = selected_fonts + ["DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False


def fallback_color(index: int) -> str:
    palette = [
        "#58a6ff",
        "#7ee787",
        "#f2cc60",
        "#2dd4bf",
        "#ffb86b",
        "#ff7b72",
        "#c792ea",
        "#ffd166",
    ]
    return palette[index % len(palette)]


def compute_fallback_positions(count: int) -> list[tuple[float, float, float]]:
    if count <= 0:
        return []

    cols = max(3, math.ceil(math.sqrt(count * 1.2)))
    rows = math.ceil(count / cols)
    spacing = 2.0
    x_offset = (cols - 1) * spacing / 2
    y_offset = (rows - 1) * spacing / 2

    positions: list[tuple[float, float, float]] = []
    for index in range(count):
        row = index // cols
        col = index % cols
        x_coord = col * spacing - x_offset
        y_coord = y_offset - row * spacing
        z_coord = 0.0
        positions.append((x_coord, y_coord, z_coord))
    return positions


def sensor_label(meta: SensorMeta, code: str) -> str:
    if meta.name and meta.name.strip():
        return f"{meta.name}"
    return code


def plot_sensor_positions(sensor_defs: dict[str, SensorMeta], title: str):
    items = sorted(sensor_defs.items(), key=lambda item: item[0])
    if not items:
        raise SystemExit("No sensor definitions found.")

    fallback_positions = compute_fallback_positions(len(items))

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("white")

    all_x: list[float] = []
    all_y: list[float] = []
    all_z: list[float] = []

    for index, ((code, meta), fallback_position) in enumerate(zip(items, fallback_positions)):
        position = meta.position if meta.position is not None else None
        if position is not None:
            x_coord, y_coord, z_coord = float(position[0]), float(position[1]), float(position[2])
        else:
            x_coord, y_coord, z_coord = fallback_position

        color = meta.color if meta.color else fallback_color(index)
        ax.scatter([x_coord], [y_coord], [z_coord], s=50, color=color, edgecolor="#1f2937", linewidth=0.6)
        ax.text(
            x_coord,
            y_coord,
            z_coord,
            f" {sensor_label(meta, code)}",
            fontsize=5.5,
            color="#111827",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.7),
        )

        all_x.append(x_coord)
        all_y.append(y_coord)
        all_z.append(z_coord)

    max_range = max(
        max(all_x) - min(all_x),
        max(all_y) - min(all_y),
        max(all_z) - min(all_z),
        1.0,
    )
    center_x = (max(all_x) + min(all_x)) / 2
    center_y = (max(all_y) + min(all_y)) / 2
    center_z = (max(all_z) + min(all_z)) / 2
    # Slight zoom-in for denser framing while keeping all points visible.
    radius = max_range * 0.36

    ax.set_xlim(center_x - radius, center_x + radius)
    ax.set_ylim(center_y - radius, center_y + radius)
    ax.set_zlim(center_z - radius, center_z + radius)

    ax.set_xlabel("X", fontsize=12, labelpad=10)
    ax.set_ylabel("Y", fontsize=12, labelpad=10)
    ax.set_zlabel("Z", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=16, pad=18)

    # Draw coordinate axes through the center for a cleaner spatial reference.
    ax.plot([center_x - radius, center_x + radius], [center_y, center_y], [center_z, center_z], color="#ef4444", linewidth=1.2)
    ax.plot([center_x, center_x], [center_y - radius, center_y + radius], [center_z, center_z], color="#10b981", linewidth=1.2)
    ax.plot([center_x, center_x], [center_y, center_y], [center_z - radius, center_z + radius], color="#3b82f6", linewidth=1.2)

    ax.view_init(elev=24, azim=-52)
    ax.grid(True, alpha=0.22)
    ax.set_box_aspect((1, 1, 0.7))

    return fig


def main() -> None:
    configure_matplotlib_fonts()

    parser = argparse.ArgumentParser(description="Plot sensor positions and names in a 3D coordinate system.")
    parser.add_argument("--sensor-def", default=str(DEFAULT_SENSOR_DEF), help="Sensor definition JSON path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output image path")
    parser.add_argument("--title", default="传感器三维位置图", help="Figure title")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    args = parser.parse_args()

    sensor_def_path = Path(args.sensor_def).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    sensor_defs = load_sensor_definitions(sensor_def_path)
    if not sensor_defs:
        raise SystemExit(f"No sensor definitions found in: {sensor_def_path}")

    fig = plot_sensor_positions(sensor_defs, args.title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved to: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
