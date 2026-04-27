from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from src.data_utils import DRIVER_COLS, load_and_clean_csv, save_cleaned_sequence, save_meta


def fft_amplitude(x: np.ndarray, fs: float):
    x = x - np.mean(x)
    n = len(x)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    amp = np.abs(np.fft.rfft(x)) / max(n, 1)
    return freq, amp


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV 数据清洗、去趋势、滤波并导出 cleaned csv")
    parser.add_argument("--raw_dir", type=str, default="raw_data", help="原始 CSV 文件夹")
    parser.add_argument("--out_dir", type=str, default="processed/cleaned", help="清洗后 csv 输出目录")
    parser.add_argument("--meta_path", type=str, default="processed/meta.json", help="元数据输出路径")
    parser.add_argument("--fft_plot", type=str, default="plots/fft_driver_z.png", help="FFT 图输出路径")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    meta_path = Path(args.meta_path)
    fft_plot = Path(args.fft_plot)

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{raw_dir} 中没有 csv 文件")

    meta_records = []
    all_raw_driver_z: List[np.ndarray] = []
    all_filtered_driver_z: List[np.ndarray] = []
    fs_values: List[float] = []

    for csv_file in csv_files:
        seq = load_and_clean_csv(csv_file)
        save_cleaned_sequence(seq, out_dir / f"{seq.name}.csv")

        meta_records.append(
            {
                "name": seq.name,
                "samples": int(seq.time_s.shape[0]),
                "fs": float(seq.fs),
                "input_dim_forward": int(seq.x_forward.shape[1]),
                "output_dim_forward": int(seq.y_forward.shape[1]),
            }
        )
        fs_values.append(float(seq.fs))
        all_raw_driver_z.append(seq.raw_driver_z)
        all_filtered_driver_z.append(seq.filtered_driver_z)

    fs_use = float(np.median(fs_values)) if fs_values else 50.0
    raw_concat = np.concatenate(all_raw_driver_z, axis=0)
    filt_concat = np.concatenate(all_filtered_driver_z, axis=0)

    f_raw, a_raw = fft_amplitude(raw_concat, fs_use)
    f_filt, a_filt = fft_amplitude(filt_concat, fs_use)

    valid = f_filt >= 0.2
    if np.any(valid):
        dom_idx = int(np.argmax(a_filt[valid]))
        dom_freq = float(f_filt[valid][dom_idx])
    else:
        dom_freq = 0.0

    fft_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(f_raw, a_raw, label="Driver Z - Raw", alpha=0.7)
    plt.plot(f_filt, a_filt, label="Driver Z - Filtered", alpha=0.9)
    if dom_freq > 0:
        plt.axvline(dom_freq, color="r", linestyle="--", label=f"Dominant ~ {dom_freq:.3f} Hz")
    plt.xlim(0, 25)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("FFT of Driver Z (WSMS00012.AccZ)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fft_plot, dpi=150)
    plt.close()

    meta = {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "files": meta_records,
        "driver_cols": DRIVER_COLS,
        "dominant_frequency_hz": dom_freq,
        "fs_median": fs_use,
        "fft_plot": str(fft_plot),
    }
    save_meta(meta, meta_path)

    print(f"预处理完成: {len(meta_records)} 个文件")
    print(f"主频(驱动Z): {dom_freq:.4f} Hz")
    print(f"FFT图已保存: {fft_plot}")


if __name__ == "__main__":
    main()
