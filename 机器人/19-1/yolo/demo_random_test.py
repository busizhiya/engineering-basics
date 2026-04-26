from __future__ import annotations

import argparse
import random
from pathlib import Path

from ultralytics import YOLO


def read_images_from_dir(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(f"Input dir created, please put images here: {input_dir}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts])
    if not images:
        raise ValueError(f"No images found in input dir: {input_dir}")
    return images


def latest_best_pt(root: Path) -> Path:
    candidate = root / "runs" / "yolov11n_my_data1" / "weights" / "best.pt"
    if candidate.exists():
        return candidate
    matches = sorted(root.glob("runs/**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError("Could not find any best.pt under runs/**/weights/")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Random demo inference on new images from input directory")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Dataset root")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory containing new images")
    parser.add_argument("--weights", type=Path, default=None, help="Path to trained .pt weights")
    parser.add_argument("--num", type=int, default=8, help="Number of random demo images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    root = args.root.resolve()
    input_dir = args.input_dir.resolve() if args.input_dir else root / "input"
    weights = args.weights.resolve() if args.weights else latest_best_pt(root)

    image_paths = read_images_from_dir(input_dir)
    k = min(max(args.num, 1), len(image_paths))
    rng = random.Random(args.seed)
    sampled = rng.sample(image_paths, k)

    print(f"Using weights: {weights}")
    print(f"Using input dir: {input_dir}")
    print(f"Sampling {k}/{len(image_paths)} images")

    model = YOLO(str(weights))
    run_name = f"random_demo_input_{args.seed}_{k}"
    results = model.predict(
        source=[str(p) for p in sampled],
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        save_txt=False,
        project=str(root / "demo_outputs"),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )

    print("Sampled images:")
    for p in sampled:
        print(f"- {p}")

    if results:
        save_dir = Path(results[0].save_dir)
        print(f"Demo outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()