#!/usr/bin/env python3
import os
import csv
from datetime import datetime
from uuid import uuid4
import argparse

# Set env flags BEFORE importing evaluation to ensure flags are read
parser = argparse.ArgumentParser(description="Run sample evaluation on generated images")
parser.add_argument("--topic", default="컬러 사각형 비교", help="Topic text for BLIP2 similarity")
parser.add_argument("--prompt", default="빨간 사각형 이미지", help="Prompt text for CLIP similarity")
parser.add_argument("--ref", default="ref_red.png", help="Reference image filename in uploads/")
parser.add_argument("--enable-clip", action="store_true", help="Enable CLIP metric (requires internet/HF token)")
parser.add_argument("--enable-blip2", action="store_true", help="Enable BLIP2 metric (heavy model)")
parser.add_argument("--enable-lpips", action="store_true", help="Enable LPIPS metric")
args = parser.parse_args()

# Default: only LPIPS enabled for quick offline testing
os.environ.setdefault("DISABLE_CLIP", "1")
os.environ.setdefault("DISABLE_BLIP2", "1")
os.environ.setdefault("DISABLE_LPIPS", "0")

# Override by flags
if args.enable_clip:
    os.environ["DISABLE_CLIP"] = "0"
if args.enable_blip2:
    os.environ["DISABLE_BLIP2"] = "0"
if args.enable_lpips:
    os.environ["DISABLE_LPIPS"] = "0"

from PIL import Image, ImageChops
import sys

# Import after env flags are set and ensure project root in sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import evaluation  # noqa: E402

UPLOADS_DIR = os.path.join(ROOT, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


def ensure_sample_images():
    """Create three sample images if missing: red ref, red with noise, blue."""
    ref_path = os.path.join(UPLOADS_DIR, "ref_red.png")
    near_path = os.path.join(UPLOADS_DIR, "red_noise.png")
    far_path = os.path.join(UPLOADS_DIR, "blue.png")

    if not os.path.exists(ref_path):
        Image.new("RGB", (256, 256), (220, 30, 30)).save(ref_path)
    if not os.path.exists(near_path):
        base = Image.open(ref_path).copy()
        # Add slight noise by shifting and blending
        shifted = ImageChops.offset(base, 2, 2)
        Image.blend(base, shifted, alpha=0.2).save(near_path)
    if not os.path.exists(far_path):
        Image.new("RGB", (256, 256), (30, 60, 220)).save(far_path)

    return {
        "ref_red.png": ref_path,
        "red_noise.png": near_path,
        "blue.png": far_path,
    }


def run_eval(images_map):
    ref_name = args.ref
    if ref_name not in images_map:
        raise SystemExit(f"Reference image '{ref_name}' not found in uploads/")

    evaluator = evaluation.get_evaluator()

    results = []
    for name, path in images_map.items():
        if name == ref_name:
            continue
        r = evaluator.evaluate(path, images_map[ref_name], args.prompt, args.topic)
        r["filename"] = name
        results.append(r)
    return results


def write_csv(results):
    csv_filename = f"sample_results_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}.csv"
    csv_path = os.path.join(UPLOADS_DIR, csv_filename)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "filename",
            "clip_score",
            "clip_norm",
            "blip2_similarity",
            "blip2_norm",
            "lpips_score",
            "lpips_norm",
            "final_score",
        ])
        for r in results:
            w.writerow([
                r["filename"],
                r["clip_score"],
                r["clip_norm"],
                r["blip2_similarity"],
                r["blip2_norm"],
                r["lpips_score"],
                r["lpips_norm"],
                r["final_score"],
            ])
    return csv_filename, csv_path


def main():
    images_map = ensure_sample_images()
    results = run_eval(images_map)
    csv_name, csv_path = write_csv(results)

    # Pretty print
    print("\nResults:\n")
    for r in results:
        print(
            f"- {r['filename']}: final={r['final_score']:.2f} "
            f"(clip {r['clip_score']:.4f}/{r['clip_norm']:.2f}, "
            f"blip2 {r['blip2_similarity']:.4f}/{r['blip2_norm']:.2f}, "
            f"lpips {r['lpips_score']:.4f}/{r['lpips_norm']:.2f})"
        )
    print(f"\nCSV saved: {csv_path}")


if __name__ == "__main__":
    main()
