"""
Compute PSNR and SSIM between saved SR outputs and ground truth HR images.
Assumes outputs were saved by sr_infer.py into OUTPUT_DIR with naming:
  <basename>_sr_x{scale}.png
  <basename>_hr_gt.png
"""

import os
import csv
import numpy as np
from skimage import io, img_as_float32
from utils import list_image_files, compute_psnr_ssim
from config import OUTPUT_DIR, HR_TEST_DIR, SCALES
from tqdm import tqdm

METHODS = ["bicubic", "sr_sparse", "sr_edge"]


def _safe_load_gray(path):
    arr = img_as_float32(io.imread(path))
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr


def evaluate(scale):
    test_files = list_image_files(HR_TEST_DIR)
    # per-method dict of tuples
    all_results = {m: [] for m in METHODS}
    for f in tqdm(test_files, desc=f"Evaluating x{scale}"):
        base = os.path.splitext(os.path.basename(f))[0]
        gt_path = os.path.join(OUTPUT_DIR, f"{base}_hr_gt.png")
        if not os.path.exists(gt_path):
            continue
        gt = _safe_load_gray(gt_path)
        for m in METHODS:
            sr_path = os.path.join(OUTPUT_DIR, f"{base}_{m}_x{scale}.png")
            if not os.path.exists(sr_path):
                continue
            sr = _safe_load_gray(sr_path)
            h = min(sr.shape[0], gt.shape[0])
            w = min(sr.shape[1], gt.shape[1])
            psnr, ssim = compute_psnr_ssim(sr[:h, :w], gt[:h, :w])
            all_results[m].append((base, psnr, ssim))

    for m in METHODS:
        results = all_results[m]
        if len(results) == 0:
            print(f"[x{scale}] {m}: no results found.")
            continue
        avg_psnr = np.mean([r[1] for r in results])
        avg_ssim = np.mean([r[2] for r in results])
        print(f"[x{scale}] {m}: Images {len(results)}  PSNR {avg_psnr:.3f}  SSIM {avg_ssim:.4f}")
        csv_path = os.path.join(OUTPUT_DIR, f"metrics_{m}_x{scale}.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image', 'psnr', 'ssim'])
            for r in results:
                writer.writerow([r[0], r[1], r[2]])
        print("  saved ->", csv_path)


if __name__ == "__main__":
    for s in SCALES:
        evaluate(s)
