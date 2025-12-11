"""
Inference script:
 - load trained dictionaries D_L, D_H for a chosen scale
 - for each test LR image:
    - extract LR patches
    - compute sparse codes (OMP) possibly with edge-weighted penalty (we implement simple weighting by scaling dictionary columns)
    - reconstruct HR patches
    - aggregate to HR image
    - optional back-projection refinement
 - save results
"""

import os
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import orthogonal_mp
from joblib import load
from config import *
from utils import *

import argparse

# ---------------------------------------------------------------------------#
# Dictionary helpers
# ---------------------------------------------------------------------------#
def load_dicts(scale):
    D_L = np.load(os.path.join(OUTPUT_DIR, f"D_L_x{scale}.npy"))
    D_H = np.load(os.path.join(OUTPUT_DIR, f"D_H_x{scale}.npy"))
    return D_L, D_H

def weighted_omp(D, y, n_nonzero_coefs=6, weights=None):
    """
    Basic approach to edge-weighted sparse coding:
    If weights is provided (per-atom penalties), we can re-scale atoms:
       D' = D / weights
       Solve for alpha' with OMP, then alpha = alpha' / weights
    weights: array shape (n_atoms,) > 0 ; smaller weight => less penalty (encourage)
    """
    if weights is None:
        coef = orthogonal_mp(D, y.reshape(1, -1).T, n_nonzero_coefs=n_nonzero_coefs)
        return coef.ravel()
    w = np.array(weights)
    D_scaled = D / (w[None, :] + 1e-12)
    coef_scaled = orthogonal_mp(D_scaled, y.reshape(1, -1).T, n_nonzero_coefs=n_nonzero_coefs)
    coef = (coef_scaled.ravel()) / (w + 1e-12)
    return coef

def back_projection(hr, lr, scale, iterations=8):
    """Classic iterative back-projection to enforce LR consistency"""
    hr_cp = hr.copy()
    for i in range(iterations):
        down = downsample(hr_cp, scale)
        err = lr - down
        # upsample error and add
        up_err = upsample(err, scale)
        hr_cp = hr_cp + up_err
    return hr_cp

def infer_single_image(lr_img, D_L, D_H, scale, use_edge_weights=True):
    """
    Reconstruct HR image via sparse coding. If use_edge_weights=False, this is the
    plain sparse SR baseline; if True, applies the simple edge-weighting heuristic.
    """
    p_lr = LR_PATCH_SIZE
    p_hr = LR_PATCH_SIZE * scale + HR_PAD
    h_lr, w_lr = lr_img.shape
    hr_patches = []
    for i in range(0, h_lr - p_lr + 1, PATCH_STEP):
        for j in range(0, w_lr - p_lr + 1, PATCH_STEP):
            lp = lr_img[i:i+p_lr, j:j+p_lr]
            vec = lp.reshape(1, -1)
            vec = vec - np.mean(vec)  # zero-mean
            # edge-aware weights (disabled when use_edge_weights is False)
            mag = gradient_magnitude(lp)
            edge_strength = mag.mean()
            if use_edge_weights and edge_strength > 0.0:
                w = np.ones(D_L.shape[1]) * (1.0 / (1.0 + edge_strength))
            else:
                w = None
            alpha = weighted_omp(D_L, vec, n_nonzero_coefs=SPARSITY, weights=w)
            hr_vec = D_H.dot(alpha)
            hr_patch = hr_vec.reshape(p_hr, p_hr)
            # add local mean to stabilize brightness
            mean_lr_up = np.mean(upsample(lp, scale))
            hr_patch = hr_patch + mean_lr_up
            hr_patches.append(hr_patch)

    target_shape = (h_lr * scale, w_lr * scale)
    recon = np.zeros(target_shape, dtype=np.float32)
    weight = np.zeros(target_shape, dtype=np.float32)
    idx = 0
    for i in range(0, h_lr - p_lr + 1, PATCH_STEP):
        for j in range(0, w_lr - p_lr + 1, PATCH_STEP):
            hi = i * scale
            hj = j * scale
            patch = hr_patches[idx]
            ph, pw = patch.shape
            if hi + ph <= target_shape[0] and hj + pw <= target_shape[1]:
                recon[hi:hi+ph, hj:hj+pw] += patch
                weight[hi:hi+ph, hj:hj+pw] += 1.0
            else:
                ph_eff = min(ph, target_shape[0] - hi)
                pw_eff = min(pw, target_shape[1] - hj)
                recon[hi:hi+ph_eff, hj:hj+pw_eff] += patch[:ph_eff, :pw_eff]
                weight[hi:hi+ph_eff, hj:hj+pw_eff] += 1.0
            idx += 1
    mask = weight > 0
    recon[mask] /= weight[mask]
    # fill uncovered borders with bicubic upsample
    bic = upsample(lr_img, scale)
    recon[~mask] = bic[~mask]
    recon = back_projection(recon, lr_img, scale, iterations=BACKPROJECTION_ITERS)
    recon = np.clip(recon, 0.0, 1.0)
    return recon


def process_test_folder(scale):
    """
    For each test HR image, we:
      - generate LR by downsampling
      - save LR
      - bicubic baseline (upsample LR)
      - sparse SR (no edge weighting)
      - edge-aware SR (current heuristic)
      - save HR GT
    """
    makedirs(OUTPUT_DIR)
    D_L, D_H = load_dicts(scale)
    test_files = list_image_files(HR_TEST_DIR)
    for f in tqdm(test_files, desc="Testing images"):
        hr_gt = read_image_gray(f)
        lr = downsample(hr_gt, scale)

        # Reconstructions
        bicubic = upsample(lr, scale)
        sr_sparse = infer_single_image(lr, D_L, D_H, scale, use_edge_weights=False)
        sr_edge = infer_single_image(lr, D_L, D_H, scale, use_edge_weights=True)

        base = os.path.splitext(os.path.basename(f))[0]
        save_image(os.path.join(OUTPUT_DIR, f"{base}_lr_x{scale}.png"), lr)
        save_image(os.path.join(OUTPUT_DIR, f"{base}_bicubic_x{scale}.png"), bicubic)
        save_image(os.path.join(OUTPUT_DIR, f"{base}_sr_sparse_x{scale}.png"), sr_sparse)
        save_image(os.path.join(OUTPUT_DIR, f"{base}_sr_edge_x{scale}.png"), sr_edge)
        save_image(os.path.join(OUTPUT_DIR, f"{base}_hr_gt.png"), hr_gt)
    print("Inference complete. Outputs saved to", OUTPUT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()
    process_test_folder(args.scale)
