"""
Train edge-preserving K-SVD dictionaries for a given scale.
Workflow:
 - Read HR images
 - Create LR by bicubic downsampling
 - Compute HF = HR - blur(HR)
 - Extract LR patches and corresponding HF patches
 - Keep edge-rich patches only (based on gradient magnitude)
 - Train D_L (on LR patch vectors) and D_H (on HF patch vectors mapped by same coefficients)
 - Save D_L.npy and D_H.npy
"""

import os
import numpy as np
from joblib import dump
from tqdm import tqdm
from sklearn.utils import shuffle

from config import *
from utils import *
from ksvd import KSVDDictLearner

import argparse

def _patch_limit_reached(collected, max_patches):
    if max_patches is None or max_patches <= 0:
        return False
    return collected >= max_patches


def build_training_patches(hr_files, scale, max_patches=MAX_PATCHES):
    lr_patches_list = []
    hf_patches_list = []
    for f in tqdm(hr_files, desc="Images"):
        hr = read_image_gray(f)
        # make sure dimensions are divisible by scale (crop)
        h, w = hr.shape
        hr = hr[: (h // scale) * scale, : (w // scale) * scale]
        lr = downsample(hr, scale)
        # compute HF on HR
        hf = high_frequency(hr)
        # choose patch sizes
        p_lr = LR_PATCH_SIZE
        p_hr = LR_PATCH_SIZE * scale + HR_PAD  # HR patch slightly larger than scaled LR patch
        # sliding windows on LR
        for i in range(0, lr.shape[0] - p_lr + 1, PATCH_STEP):
            for j in range(0, lr.shape[1] - p_lr + 1, PATCH_STEP):
                lr_patch = lr[i:i+p_lr, j:j+p_lr]
                # corresponding top-left in HR
                hi = i * scale
                hj = j * scale
                hr_patch = hf[hi:hi+p_hr, hj:hj+p_hr]
                if hr_patch.shape != (p_hr, p_hr):
                    continue
                # edge filter on LR patch
                mag = gradient_magnitude(lr_patch)
                if mag.mean() < EDGE_THRESHOLD:
                    continue
                lr_patches_list.append(lr_patch)
                hf_patches_list.append(hr_patch)
                if _patch_limit_reached(len(lr_patches_list), max_patches):
                    break
            if _patch_limit_reached(len(lr_patches_list), max_patches):
                break
        if _patch_limit_reached(len(lr_patches_list), max_patches):
            break
    if len(lr_patches_list) == 0:
        raise RuntimeError("No patches extracted - check EDGE_THRESHOLD or dataset.")
    lr_arr = np.array(lr_patches_list)
    hf_arr = np.array(hf_patches_list)
    return lr_arr, hf_arr

def train_for_scale(scale):
    makedirs(OUTPUT_DIR)
    hr_files = list_image_files(HR_TRAIN_DIR)
    print(f"Found {len(hr_files)} HR train images.")
    lr_patches, hf_patches = build_training_patches(hr_files, scale)
    print("Patch shapes:", lr_patches.shape, hf_patches.shape)
    # vectorize
    X_lr = patches_to_vectors(lr_patches)   # (N, p*p)
    X_hf = patches_to_vectors(hf_patches)   # (N, p_hr*p_hr)
    # normalize mean 0
    X_lr = X_lr - np.mean(X_lr, axis=1, keepdims=True)
    X_hf = X_hf - np.mean(X_hf, axis=1, keepdims=True)

    # shuffle and maybe sub-sample for speed
    X_lr, X_hf = shuffle(X_lr, X_hf, random_state=RANDOM_SEED)
    n_samples = X_lr.shape[0]
    print(f"Training on {n_samples} patch pairs.")
    # Train K-SVD on LR patches
    ksvd_lr = KSVDDictLearner(n_atoms=DICT_ATOMS, sparsity=SPARSITY, n_iter=KSVD_ITERS, random_state=RANDOM_SEED, n_jobs=N_JOBS)
    ksvd_lr.fit(X_lr)
    D_L = ksvd_lr.D  # shape (n_features, n_atoms)
    # compute sparse codes for all LR patches (Gamma)
    Gamma = ksvd_lr._omp(D_L, X_lr)  # (n_atoms, n_samples)
    # Now learn D_H by solving min ||X_hf^T - D_H * Gamma||_F
    # X_hf.T shape (p_hr*p_hr, n_samples)
    XhfT = X_hf.T
    # Solve D_H = XhfT * pinv(Gamma)
    # But Gamma may be large, use least squares:
    # D_H * Gamma = XhfT  => for each row dimension we can solve
    # Use numpy lstsq for numerical stability
    print("Solving for D_H via least squares...")
    D_H, _, _, _ = np.linalg.lstsq(Gamma.T, XhfT.T, rcond=None)
    D_H = D_H.T  # shape (n_features_hr, n_atoms)
    # normalize columns of D_H
    D_H = D_H / (np.linalg.norm(D_H, axis=0, keepdims=True) + 1e-12)

    # Save dictionaries and metadata
    np.save(os.path.join(OUTPUT_DIR, f"D_L_x{scale}.npy"), D_L)
    np.save(os.path.join(OUTPUT_DIR, f"D_H_x{scale}.npy"), D_H)
    print(f"Dictionaries saved to {OUTPUT_DIR}")
    return D_L, D_H

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=2, help="scale factor to train for")
    args = parser.parse_args()
    train_for_scale(args.scale)
