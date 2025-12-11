import os
import cv2
import numpy as np
from skimage import io, color, img_as_float32
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

def makedirs(path):
    os.makedirs(path, exist_ok=True)

def read_image_gray(path):
    """Read an image and return grayscale float32 (0..1)."""
    im = io.imread(path)
    if im.ndim == 3:
        im = color.rgb2gray(im)
    return img_as_float32(im)

def save_image(path, img):
    """Save float image (0..1) to disk."""
    img_u8 = (np.clip(img, 0, 1) * 255.0).astype('uint8')
    cv2.imwrite(path, img_u8)

def gaussian_blur(img, ksize=7, sigma=1.5):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def high_frequency(img, ksize=7, sigma=1.5):
    """Return HF = img - blurred(img)"""
    blurred = gaussian_blur(img, ksize=ksize, sigma=sigma)
    hf = img - blurred
    return hf

def downsample(img, scale, anti_alias=True):
    h, w = img.shape
    new_h, new_w = h // scale, w // scale
    if anti_alias:
        # use cv2 INTER_AREA for downsampling
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def upsample(img, scale):
    h, w = img.shape
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

def gradient_magnitude(patch):
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return mag

def extract_patches(img, patch_size, step=1):
    """Return array of patches (N, patch_size, patch_size)."""
    h, w = img.shape
    patches = []
    for i in range(0, h - patch_size + 1, step):
        for j in range(0, w - patch_size + 1, step):
            p = img[i:i+patch_size, j:j+patch_size]
            patches.append(p.copy())
    return np.array(patches) if patches else np.empty((0, patch_size, patch_size))

def patches_to_vectors(patches):
    """(N, p, p) -> (p*p, N) or (N, p*p). We'll use (N, p*p)."""
    N, p, _ = patches.shape
    return patches.reshape(N, p * p)

def vectors_to_patches(vecs, p):
    N = vecs.shape[0]
    return vecs.reshape(N, p, p)

def aggregate_patches(patches, image_shape, patch_size, step=1):
    """Aggregate patches by averaging overlapping pixels.
    patches: (num_patches, p, p)
    """
    h, w = image_shape
    recon = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    idx = 0
    for i in range(0, h - patch_size + 1, step):
        for j in range(0, w - patch_size + 1, step):
            recon[i:i+patch_size, j:j+patch_size] += patches[idx]
            weight[i:i+patch_size, j:j+patch_size] += 1.0
            idx += 1
    # avoid division by zero
    mask = weight > 0
    recon[mask] /= weight[mask]
    return recon

def compute_psnr_ssim(pred, gt):
    psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim = structural_similarity(gt, pred, data_range=1.0)
    return psnr, ssim

def list_image_files(folder):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    return sorted(files)

