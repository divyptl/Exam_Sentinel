"""
scripts/create_recapture_dataset.py

Creates the recapture dataset by simulating screen-photography attacks.
Novel contribution: no public exam-specific recapture dataset exists.

Usage:
    python scripts/create_recapture_dataset.py \
        --genuine_dir data/RAISE \
        --output_dir data/recapture \
        --n_images 2000
"""
import os, sys, argparse, random
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def apply_display_gamma(img, gamma=2.2):
    f = img.astype(np.float32) / 255.0
    return (np.power(f, 1.0/gamma) * 255).astype(np.uint8)

def add_pixel_grid(img, pixel_size=None, intensity=0.05):
    if pixel_size is None:
        pixel_size = random.randint(2, 5)
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)
    for i in range(0, h, pixel_size):
        mask[i, :] = 1.0 - intensity
    for j in range(0, w, pixel_size):
        mask[:, j] = 1.0 - intensity
    return np.clip(img.astype(np.float32) * mask[:, :, np.newaxis], 0, 255).astype(np.uint8)

def add_moire_pattern(img, freq1=None, freq2=None, angle=None, amplitude=None):
    h, w = img.shape[:2]
    freq1 = freq1 or random.uniform(0.08, 0.18)
    freq2 = freq2 or random.uniform(0.09, 0.20)
    angle = angle or random.uniform(-15, 15) * np.pi / 180
    amplitude = amplitude or random.uniform(8, 22)
    x, y = np.arange(w), np.arange(h)
    xx, yy = np.meshgrid(x, y)
    xx_rot = xx * np.cos(angle) - yy * np.sin(angle)
    yy_rot = xx * np.sin(angle) + yy * np.cos(angle)
    moire = np.sin(2*np.pi*freq1*xx_rot) * np.sin(2*np.pi*freq2*yy_rot) * amplitude
    return np.clip(img.astype(np.float32) + moire[:, :, np.newaxis], 0, 255).astype(np.uint8)

def add_camera_noise(img, noise_std=None, blur_kernel=None):
    if noise_std is None:
        noise_std = random.uniform(2.0, 8.0)
    if blur_kernel is None:
        blur_kernel = random.choice([0, 0, 0, 3, 3, 5])
    result = np.clip(img.astype(np.float32) + np.random.normal(0, noise_std, img.shape),
                     0, 255).astype(np.uint8)
    if blur_kernel > 0:
        result = cv2.GaussianBlur(result, (blur_kernel, blur_kernel), 0)
    return result

def simulate_recapture(img, quality=None):
    img = cv2.resize(img, (256, 256))
    img = apply_display_gamma(img)
    if random.random() > 0.3:
        img = add_pixel_grid(img, intensity=random.uniform(0.02, 0.08))
    img = add_moire_pattern(img)
    img = add_camera_noise(img)
    quality = quality or random.randint(70, 92)
    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def process_genuine(img):
    img = cv2.resize(img, (256, 256))
    quality = random.randint(85, 98)
    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genuine_dir", required=True)
    parser.add_argument("--output_dir", default="data/recapture")
    parser.add_argument("--n_images", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    genuine_dir = Path(args.genuine_dir)
    out_g = Path(args.output_dir) / "genuine"
    out_r = Path(args.output_dir) / "recaptured"
    out_g.mkdir(parents=True, exist_ok=True)
    out_r.mkdir(parents=True, exist_ok=True)

    all_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif"]:
        all_images.extend(genuine_dir.glob(ext))

    if not all_images:
        print(f"No images found in {genuine_dir}")
        sys.exit(1)

    random.shuffle(all_images)
    n = min(args.n_images, len(all_images))
    print(f"Generating {n} genuine + {n} recaptured pairs...")
    saved = 0
    for i, img_path in enumerate(tqdm(all_images[:n])):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        try:
            cv2.imwrite(str(out_g / f"genuine_{i:05d}.jpg"), process_genuine(img))
            cv2.imwrite(str(out_r / f"recaptured_{i:05d}.jpg"), simulate_recapture(img))
            saved += 1
        except Exception as e:
            print(f"  Error: {e}")
    print(f"Saved {saved} pairs → {args.output_dir}/")

if __name__ == "__main__":
    main()
