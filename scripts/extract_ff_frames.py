"""
scripts/extract_ff_frames.py

Extracts face crops from FaceForensics++ .mp4 video files.

Usage:
    python scripts/extract_ff_frames.py \
        --ff_root /path/to/FaceForensics++ \
        --output data/FaceForensics++ \
        --every_n 10
"""
import os, sys, argparse
from pathlib import Path
import cv2
from tqdm import tqdm

FF_SUBSETS = [
    "original_sequences/actors",
    "manipulated_sequences/Deepfakes",
    "manipulated_sequences/Face2Face",
    "manipulated_sequences/FaceSwap",
    "manipulated_sequences/NeuralTextures",
]

def crop_face(frame, target_size=256):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(64, 64))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        pad = int(0.2 * max(w, h))
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
        return cv2.resize(frame[y1:y2, x1:x2], (target_size, target_size))
    h, w = frame.shape[:2]
    size = min(h, w)
    return cv2.resize(frame[(h-size)//2:(h+size)//2, (w-size)//2:(w+size)//2],
                      (target_size, target_size))

def extract_frames(video_path, output_dir, every_n=10, max_frames=300):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    frame_idx, saved = 0, 0
    video_name = Path(video_path).stem
    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n == 0:
            face = crop_face(frame)
            cv2.imwrite(os.path.join(output_dir, f"{video_name}_{frame_idx:06d}.png"), face)
            saved += 1
        frame_idx += 1
    cap.release()
    return saved

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ff_root", required=True)
    parser.add_argument("--output", default="data/FaceForensics++")
    parser.add_argument("--every_n", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--split_ratio", type=float, default=0.85)
    args = parser.parse_args()

    ff_root = Path(args.ff_root)
    total = 0
    for subset in FF_SUBSETS:
        is_fake = "manipulated" in subset
        label = "fake" if is_fake else "real"
        subset_path = ff_root / subset / "c23" / "videos"
        if not subset_path.exists():
            continue
        videos = list(subset_path.glob("*.mp4"))
        n_train = int(len(videos) * args.split_ratio)
        print(f"\n{subset.split('/')[-1]} ({label}): {len(videos)} videos")
        for i, vp in enumerate(tqdm(videos)):
            split = "train" if i < n_train else "val"
            saved = extract_frames(str(vp), os.path.join(args.output, label, split),
                                   args.every_n, args.max_frames)
            total += saved
    print(f"\nTotal frames: {total:,}")

if __name__ == "__main__":
    main()
