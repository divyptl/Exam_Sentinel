"""
scripts/download_weights.py

Downloads pretrained EfficientNet-B3 backbone weights via timm.
Sets up data directory structure.

Usage:
    python scripts/download_weights.py
"""
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def download_backbone():
    print("[1/3] Downloading EfficientNet-B3 backbone weights...")
    try:
        import timm
        model = timm.create_model('efficientnet_b3', pretrained=True)
        print("  ✓ EfficientNet-B3 weights downloaded and cached")
        del model
    except Exception as e:
        print(f"  ✗ Failed: {e}")

def setup_data_dirs():
    print("[2/3] Setting up data directories...")
    dirs = [
        "data/FaceForensics++/real/train", "data/FaceForensics++/real/val",
        "data/FaceForensics++/fake/train", "data/FaceForensics++/fake/val",
        "data/CASIA_v2/Au", "data/CASIA_v2/Tp",
        "data/recapture/genuine", "data/recapture/recaptured",
        "data/genuine", "data/demo", "models/weights", "logs",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("  ✓ Data directories created")

def create_demo_data():
    print("[3/3] Creating synthetic demo frames...")
    try:
        import numpy as np, cv2
        demo_dir = Path("data/demo")
        genuine = np.ones((256, 256, 3), dtype=np.uint8) * 180
        cv2.ellipse(genuine, (128, 100), (60, 75), 0, 0, 360, (220, 185, 155), -1)
        cv2.imwrite(str(demo_dir / "genuine_frame.jpg"), genuine)
        deepfake = genuine.copy()
        for i in range(0, 256, 8):
            deepfake[:, i] = np.clip(deepfake[:, i].astype(int) + 8, 0, 255)
        cv2.imwrite(str(demo_dir / "deepfake_frame.jpg"), deepfake)
        print("  ✓ Synthetic demo frames created in data/demo/")
    except ImportError:
        print("  ! opencv-python not installed")

if __name__ == "__main__":
    download_backbone()
    setup_data_dirs()
    create_demo_data()
    print("\nSetup complete! Run: python scripts/run_demo.py --mode dashboard")
