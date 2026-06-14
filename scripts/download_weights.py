"""scripts/download_weights.py — Download EfficientNet-B3 + create data dirs."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("[1/3] Downloading EfficientNet-B3 backbone...")
    try:
        import timm
        m = timm.create_model('efficientnet_b3', pretrained=True)
        print("  ✓ Weights cached via timm")
        del m
    except Exception as e:
        print(f"  ✗ {e} — run: pip install timm")

    print("[2/3] Creating data directories...")
    for d in ["data/FaceForensics++/real/train","data/FaceForensics++/real/val",
              "data/FaceForensics++/fake/train","data/FaceForensics++/fake/val",
              "data/CASIA_v2/Au","data/CASIA_v2/Tp",
              "data/recapture/genuine","data/recapture/recaptured",
              "data/genuine","data/demo","models/weights","logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("  ✓ Directories created")

    print("[3/3] Creating synthetic demo frames...")
    try:
        import numpy as np, cv2
        genuine = np.ones((256,256,3),dtype=np.uint8)*180
        cv2.ellipse(genuine,(128,100),(60,75),0,0,360,(220,185,155),-1)
        cv2.imwrite("data/demo/genuine_frame.jpg", genuine)
        fake = genuine.copy()
        for i in range(0,256,8):
            fake[:,i] = np.clip(fake[:,i].astype(int)+10,0,255)
        cv2.imwrite("data/demo/deepfake_frame.jpg", fake)
        print("  ✓ Demo frames created")
    except ImportError:
        print("  ! opencv-python not installed yet")

    print("\nSetup complete! Run: python scripts/run_demo.py --mode dashboard")

if __name__ == "__main__":
    main()
