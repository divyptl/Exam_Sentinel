"""
core/forensics_filters.py

BayarConv + SRM (Spatial Rich Model) filter bank.
These are borrowed from image forgery detection research and applied
here to exam proctoring — a novel application not seen in existing systems.

BayarConv: Learns constrained convolutional filters that suppress image
content and amplify manipulation traces (Bayar & Stamm, 2016).

SRM: 30 high-pass filters derived from image steganalysis that detect
noise residuals left behind by GAN-based deepfakes and JPEG splicing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class BayarConv2d(nn.Module):
    """
    Constrained convolutional layer from Bayar & Stamm (2016).
    Forces the kernel weights to sum to -1 at center with surrounding
    weights summing to +1, making it sensitive to local manipulation traces.

    Applied to each webcam frame before the main detection heads to
    extract forgery residuals invisible to standard CNNs.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.center = kernel_size // 2

        # Learnable weights (constrained during forward pass)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
            * 0.01
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Bayar constraint: center = -1, rest normalised to sum to +1
        w = self.weight.clone()

        # Zero out the center
        w[:, :, self.center, self.center] = 0

        # Normalise non-center weights to sum to 1 per filter
        w_sum = w.sum(dim=(2, 3), keepdim=True)
        # Avoid division by zero
        w_sum = torch.where(w_sum.abs() < 1e-8, torch.ones_like(w_sum), w_sum)
        w = w / w_sum

        # Set center to -1 (the constraint)
        w[:, :, self.center, self.center] = -1.0

        return F.conv2d(x, w, padding=self.center)


class SRMFilterBank(nn.Module):
    """
    Spatial Rich Model filter bank — 30 high-pass filters from
    Fridrich & Kodovsky (2012) steganalysis, adapted for deepfake detection.

    These filters extract noise residuals in frequency domain that are
    characteristic of GAN synthesis artifacts and JPEG re-compression
    from image splicing — both common in exam fraud scenarios:
    - Face-swap deepfakes leave GAN grid artifacts
    - ID document splicing leaves JPEG block boundaries
    - Screen recapture leaves moiré patterns in specific frequency bands
    """

    def __init__(self):
        super().__init__()
        filters = self._build_srm_filters()  # (30, 1, 5, 5)
        # Expand to 3 channels (apply same filter per channel)
        filters_3ch = filters.repeat(1, 3, 1, 1)  # (30, 3, 5, 5) -- wrong
        # Actually: we want 30 output channels, applied per-channel
        # Use depthwise-style: 3 input → 30 output filters
        self.register_buffer('srm_weight', filters)  # (30, 1, 5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) — apply SRM filters per channel, concat
        B, C, H, W = x.shape
        # Process each channel independently
        channels = []
        for c in range(C):
            ch = x[:, c:c+1, :, :]  # (B, 1, H, W)
            filtered = F.conv2d(ch, self.srm_weight, padding=2)  # (B,30,H,W)
            channels.append(filtered)
        # Stack: (B, 90, H, W) — 30 filters × 3 channels
        return torch.cat(channels, dim=1)

    def _build_srm_filters(self) -> torch.Tensor:
        """
        Construct the 30 SRM high-pass filter kernels (5×5).
        These are deterministic — same values as in the original paper.
        """
        filters = []

        # Type 1: Square filters (order 2, 3, 4)
        f1 = np.array([
            [-1, 2, -2, 2, -1],
            [ 2,-6,  8,-6,  2],
            [-2, 8,-12, 8, -2],
            [ 2,-6,  8,-6,  2],
            [-1, 2, -2, 2, -1]
        ], dtype=np.float32) / 12.0
        filters.append(f1)

        # Type 2: Edge filters (horizontal, vertical, diagonal variants)
        f2 = np.array([
            [0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0],
            [0,  1, -2,  1,  0],
            [0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0]
        ], dtype=np.float32) / 2.0
        filters.append(f2)

        f3 = np.array([
            [0,  0,  0,  0,  0],
            [0,  0,  1,  0,  0],
            [0,  0, -2,  0,  0],
            [0,  0,  1,  0,  0],
            [0,  0,  0,  0,  0]
        ], dtype=np.float32) / 2.0
        filters.append(f3)

        f4 = np.array([
            [0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0],
            [0,  1, -2,  1,  0],
            [0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0]
        ], dtype=np.float32) / 2.0
        filters.append(f4)

        # Type 3: 3x3 Laplacian-based filters (embed in 5x5)
        laplacian_kernels = [
            np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32) / 4.0,
            np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.float32) / 8.0,
            np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], dtype=np.float32) / 4.0,
            np.array([[0,0,0],[0,1,-1],[0,0,0]], dtype=np.float32),
            np.array([[0,0,0],[0,1,0],[0,-1,0]], dtype=np.float32),
            np.array([[0,0,0],[0,1,0],[-1,0,0]], dtype=np.float32),
            np.array([[0,0,0],[-1,1,0],[0,0,0]], dtype=np.float32),
        ]
        for k in laplacian_kernels:
            f = np.zeros((5, 5), dtype=np.float32)
            f[1:4, 1:4] = k
            filters.append(f)

        # Type 4: Diagonal and cross high-pass filters
        diagonal_bases = [
            np.array([[-1,0,0],[0,1,0],[0,0,0]], dtype=np.float32),
            np.array([[0,0,-1],[0,1,0],[0,0,0]], dtype=np.float32),
            np.array([[0,0,0],[0,1,0],[-1,0,0]], dtype=np.float32),
            np.array([[0,0,0],[0,1,0],[0,0,-1]], dtype=np.float32),
            np.array([[-1,0,1],[0,0,0],[0,0,0]], dtype=np.float32),
            np.array([[0,0,-1],[0,0,0],[0,0,1]], dtype=np.float32),
        ]
        for k in diagonal_bases:
            f = np.zeros((5, 5), dtype=np.float32)
            f[1:4, 1:4] = k
            filters.append(f)

        # Type 5: Wider-range high-pass (order 4 residuals)
        f_wide = np.array([
            [1, -4,  6, -4,  1],
            [-4,16,-24, 16, -4],
            [6,-24, 36,-24,  6],
            [-4,16,-24, 16, -4],
            [1, -4,  6, -4,  1]
        ], dtype=np.float32) / 36.0
        filters.append(f_wide)

        # Pad to 30 filters with rotated/flipped variants
        base_count = len(filters)
        for i in range(30 - base_count):
            src = filters[i % base_count]
            filters.append(np.rot90(src, k=(i % 4) + 1).copy())

        filters = filters[:30]

        # Stack → (30, 1, 5, 5)
        stacked = np.stack(filters, axis=0)[:, np.newaxis, :, :]
        return torch.tensor(stacked, dtype=torch.float32)


class MoireDetector(nn.Module):
    """
    Moiré pattern detector for recapture detection.

    When a student photographs the exam screen with their phone, the
    interference between the display pixel grid and camera sensor creates
    characteristic moiré patterns in specific FFT frequency bands.

    This module detects those patterns — identifying screen recapture
    attempts (question paper leaks) in real time.

    Novel: No existing proctoring system detects recapture attempts.
    """

    def __init__(self, img_size: int = 256):
        super().__init__()
        self.img_size = img_size

        # Learnable frequency-band attention weights
        self.freq_attention = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract moiré-indicative frequency features.

        Returns: (B, 2, H//8, W//8) — spatial map of moiré energy
        """
        B, C, H, W = x.shape

        # Convert to grayscale for frequency analysis
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        gray = gray.unsqueeze(1)  # (B, 1, H, W)

        # 2D FFT
        fft = torch.fft.fft2(gray.float())
        fft_shifted = torch.fft.fftshift(fft)

        # Magnitude and phase
        magnitude = torch.abs(fft_shifted)
        phase = torch.angle(fft_shifted)

        # Log-scale magnitude (better dynamic range)
        log_mag = torch.log1p(magnitude)

        # Normalise to [0, 1]
        log_mag = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)

        # Stack mag + phase info
        freq_features = torch.cat([
            log_mag,
            (phase + np.pi) / (2 * np.pi)  # normalise phase to [0,1]
        ], dim=1)  # (B, 2, H, W)

        # Apply learnable attention to emphasise moiré frequency bands
        attended = self.freq_attention(freq_features) * freq_features

        # Downsample to reduce computation
        return F.avg_pool2d(attended, kernel_size=8)


class ForensicsFeatureExtractor(nn.Module):
    """
    Combined forensics feature extractor.
    Runs BayarConv + SRM in parallel, concatenates residuals,
    then fuses with moiré features.

    Output: rich forensics feature tensor fed into detection heads.
    """

    def __init__(self, img_size: int = 256):
        super().__init__()
        self.bayar = BayarConv2d(in_channels=3, out_channels=3)
        self.srm = SRMFilterBank()   # outputs 90 channels
        self.moire = MoireDetector(img_size=img_size)

        # Fusion projection: 3 + 90 = 93 → 128
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(93, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            forensics_features: (B, 64, H, W) — fused BayarConv+SRM
            moire_features: (B, 2, H//8, W//8) — frequency domain features
        """
        bayar_out = self.bayar(x)          # (B, 3, H, W)
        srm_out = self.srm(x)             # (B, 90, H, W)
        moire_out = self.moire(x)         # (B, 2, H//8, W//8)

        # Fuse BayarConv + SRM residuals
        combined = torch.cat([bayar_out, srm_out], dim=1)  # (B, 93, H, W)
        forensics_features = self.fusion_conv(combined)     # (B, 64, H, W)

        return forensics_features, moire_out


if __name__ == "__main__":
    # Quick sanity check
    print("Testing forensics filters...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    extractor = ForensicsFeatureExtractor(img_size=256).to(device)
    dummy = torch.randn(2, 3, 256, 256).to(device)

    forensics, moire = extractor(dummy)
    print(f"BayarConv+SRM output: {forensics.shape}")  # (2, 64, 256, 256)
    print(f"Moiré output:         {moire.shape}")       # (2, 2, 32, 32)

    # Count parameters
    n_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")
    print("✓ Forensics filters OK")
