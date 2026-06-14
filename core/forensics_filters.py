"""
core/forensics_filters.py

BayarConv + SRM (Spatial Rich Model) filter bank + MoireDetector.
Novel: first application of these forensics filters to exam proctoring.

BayarConv: Bayar & Stamm, IHMS 2016
SRM: Fridrich & Kodovsky, IEEE TIFS 2012
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class BayarConv2d(nn.Module):
    """
    Constrained convolutional layer.
    Center weight forced to -1, surrounding weights normalised to sum to +1.
    Suppresses image content, amplifies manipulation traces.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.center = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.clone()
        w[:, :, self.center, self.center] = 0
        w_sum = w.sum(dim=(2, 3), keepdim=True)
        w_sum = torch.where(w_sum.abs() < 1e-8, torch.ones_like(w_sum), w_sum)
        w = w / w_sum
        w[:, :, self.center, self.center] = -1.0
        return F.conv2d(x, w, padding=self.center)


class SRMFilterBank(nn.Module):
    """
    30 deterministic high-pass SRM filters (5x5).
    Fixed weights — not learnable.
    Detects GAN artifacts, JPEG boundaries, splicing noise residuals.
    """
    def __init__(self):
        super().__init__()
        filters = self._build_srm_filters()
        self.register_buffer('srm_weight', filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        channels = []
        for c in range(C):
            ch = x[:, c:c+1, :, :]
            filtered = F.conv2d(ch, self.srm_weight, padding=2)
            channels.append(filtered)
        return torch.cat(channels, dim=1)  # (B, 90, H, W)

    def _build_srm_filters(self) -> torch.Tensor:
        filters = []

        f1 = np.array([
            [-1, 2,-2, 2,-1],
            [ 2,-6, 8,-6, 2],
            [-2, 8,-12,8,-2],
            [ 2,-6, 8,-6, 2],
            [-1, 2,-2, 2,-1]
        ], dtype=np.float32) / 12.0
        filters.append(f1)

        for kernel in [
            np.array([[0,0,0],[0,1,-2],[0,1,0]], dtype=np.float32) / 2.0,
            np.array([[0,1,0],[0,-2,0],[0,1,0]], dtype=np.float32) / 2.0,
            np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.float32) / 8.0,
            np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], dtype=np.float32) / 4.0,
            np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32) / 4.0,
            np.array([[0,0,0],[0,1,-1],[0,0,0]], dtype=np.float32),
            np.array([[0,0,0],[0,1,0],[0,-1,0]], dtype=np.float32),
            np.array([[-1,0,0],[0,1,0],[0,0,0]], dtype=np.float32),
            np.array([[0,0,-1],[0,1,0],[0,0,0]], dtype=np.float32),
        ]:
            f = np.zeros((5, 5), dtype=np.float32)
            f[1:4, 1:4] = kernel
            filters.append(f)

        f_wide = np.array([
            [ 1,-4, 6,-4, 1],
            [-4,16,-24,16,-4],
            [ 6,-24,36,-24, 6],
            [-4,16,-24,16,-4],
            [ 1,-4, 6,-4, 1]
        ], dtype=np.float32) / 36.0
        filters.append(f_wide)

        # Pad to 30 with rotations
        base = len(filters)
        for i in range(30 - base):
            src = filters[i % base]
            filters.append(np.rot90(src, k=(i % 4) + 1).copy())

        stacked = np.stack(filters[:30], axis=0)[:, np.newaxis, :, :]
        return torch.tensor(stacked, dtype=torch.float32)


class MoireDetector(nn.Module):
    """
    Detects moiré patterns via 2D FFT analysis.
    Moiré appears when a student photographs the exam screen —
    interference between display pixels and camera sensor grid.
    Novel: no proctoring system has ever detected this.
    """
    def __init__(self, img_size: int = 256):
        super().__init__()
        self.img_size = img_size
        self.freq_attention = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        gray = gray.unsqueeze(1)

        fft = torch.fft.fft2(gray.float())
        fft_shifted = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shifted)
        phase = torch.angle(fft_shifted)
        log_mag = torch.log1p(magnitude)
        log_mag = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)

        freq_features = torch.cat([
            log_mag,
            (phase + np.pi) / (2 * np.pi)
        ], dim=1)

        attended = self.freq_attention(freq_features) * freq_features
        return F.avg_pool2d(attended, kernel_size=8)


class ForensicsFeatureExtractor(nn.Module):
    """Fuses BayarConv + SRM residuals + MoireDetector frequency features."""

    def __init__(self, img_size: int = 256):
        super().__init__()
        self.bayar = BayarConv2d(in_channels=3, out_channels=3)
        self.srm   = SRMFilterBank()
        self.moire = MoireDetector(img_size=img_size)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(93, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bayar_out = self.bayar(x)
        srm_out   = self.srm(x)
        moire_out = self.moire(x)
        combined  = torch.cat([bayar_out, srm_out], dim=1)
        forensics = self.fusion_conv(combined)
        return forensics, moire_out


if __name__ == "__main__":
    print("Testing forensics filters...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    extractor = ForensicsFeatureExtractor(img_size=256).to(device)
    dummy = torch.randn(2, 3, 256, 256).to(device)
    forensics, moire = extractor(dummy)
    print(f"BayarConv+SRM output : {forensics.shape}")
    print(f"Moiré output         : {moire.shape}")
    n = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    print(f"Trainable params     : {n:,}")
    print("✓ Forensics filters OK")
