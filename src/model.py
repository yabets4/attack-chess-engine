"""ChessNet — SE-ResNet with dual policy + value heads.

Architecture (Leela Chess Zero inspired):
  Input  (BOARD_CHANNELS × 8 × 8)
    → Conv 3×3 → BN → Mish                  (→ filters)
    → SE-ResBlock × num_res                  (residual + squeeze-excitation)
    ├→ Policy Head  (2 × Conv1×1-BN-Mish → FC hidden → FC 4096)
    └→ Value Head   (2 × Conv1×1-BN-Mish → FC → FC → FC → Tanh)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from board_encoding import BOARD_CHANNELS, MOVE_CHANNELS


# ---------- Mish activation ----------

class Mish(nn.Module):
    """Mish: self-regularised non-monotonic activation.
    Smoother gradients than ReLU, used by YOLOv4 / Leela.
    """
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# ---------- Squeeze-and-Excitation ----------

class SEBlock(nn.Module):
    """Channel attention via global-average-pool → FC → ReLU → FC → Sigmoid."""

    def __init__(self, channels, ratio=4):
        super().__init__()
        mid = max(channels // ratio, 16)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        w = x.mean(dim=(2, 3))          # (B, C)
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * w


# ---------- SE-ResBlock ----------

class SEResBlock(nn.Module):
    """Pre-activation residual block with SE channel attention."""

    def __init__(self, channels, se_ratio=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, ratio=se_ratio)
        self.act = Mish()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.act(out + x)


# ---------- Main network ----------

class ChessNet(nn.Module):
    """Full chess network with SE-ResNet backbone, policy head, value head.

    Args:
        num_res:  number of SE residual blocks  (default 15)
        filters:  channel width                 (default 256)
        se_ratio: SE squeeze ratio              (default 4)
        p_drop:   dropout in FC heads           (default 0.2)
    """

    def __init__(self, num_res=17, filters=256, se_ratio=4, p_drop=0.2):
        super().__init__()

        # --- Input stem ---
        self.conv_in = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            Mish(),
        )

        # --- Residual tower ---
        self.res_blocks = nn.Sequential(
            *[SEResBlock(filters, se_ratio) for _ in range(num_res)]
        )

        # --- Policy head (2 convs → hidden FC → output FC) ---
        self.policy_conv = nn.Sequential(
            nn.Conv2d(filters, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            Mish(),
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            Mish(),
        )
        policy_flat = 32 * 8 * 8  # 2048
        self.policy_fc = nn.Sequential(
            nn.Linear(policy_flat, 4096),
            Mish(),
            nn.Dropout(p_drop),
            nn.Linear(4096, MOVE_CHANNELS),
        )

        # --- Value head (2 convs → 3 FC layers → Tanh) ---
        self.value_conv = nn.Sequential(
            nn.Conv2d(filters, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            Mish(),
            nn.Conv2d(32, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            Mish(),
        )
        value_flat = 16 * 8 * 8  # 1024
        self.value_fc = nn.Sequential(
            nn.Linear(value_flat, 256),
            Mish(),
            nn.Dropout(p_drop),
            nn.Linear(256, 64),
            Mish(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Accept (B, H, W, C) layout and convert to (B, C, H, W)
        if x.shape[1] != BOARD_CHANNELS:
            x = x.permute(0, 3, 1, 2)

        h = self.conv_in(x)
        h = self.res_blocks(h)

        # Policy
        p = self.policy_conv(h).reshape(h.size(0), -1)
        p = self.policy_fc(p)

        # Value
        v = self.value_conv(h).reshape(h.size(0), -1)
        v = self.value_fc(v).squeeze(-1)

        return p, v