"""PyTorch Dataset for chess board positions.

Loads boards.npy + moves.npy (required) and optionally values.npy +
aggression.npy for the full training pipeline.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from board_encoding import BOARD_CHANNELS


class ChessDataset(Dataset):
    def __init__(self, boards_path, moves_path,
                 values_path=None, aggression_path=None):
        self.boards = np.load(boards_path, mmap_mode='r')
        self.moves = np.load(moves_path, mmap_mode='r')
        assert len(self.boards) == len(self.moves), (
            f'boards ({len(self.boards)}) and moves ({len(self.moves)}) '
            f'length mismatch')

        # Optional value targets
        if values_path and os.path.exists(values_path):
            self.values = np.load(values_path, mmap_mode='r')
            assert len(self.values) == len(self.boards)
        else:
            self.values = None

        # Optional aggression scores
        if aggression_path and os.path.exists(aggression_path):
            self.aggression = np.load(aggression_path, mmap_mode='r')
            assert len(self.aggression) == len(self.boards)
        else:
            self.aggression = None

    def __len__(self):
        return len(self.boards)

    @property
    def has_values(self):
        return self.values is not None

    @property
    def has_aggression(self):
        return self.aggression is not None

    def __getitem__(self, idx):
        raw = np.array(self.boards[idx]).reshape(BOARD_CHANNELS, 8, 8)
        x = torch.from_numpy(raw.astype(np.float32))
        # Rescale channel 18 (half-move clock) from uint8 0-255 → float 0-1
        if raw.dtype == np.uint8:
            x[18] = x[18] / 255.0
        y = torch.tensor(int(self.moves[idx]), dtype=torch.long)

        v = torch.tensor(0.0)
        if self.values is not None:
            v = torch.tensor(float(self.values[idx]), dtype=torch.float32)

        a = torch.tensor(0.0)
        if self.aggression is not None:
            a = torch.tensor(float(self.aggression[idx]), dtype=torch.float32)

        return x, y, v, a