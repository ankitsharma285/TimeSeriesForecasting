import numpy as np
import torch
from typing import Generator, Tuple, Dict, Any

class NoiseInjector:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def inject(self, x: np.ndarray, noise_type: str, severity: float) -> np.ndarray:
        x_corrupted = x.copy().astype(float)
        if noise_type == "none" or severity == 0.0:
            return x_corrupted
        
        # Calculate scale per column for realistic relative noise
        col_stds = np.std(x_corrupted, axis=0, keepdims=True) + 1e-5

        if noise_type == "gaussian":
            noise = self.rng.normal(0, severity * col_stds, size=x_corrupted.shape)
            x_corrupted += noise
        elif noise_type == "missing":
            mask = self.rng.random(size=x_corrupted.shape) < severity
            x_corrupted[mask] = 0.0  # Zero-fill dropouts
        elif noise_type == "spike":
            mask = self.rng.random(size=x_corrupted.shape) < (severity * 0.05)
            spikes = self.rng.choice([-5, 5], size=x_corrupted.shape) * col_stds
            x_corrupted[mask] += spikes[mask]
        elif noise_type == "drift":
            # Simulate a continuous covariate drift over time
            x_corrupted += severity * col_stds
        return x_corrupted

class StreamingSimulator:
    """Simulates a live production stream window-by-window."""
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def stream_windows(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        # Yield one evaluation window at a time sequentially
        total_len = len(self.data) - self.seq_len - self.pred_len + 1
        for i in range(total_len):
            x_window = self.data[i : i + self.seq_len]
            y_window = self.data[i + self.seq_len : i + self.seq_len + self.pred_len]
            yield x_window, y_window