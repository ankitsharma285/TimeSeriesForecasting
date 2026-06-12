import torch
import torch.nn as nn

class NaivePersistence(nn.Module):
    """
    Baseline 1: Last-Value Repeat (Persistence)
    Takes the absolute last known value in the sequence and repeats it 
    for all future prediction steps.
    """
    def __init__(self, seq_len, pred_len, enc_in):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in

    def forward(self, x):
        # x shape: [Batch, Seq_len, Channel]
        # Pull the last time step: [Batch, 1, Channel]
        last_step = x[:, -1:, :] 
        
        # Repeat it along the time dimension for pred_len steps
        # Output shape: [Batch, Pred_len, Channel]
        out = last_step.repeat(1, self.pred_len, 1)
        return out


class RepeatLastWindow(nn.Module):
    """
    Baseline 2: Window Replication
    Takes the last segment of history matching the length of the prediction horizon
    and projects it directly forward. (If pred_len > seq_len, it tiles it).
    """
    def __init__(self, seq_len, pred_len, enc_in):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in

    def forward(self, x):
        # x shape: [Batch, Seq_len, Channel]
        # Take the trailing slice of length self.pred_len
        if self.seq_len >= self.pred_len:
            historical_window = x[:, -self.pred_len:, :]
            return historical_window
        else:
            # If historical lookback is too short, tile it to match pred_len
            historical_window = x[:, -self.seq_len:, :]
            repeats = (self.pred_len // self.seq_len) + 1
            out = historical_window.repeat(1, repeats, 1)
            return out[:, :self.pred_len, :]