import torch
import torch.nn as nn
import math 

class Model(nn.Module):
    """
    Optimized Linear Model for Time-Series Forecasting.
    Eliminates Python loops for channel-independence via batched operations.
    """
    def __init__(self, seq_len, pred_len, enc_in, individual):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual

        if self.individual:
            # Instead of a ModuleList loop, use raw parameters for an all-channel parallel matrix multiplication
            # Shape: [Channels, Input_len, Output_len]
            self.weight = nn.Parameter(torch.empty(self.channels, self.seq_len, self.pred_len))
            self.bias = nn.Parameter(torch.empty(self.channels, self.pred_len))
            self._reset_parameters()
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def _reset_parameters(self):
        """Standard Kaiming/Uniform initialization for custom parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # Input x shape: [Batch, Seq_len, Channel]
        
        if self.individual:
            # 1. Permute to [Channel, Batch, Seq_len] to align channels for batch matrix multiplication
            x = x.permute(2, 0, 1) 
            
            # 2. Perform parallel matrix multiplication across all channels at once
            # [C, B, S] x [C, S, P] -> [C, B, P]
            out = torch.bmm(x, self.weight) + self.bias.unsqueeze(1)
            
            # 3. Permute back to standard format: [Batch, Pred_len, Channel]
            return out.permute(1, 2, 0)
            
        else:
            # Single shared linear layer pathway
            # [Batch, Seq_len, Channel] -> [Batch, Channel, Seq_len]
            x = x.permute(0, 2, 1)
            # Apply Linear layer along the last dimension (Seq_len -> Pred_len)
            x = self.Linear(x)
            # [Batch, Channel, Pred_len] -> [Batch, Pred_len, Channel]
            return x.permute(0, 2, 1)