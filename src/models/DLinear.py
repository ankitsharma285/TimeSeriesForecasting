import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition Linear Model (DLinear) optimized for GPU performance.
    Eliminates ModuleList loop iterations for channel-independence via parallel batched matrix multiplications.
    """
    def __init__(self, seq_len, pred_len, enc_in, individual, moving_avg=25):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decomposition Kernel
        self.decompsition = series_decomp(moving_avg)
        self.channels = enc_in
        self.individual = individual
        
        if self.individual:
            # Vectorized channel weights: Shape [Channels, Seq_len, Pred_len]
            self.seasonal_weight = nn.Parameter(torch.empty(self.channels, self.seq_len, self.pred_len))
            self.seasonal_bias = nn.Parameter(torch.empty(self.channels, self.pred_len))
            
            self.trend_weight = nn.Parameter(torch.empty(self.channels, self.seq_len, self.pred_len))
            self.trend_bias = nn.Parameter(torch.empty(self.channels, self.pred_len))
            
            self._reset_parameters()
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def _reset_parameters(self):
        """Standard uniform weight initialization matching nn.Linear default allocations."""
        nn.init.kaiming_uniform_(self.seasonal_weight, a=math.sqrt(5))
        nn.init.zeros_(self.seasonal_bias)
        
        nn.init.kaiming_uniform_(self.trend_weight, a=math.sqrt(5))
        nn.init.zeros_(self.trend_bias)

    def forward(self, x):
        # Input shape x: [Batch, Seq_len, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        
        if self.individual:
            # 1. Align device contexts explicitly to prevent cross-device runtime bugs
            seasonal_w = self.seasonal_weight.to(x.device)
            seasonal_b = self.seasonal_bias.to(x.device)
            trend_w = self.trend_weight.to(x.device)
            trend_b = self.trend_bias.to(x.device)

            # 2. Permute from [B, S, C] to [C, B, S] to isolate individual channels as batch items
            seasonal_init = seasonal_init.permute(2, 0, 1)
            trend_init = trend_init.permute(2, 0, 1)
            
            # 3. Parallel Batched Matrix Multiplication across all channels instantly
            # [C, B, S] x [C, S, P] -> [C, B, P]
            seasonal_output = torch.bmm(seasonal_init, seasonal_w) + seasonal_b.unsqueeze(1)
            trend_output = torch.bmm(trend_init, trend_w) + trend_b.unsqueeze(1)
            
            # 4. Re-combine structural paths and permute back to [Batch, Pred_len, Channel]
            x = seasonal_output + trend_output
            return x.permute(1, 2, 0)
            
        else:
            # Shared parameter linear pathway
            seasonal_init = seasonal_init.permute(0, 2, 1)
            trend_init = trend_init.permute(0, 2, 1)
            
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            
            x = seasonal_output + trend_output
            return x.permute(0, 2, 1)