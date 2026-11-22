import torch
import torch.nn as nn
from mamba import Mamba, MambaConfig 

class TemporalPositionalEncoding(nn.Module):
    """1D temporal positional encoding: add a learnable position table to [B, L, C]."""
    def __init__(self, channels, length):
        super().__init__()
        self.register_buffer('position_ids', torch.arange(length))
        self.embedding = nn.Embedding(length, channels)
        self.length = length

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        pos_embed = self.embedding(self.position_ids).unsqueeze(0)  # [1, L, C]
        return x + pos_embed


class TemporalEncoderLayers(nn.Module):
    """Self-attention along the time dimension: tokens are time steps, feature dim is C."""
    def __init__(self, d_model, num_encoder_layers=3, dim_feedforward=384,
                 nhead=1, reverse=False, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, \
            f'd_model={d_model} 必须能被 nhead={nhead} 整除'
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation='gelu', norm_first=False,
            batch_first=True,          # key fix: keep the batch dimension first
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)
        self.reverse = reverse

    def forward(self, x):
        # x: [B, L, C]
        if self.reverse:
            x = torch.flip(x, dims=[1])
        x = self.encoder(x)
        if self.reverse:
            x = torch.flip(x, dims=[1])
        return x


class TransBlock(nn.Module):
    """Temporal TransBlock: BN1d -> positional encoding -> time-dimension Transformer."""
    def __init__(self, n_features, seq_len, num_encoder_layers=1, nhead=1):
        super().__init__()
        assert n_features % nhead == 0
        self.BN1 = nn.BatchNorm1d(n_features)
        self.pos1 = TemporalPositionalEncoding(n_features, seq_len)
        self.attn1 = TemporalEncoderLayers(
            d_model=n_features,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=n_features * 2,
            nhead=nhead,
        )

    def forward(self, x):
        # x: [B, L, C]
        x = x.permute(0, 2, 1)     # [B, C, L]
        x = self.BN1(x)            # batch norm applied over the C dimension
        x = x.permute(0, 2, 1)     # [B, L, C]
        x = self.pos1(x)
        x = self.attn1(x)
        return x

class MambaLayers(nn.Module):
    """DLRMT temporal backbone: Mamba (selective state space) scans along the time
    dimension L, with C as the feature dimension. With linear complexity and
    input-dependent selective state transitions, it retains long-range temporal
    memory without truncation. Input and output are both [B, L, C]."""
    def __init__(self, d_model, n_layers):
        super(MambaLayers, self).__init__()
        # use_cuda controls whether the official fused CUDA kernel is called;
        # when unavailable, fall back to the PyTorch implementation in mamba.py to avoid errors
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers, use_cuda=True))

    def forward(self, x):
        x = self.mamba(x)            # Mamba processes [B, L, C] along the time dimension
        return x

class BiMambaLayers(nn.Module):
    """Bidirectional Mamba (optional extension, not the default configuration of the
    main paper): forward + flipped backward, merged according to `merge`.
    The main paper uses unidirectional Mamba as the temporal backbone; bidirectionality
    here is only an ablation/enhancement comparison. Input/output are both [B, L, C],
    with Mamba scanning along time dimension L and C as the feature dimension.
    - forward branch: Mamba(x)
    - backward branch: flip(Mamba(flip(x, dim=1)), dim=1)
    """

    def __init__(self, d_model, n_layers, merge='sum', share_weights=False):
        super().__init__()
        assert merge in ('sum', 'concat', 'gate'), \
            f"merge 只支持 'sum' / 'concat' / 'gate'，收到 {merge}"
        self.merge = merge
        use_cuda = torch.cuda.is_available()

        cfg = MambaConfig(d_model=d_model, n_layers=n_layers, use_cuda=use_cuda)
        self.mamba_fwd = Mamba(cfg)

        if share_weights:
            # forward and backward share Mamba weights (fewer parameters, slightly weaker expressiveness)
            self.mamba_bwd = self.mamba_fwd
        else:
            # forward and backward use separate Mamba parameters
            self.mamba_bwd = Mamba(MambaConfig(
                d_model=d_model, n_layers=n_layers, use_cuda=use_cuda))

        # gating weights for merge='gate'
        if merge == 'gate':
            self.gate = nn.Linear(2 * d_model, 2 * d_model)
        # projection for merge='concat'
        if merge == 'concat':
            self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        # x: [B, L, C]
        y_fwd = self.mamba_fwd(x)                                     # forward
        y_bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])       # backward

        if self.merge == 'sum':
            return y_fwd + y_bwd
        elif self.merge == 'concat':
            return self.proj(torch.cat([y_fwd, y_bwd], dim=-1))
        else:  # gate
            g = torch.sigmoid(self.gate(torch.cat([y_fwd, y_bwd], dim=-1)))
            g_f, g_b = g.chunk(2, dim=-1)
            return g_f * y_fwd + g_b * y_bwd

class StandardCA(nn.Module):
    """Channel attention adapted for time series [B, L, C]."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        # No need for nn.AdaptiveAvgPool; x.mean(dim=1) is sufficient
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, L, C]
        b, l, c = x.size()
        x=x.permute(0,2,1)
        # 1. pool over the time dimension: [B, L, C] -> [B, C]
        y = x.mean(dim=1) 
        # 2. learn channel weights: [B, C] -> [B, C]
        y = self.fc(y)
        # 3. expand dims for broadcast multiplication: [B, C] -> [B, 1, C]
        y = y.unsqueeze(1)
        x=x*y
        x=x.permute(0,2,1)
        return x