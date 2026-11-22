import math
import torch
import torch.nn as nn
from block import *
from lowrank import DynamicRankLinear,LowRankMultiheadAttention,convert_mha_to_lowrank
from cfm import ChannelFusionModule

class DataEmbedding_inverted(nn.Module):
    """Inverted embedding: the full history sequence of each variable -> one token."""

    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, C] -> [B, C, L] -> [B, C, d_model]
        return self.dropout(self.value_embedding(x.permute(0, 2, 1)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1,
                 layer_index=0, base_rank=16, alpha=0.5,
                 lowrank_ffn=True):
        super().__init__()
        if lowrank_ffn:
            self.linear1 = DynamicRankLinear(
                d_model, d_ff, layer_index, base_rank, alpha)
            self.linear2 = DynamicRankLinear(
                d_ff, d_model, layer_index, base_rank, alpha)
        else:
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(
            self.dropout(self.activation(self.linear1(x)))
        )


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 layer_index=0, base_rank=16, alpha=0.5,
                 lowrank_attn=True, lowrank_ffn=True):
        super().__init__()

        if lowrank_attn:
            self.attention = LowRankMultiheadAttention(
                d_model, n_heads, dropout=dropout,
                layer_index=layer_index,
                base_rank=base_rank,
                alpha=alpha,
                lowrank_out=False,
            )
        else:
            self.attention = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True)

        self.ffn = FeedForward(
            d_model, d_ff, dropout,
            layer_index=layer_index,
            base_rank=base_rank,
            alpha=alpha,
            lowrank_ffn=lowrank_ffn,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class EncoderStack(nn.Module):
    def __init__(self, n_features, window_size, d_model, n_heads, d_ff,
                 e_layers=1, dropout=0.1,
                 base_rank=16, alpha=0.5,
                 lowrank_attn=True, lowrank_ffn=True):
        super().__init__()
        self.embed = DataEmbedding_inverted(window_size, d_model, dropout)
        self.pos = PositionalEncoding(d_model, max_len=n_features)
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model, n_heads, d_ff, dropout,
                layer_index=i,
                base_rank=base_rank,
                alpha=alpha,
                lowrank_attn=lowrank_attn,
                lowrank_ffn=lowrank_ffn,
            )
            for i in range(e_layers)
        ])

    def forward(self, x):
        x = self.pos(self.embed(x))
        for layer in self.layers:
            x = layer(x)
        return x


class RevIN(nn.Module):
    """Reversible instance normalization along the time dimension (dim=1)."""

    def __init__(self, num_features, eps=1e-5, affine=True, dim=1):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.dim = dim
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = x.mean(dim=self.dim, keepdim=True).detach()
            self.stdev = torch.sqrt(
                x.var(dim=self.dim, keepdim=True, unbiased=False) + self.eps
            ).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
        return x

class MyModel(nn.Module):
    def __init__(self, n_features, window_size, horizon,
                 d_model=128, n_heads=4, e_layers=1, d_ff=256,
                 dropout=0.1,
                 base_rank=32, alpha=0.5,
                 lowrank_attn=True, lowrank_ffn=True):
        super().__init__()
        self.revin = RevIN(n_features, dim=1)
        self.mamba0 = MambaLayers(d_model=n_features, n_layers=2)
        self.ca = StandardCA(window_size, reduction=8)
        self.encoder_layers = EncoderStack(
            n_features, window_size, d_model, n_heads, d_ff,
            e_layers, dropout,
            base_rank=base_rank,
            alpha=alpha,
            lowrank_attn=lowrank_attn,
            lowrank_ffn=lowrank_ffn,
        )
        self.projection = nn.Linear(d_model, horizon)
        self.cfm = ChannelFusionModule(patch_size=window_size,n_vars=n_features 
        ,d_model=window_size, dim=64,depth=2,heads=4,dropout=dropout)

    def forward(self, x):
        x = self.revin(x, 'norm')
        B, L, C = x.shape
        ### CCTM
        x_cfm = x.permute(0, 2, 1)
        x_cfm = torch.cat([x_cfm, x_cfm], dim=-1)  
        x_cfm_out, _, _ = self.cfm(x_cfm)
        x_cfm_out = x_cfm_out.view(B, C, L)
        x = x + x_cfm_out.permute(0, 2, 1)
        
        x = self.mamba0(x)
        x = self.ca(x)

        ### VLRT
        enc_out = self.encoder_layers(x)
        out = self.projection(enc_out).permute(0, 2, 1)
        return self.revin(out, 'denorm')


if __name__ == '__main__':
    from config import get_config

    cfg = get_config('BaffleCreek')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel(
        n_features=cfg['n_features'],
        window_size=cfg['window_size'],
        horizon=cfg['horizon'],
    ).to(device)

    x = torch.randn(4, cfg['window_size'], cfg['n_features']).to(device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (4, cfg['horizon'], cfg['n_features'])
    print(f'✅ Forward OK, output {tuple(y.shape)}, '
          f'params={sum(p.numel() for p in model.parameters()):,}')