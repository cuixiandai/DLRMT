import math
import torch
import torch.nn as nn


class DynamicRankLinear(nn.Module):
    """The progressive variable-axis low-rank bottleneck of VLRT: W ~= U V.
    The rank is not a fixed hyperparameter but a function of depth:
    rank = min(max_rank, ceil(base_rank*(1+layer_index)^alpha)). Shallow layers compress
    toward shared low-frequency modes, while deeper layers relax to variable-specific
    details, separating denoising from detail preservation along depth."""
    def __init__(self, in_dim, out_dim, layer_index,
                 base_rank=16, alpha=0.5, bias=True):
        super().__init__()
        max_rank = min(in_dim, out_dim)
        self.rank = min(
            max_rank,
            math.ceil(base_rank * (1 + layer_index) ** alpha)
        )
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.U = nn.Parameter(torch.empty(out_dim, self.rank))
        self.V = nn.Parameter(torch.empty(self.rank, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        nn.init.normal_(self.U, std=1.0 / math.sqrt(self.rank))

    @classmethod
    def from_linear(cls, linear, layer_index, base_rank=16, alpha=0.5):
        out_dim, in_dim = linear.weight.shape
        obj = cls(
            in_dim, out_dim, layer_index,
            base_rank=base_rank, alpha=alpha,
            bias=linear.bias is not None
        )
        W = linear.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = obj.rank
        U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
        sqrt_s = S_r.sqrt()
        obj.U.data = (U_r * sqrt_s).to(linear.weight.dtype)
        obj.V.data = (sqrt_s[:, None] * Vh_r).to(linear.weight.dtype)
        if linear.bias is not None:
            obj.bias.data.copy_(linear.bias.data)
        return obj.to(linear.weight.device)

    def forward(self, x):
        z = torch.matmul(x, self.V.t())   # (..., rank)
        y = torch.matmul(z, self.U.t())   # (..., out_dim)
        if self.bias is not None:
            y = y + self.bias
        return y


class LowRankMultiheadAttention(nn.Module):
    """The variable-axis low-rank attention of VLRT: split Q/K/V into low-rank
    projections (DynamicRankLinear). Under inverted embedding (each variable's full
    history = 1 token), perform compressed attention along the variable axis."""
    def __init__(self, d_model, n_heads, dropout=0.1,
                 layer_index=0, base_rank=16, alpha=0.5,
                 lowrank_out=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.layer_index = layer_index
        self.base_rank = base_rank
        self.alpha = alpha

        self.q_proj = DynamicRankLinear(
            d_model, d_model, layer_index, base_rank, alpha)
        self.k_proj = DynamicRankLinear(
            d_model, d_model, layer_index, base_rank, alpha)
        self.v_proj = DynamicRankLinear(
            d_model, d_model, layer_index, base_rank, alpha)

        if lowrank_out:
            self.out_proj = DynamicRankLinear(
                d_model, d_model, layer_index, base_rank, alpha)
        else:
            self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value,
                attn_mask=None, key_padding_mask=None,
                need_weights=False, is_causal=False):
        B, Tq, D = query.shape
        Tk = key.shape[1]

        q = self.q_proj(query).view(
            B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(
            B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(
            B, Tk, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        if is_causal:
            causal = torch.ones(
                Tq, Tk, dtype=torch.bool, device=scores.device).tril()
            scores = scores.masked_fill(~causal, float('-inf'))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float('-inf'))
            else:
                scores = scores + attn_mask

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        out = self.out_proj(out)
        return out, None


@torch.no_grad()
def convert_mha_to_lowrank(mha, lowrank_mha):
    """Truncate trained nn.MultiheadAttention weights via SVD and load them into a low-rank MHA."""
    D = mha.embed_dim
    W = mha.in_proj_weight.data
    b = mha.in_proj_bias.data if mha.in_proj_bias is not None else None

    Wq, Wk, Wv = W.chunk(3, dim=0)
    if b is not None:
        bq, bk, bv = b.chunk(3, dim=0)
    else:
        bq = bk = bv = None

    for name, Wp, bp in [
        ('q_proj', Wq, bq),
        ('k_proj', Wk, bk),
        ('v_proj', Wv, bv),
    ]:
        tmp = nn.Linear(D, D, bias=bp is not None).to(W.device)
        tmp.weight.data.copy_(Wp)
        if bp is not None:
            tmp.bias.data.copy_(bp)
        new_proj = DynamicRankLinear.from_linear(
            tmp, lowrank_mha.layer_index,
            lowrank_mha.base_rank, lowrank_mha.alpha
        )
        setattr(lowrank_mha, name, new_proj)

    if isinstance(lowrank_mha.out_proj, nn.Linear):
        lowrank_mha.out_proj.weight.data.copy_(mha.out_proj.weight.data)
        if mha.out_proj.bias is not None:
            lowrank_mha.out_proj.bias.data.copy_(mha.out_proj.bias.data)
    else:
        lowrank_mha.out_proj = DynamicRankLinear.from_linear(
            mha.out_proj, lowrank_mha.layer_index,
            lowrank_mha.base_rank, lowrank_mha.alpha
        )