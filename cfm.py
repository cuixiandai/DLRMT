import torch
import torch.nn as nn
import math
from einops import rearrange
from torch.nn.functional import gumbel_softmax

# ==========================================
# 1. Dynamic contrastive loss (CCTM: drives variable adjacency structure learning)
# ==========================================
class DynamicalContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, k=0.3):
        super(DynamicalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.k = k

    def _stable_scores(self, scores):
        max_scores = torch.max(scores, dim=-1)[0].unsqueeze(-1)
        stable_scores = scores - max_scores
        return stable_scores

    def forward(self, scores, attn_mask, norm_matrix):
        b = scores.shape[0]
        n_vars = scores.shape[-1]

        cosine = (scores / norm_matrix).mean(1)
        pos_scores = torch.exp(cosine / self.temperature) * attn_mask
        all_scores = torch.exp(cosine / self.temperature)

        clustering_loss = -torch.log(pos_scores.sum(dim=-1) / all_scores.sum(dim=-1))

        eye = torch.eye(attn_mask.shape[-1]).unsqueeze(0).repeat(b, 1, 1).to(attn_mask.device)
        regular_loss = 1 / (n_vars * (n_vars - 1)) * torch.norm(eye.reshape(b, -1) - attn_mask.reshape((b, -1)),
                                                                p=1, dim=-1)
        loss = clustering_loss.mean(1) + self.k * regular_loss
        mean_loss = loss.mean()
        return mean_loss

# ==========================================
# 2. Channel mask generator (CCTM: learns a near-binary variable adjacency graph)
# ==========================================
class channel_mask_generator(torch.nn.Module):
    def __init__(self, input_size, n_vars):
        super(channel_mask_generator, self).__init__()
        self.generator = nn.Sequential(torch.nn.Linear(input_size * 2, n_vars, bias=False), nn.Sigmoid())
        with torch.no_grad():
            self.generator[0].weight.zero_()
        self.n_vars = n_vars

    def forward(self, x):  # x: [(bs x patch_num) x n_vars x patch_size]
        distribution_matrix = self.generator(x)
        resample_matrix = self._bernoulli_gumbel_rsample(distribution_matrix)

        inverse_eye = 1 - torch.eye(self.n_vars).to(x.device)
        diag = torch.eye(self.n_vars).to(x.device)

        resample_matrix = torch.einsum("bcd,cd->bcd", resample_matrix, inverse_eye) + diag
        return resample_matrix

    def _bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape
        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        r_flatten_matrix = 1 - flatten_matrix

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)
        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)
        return resample_matrix

# ==========================================
# 3. Cross-channel attention Transformer (CCTM: gates information flow only along valid edges)
# ==========================================
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class c_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.d_k = math.sqrt(self.dim_head)
        inner_dim = dim_head * heads
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.dynamicalContranstiveLoss = DynamicalContrastiveLoss(k=regular_lambda, temperature=temperature)

    def forward(self, x, attn_mask=None):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        scale = 1 / self.d_k

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        dynamical_contrastive_loss = None
        scores = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        q_norm = torch.norm(q, dim=-1, keepdim=True)
        k_norm = torch.norm(k, dim=-1, keepdim=True)
        norm_matrix = torch.einsum('bhid,bhjd->bhij', q_norm, k_norm)
        
        if attn_mask is not None:
            def _mask(scores, attn_mask):
                large_negative = -math.log(1e10)
                attention_mask = torch.where(attn_mask == 0, large_negative, 0)
                scores = scores * attn_mask.unsqueeze(1) + attention_mask.unsqueeze(1)
                return scores

            masked_scores = _mask(scores, attn_mask)
            dynamical_contrastive_loss = self.dynamicalContranstiveLoss(scores, attn_mask, norm_matrix)
        else:
            masked_scores = scores

        attn = self.attend(masked_scores * scale)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), attn, dynamical_contrastive_loss

class c_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, c_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, regular_lambda=regular_lambda, temperature=temperature)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, attn_mask=None):
        total_loss = 0
        attn = None
        for attn_layer, ff in self.layers:
            x_n, attn, dcloss = attn_layer(x, attn_mask=attn_mask)
            total_loss += dcloss
            x = x_n + x
            x = ff(x) + x
        dcloss = total_loss / len(self.layers)
        return x, attn, dcloss

class Trans_C(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head, dropout, patch_dim, horizon, d_model, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.transformer = c_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, regular_lambda=regular_lambda, temperature=temperature)
        self.mlp_head = nn.Linear(dim, d_model)

    def forward(self, x, attn_mask=None):
        x = self.to_patch_embedding(x)
        x, attn, dcloss = self.transformer(x, attn_mask)
        x = self.dropout(x)
        x = self.mlp_head(x).squeeze()
        return x, dcloss

# ==========================================
# 4. Wrapped channel fusion module (Channel Fusion Module)
# ==========================================
class ChannelFusionModule(nn.Module):
    """
    CCTM (Cross-Channel Temporal Mixing) of DLRMT.
    Rewrites cross-channel fusion from dense soft attention into explicit structural decisions:
      1) The channel mask generator first learns a near-binary variable adjacency graph
         (deciding which variable pairs are coupled);
      2) The cross-channel Transformer gates attention only along valid edges, structurally
         blocking noise leakage and spurious correlation propagation.
    Contains: channel mask generator + structure-constrained cross-channel attention
    network + dynamic contrastive loss.
    """
    def __init__(self, patch_size, n_vars, dim=64, depth=3, heads=2, mlp_dim=256, 
                 dim_head=64, dropout=0.2, horizon=192, d_model=128, 
                 regular_lambda=0.5, temperature=0.07):
        super().__init__()
        # input_size here corresponds to the original patch_size (it is multiplied by 2
        # internally to match the concatenated complex-valued features)
        self.mask_generator = channel_mask_generator(input_size=patch_size, n_vars=n_vars)
        # Cross-channel attention backbone (the CCTM cross-channel attention path in the paper);
        # the name frequency_transformer is kept only for compatibility
        self.frequency_transformer = Trans_C(
            dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
            dim_head=dim_head, dropout=dropout, patch_dim=patch_size * 2,
            horizon=horizon, d_model=d_model, regular_lambda=regular_lambda,
            temperature=temperature
        )

    def forward(self, x):
        """
        x: [(bs * patch_num), n_vars, patch_size * 2]
        """
        # 1. Generate the attention/fusion mask between channels
        channel_mask = self.mask_generator(x)
        
        # 2. Use the mask for cross-channel information interaction and fusion
        out, dcloss = self.frequency_transformer(x, channel_mask)
        
        return out, dcloss, channel_mask


# ==========================================
# 5. Test code
# ==========================================
if __name__ == "__main__":
    print("=== Testing Channel Fusion Module (CFM) ===")
    
    # Simulate the model input dimensions
    bs = 2                # batch_size
    patch_num = 5         # number of split patches
    n_vars = 4            # number of channels (corresponds to configs.c_in)
    patch_size = 16       # length of a single patch
    
    # The actual input shape to CFM is (bs * patch_num, n_vars, patch_size * 2)
    # The factor of 2 comes from the concatenated real and imaginary parts in the original model
    test_input = torch.randn(bs * patch_num, n_vars, patch_size * 2)
    
    # Initialize the module
    cfm = ChannelFusionModule(
        patch_size=patch_size,
        n_vars=n_vars,
        dim=64,
        depth=2,
        heads=2,
        mlp_dim=128,
        dim_head=32,
        dropout=0.1,
        horizon=192,
        d_model=128,
        regular_lambda=0.5,
        temperature=0.07
    )
    
    # Forward pass
    out, dcloss, mask = cfm(test_input)
    
    print(f"\n[Input Info]")
    print(f"Input Tensor Shape: {test_input.shape}  -> (bs * patch_num, n_vars, patch_size * 2)")
    
    print(f"\n[Output Info]")
    print(f"Output Feature Shape: {out.shape}     -> (bs * patch_num, n_vars, d_model)")
    print(f"Dynamic Contrastive Loss: {dcloss.item():.4f}")
    print(f"Generated Channel Mask Shape: {mask.shape} -> (bs * patch_num, n_vars, n_vars)")
    print("\nTest Passed Successfully!")