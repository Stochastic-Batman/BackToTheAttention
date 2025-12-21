import os
import torch
import torch.nn as nn


# I read the paper about the Rotary Embedding in August 2025, and as it is better than simple absolute or relative embeddings, why not use it?
class RoPE(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base

        # precompute frequencies: \theta_i = base^{-2i/dim}
        inv_freq = 1.0 / (base ** (torch.arange(start=0, end=dim, step=2, dtype=torch.float32) / dim))

        # to save as part of a model's persistent state without treating it as a learnable parameter
        self.register_buffer(name="inv_freq", tensor=inv_freq, persistent=True)  # shape: (dim / 2,)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        positions = torch.arange(0, seq_len, device=x.device, dtype=self.inv_freq.dtype)  # (seq_len,)
        angles = positions[:, None] * self.inv_freq[None, :]  # (seq_len, dim//2)
        sin, cos = torch.sin(angles), torch.cos(angles)  # (seq_len, dim / 2)
        # expand dims for broadcasting: (seq_len, dim//2) -> (1, seq_len, dim//2)
        sin = sin.unsqueeze(0)
        cos = cos.unsqueeze(0)

        # split the last dimension
        x1 = x[..., : self.dim // 2]  # (B, seq_len, dim / 2)
        x2 = x[..., self.dim // 2:]  # (B, seq_len, dim / 2)
        # Rotation matrix
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        return torch.cat([rotated_x1, rotated_x2], dim=-1)


# DeTentionBlock is the core repeating unit that applies RoPE-enhanced self-attention and feed-forward processing with residual connections.
class DeTentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_hidden_size: int = 256, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model  # must be even and divisible by n_heads
        self.n_heads = n_heads
        self.ff_hidden_size = ff_hidden_size
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.rope = RoPE(dim=d_model, base=10000.0)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)  # makes input (batch, seq, dim) instead of (seq, batch, dim)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden_size),
            nn.GLU(dim=-1),  # https://arxiv.org/pdf/2002.05202; please, check the last line of section 4 (Conclusions)
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_size // 2, d_model)
        )
        self.dropout_layer = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer with residual connection
        residual = x
        x = self.norm1(x)
        q, k = self.rope(x, seq_len=x.shape[1]), self.rope(x, seq_len=x.shape[1])
        attn_output, _ = self.attention(query=q, key=k, value=x, need_weights=False)  # I do not intend to visualize attention
        x = self.dropout_layer(attn_output) + residual

        # FF sub-layer with residual connection
        residual = x
        x = self.ff(self.norm2(x))
        x = self.dropout_layer(x) + residual

        return x


# DeTention is the complete end-to-end model that handles input projection from raw prices, stacks multiple DeTentionBlocks and produces the final prediction.
class DeTention(nn.Module):
    def __init__(self, seq_len: int = 30, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, ff_hidden_size: int = 256, dropout: float = 0.2, use_avg_pool: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_avg_pool = use_avg_pool

        self.input_proj = nn.Linear(1, d_model)  # 'Close' -> d_model
        self.DeTention_blocks = nn.ModuleList([DeTentionBlock(d_model=d_model, n_heads=n_heads, ff_hidden_size=ff_hidden_size, dropout=dropout) for i in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, 1)

        # optional global average pooling instead of last token
        if use_avg_pool:
            self.pool = nn.AdaptiveAvgPool1d(output_size=1)  # (B, d_model, L) -> (B, d_model, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)  # (batch_size, seq_len, 1) -> (batch_size, seq_len, d_model)
        for block in self.DeTention_blocks:
            x = block(x)
        x = self.norm(x)

        # pooling or last token
        if self.use_avg_pool:
            x = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, d_model, L) -> (B, d_model); nn.AdaptiveAvgPool1d expects channel dimension second (like images: N, C, L)
        else:
            x = x[:, -1, :]  # last position

        x = self.output_head(x)  # next price prediction: (batch_size, 1)

        return x.squeeze(dim=-1)  # (batch_size,)


def save_model(model: DeTention, path: str = "models/DeTention.pth") -> None:  # save the entire model (architecture + weights)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'seq_len': model.seq_len,
            'd_model': model.d_model,
            'use_avg_pool': model.use_avg_pool
        }
    }, path)


def load_model(path: str = "models/DeTention.pth", **model_kwargs) -> DeTention:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=True)

    config = checkpoint.get('config', {})
    config.update(model_kwargs)
    model = DeTention(**config)
    model.load_state_dict(checkpoint['state_dict'])

    return model