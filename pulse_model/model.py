"""
Pulse-consistency transformer.

Same building blocks as the from-scratch LLM (multi-head attention, LayerNorm with gamma/beta,
MLP, pre-norm residual blocks), changed for this task:
  - tokens are FACE REGIONS, not words: each token is a linear embedding of that region's
    10 s pulse waveform plus a learned region embedding (the region embedding replaces
    positional encoding)
  - a CLS token collects the answer; output is one logit (sigmoid -> probability of fake)
  - hidden regions (e.g. neck under a hijab) are masked out of attention
  - everything is batched and PyTorch autograd does the backward pass, so the model can be
    exported to ONNX for the laptop
"""
import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.last_attn = None  # [B, heads, N, N], kept for visualising which regions were compared

    def forward(self, E, pad):
        # E: [B, N, d_model]; pad: [B, N] True where the token is hidden
        B, N, _ = E.shape
        Q = self.W_Q(E).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(E).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(E).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / self.d_k ** 0.5
        scores = scores.masked_fill(pad[:, None, None, :], -1e9)
        A = torch.softmax(scores, dim=-1)
        self.last_attn = A.detach()

        out = (self.drop(A) @ V).transpose(1, 2).reshape(B, N, -1)
        return self.W_O(out)


class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, E):
        mu = E.mean(dim=-1, keepdim=True)
        var = ((E - mu) ** 2).mean(dim=-1, keepdim=True)
        return self.gamma * (E - mu) / (var + 1e-5) ** 0.5 + self.beta


class MLP(nn.Module):
    def __init__(self, d_model, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, E):
        return self.net(E)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attention = AttentionBlock(d_model, num_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio * d_model, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, E, pad):
        E = E + self.drop(self.attention(self.norm1(E), pad))
        E = E + self.drop(self.mlp(self.norm2(E)))
        return E


class PulseTransformer(nn.Module):
    def __init__(self, num_regions, in_features, d_model=64, num_heads=4, num_layers=3,
                 mlp_ratio=4, dropout=0.1):
        # in_features = waveform samples (300) + timing features (11 with xcorr_lags=5)
        super().__init__()
        self.args = dict(num_regions=num_regions, in_features=in_features, d_model=d_model,
                         num_heads=num_heads, num_layers=num_layers, mlp_ratio=mlp_ratio,
                         dropout=dropout)
        self.token_embed = nn.Linear(in_features, d_model)
        self.region_embed = nn.Parameter(torch.zeros(num_regions, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.region_embed, mean=0, std=0.02)
        nn.init.normal_(self.cls, mean=0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, mask):
        # x: [B, R, in_features] from data.make_tokens; mask: [B, R] 1 = region visible, 0 = hidden
        B = x.shape[0]
        tokens = self.token_embed(x) + self.region_embed
        E = torch.cat([self.cls.expand(B, -1, -1), tokens], dim=1)
        visible = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1) > 0.5  # CLS always visible
        pad = ~visible
        for block in self.blocks:
            E = block(E, pad)
        return self.head(self.norm(E[:, 0])).squeeze(-1)  # logit; sigmoid(logit) = P(fake)


def load_model(path, device="cpu"):
    """Loads a checkpoint written by train.py. Returns (model, checkpoint dict with regions/threshold/etc)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = PulseTransformer(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
