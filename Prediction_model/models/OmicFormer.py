"""
OmicFormer core architecture.

This file re-implements the exact computation graph of the original
OmicFormer model (see Fig. c in the paper / README), with two changes only:

Naming: the two input channels are now named according to the paper's
terminology (see channel_generator.py / Fig. b):
  ``x_label_sorted``: features ordered by feature-label (feature-task)
  correlation ranking. This was previously called ``x_cont``.
  ``x_self_corr``: features re-ordered by the Gromov-Wasserstein
  optimal-transport module so that highly self-correlated features
  sit close together along the 1D sequence. This was previously
  called ``tabmap_img`` (a legacy name from an earlier prototype).

Comments translated to English.

No layer, no tensor shape, and no forward-pass computation has been changed.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
from torch.nn import Module, ModuleList


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class PreNorm(Module):
    """Pre-LayerNorm wrapper applied before a sub-module (attention or FFN)."""

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GEGLU(Module):
    """Gated GELU activation used inside the feed-forward block."""

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(Module):
    """Standard multi-head self-attention, returns both the output and the
    post-softmax attention map (used for interpretability / feature
    attribution, see Fig. 2 and the biomarker-discovery analyses)."""

    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)
        out = einsum("b h i j, b h j d -> b h i d", dropped_attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out), attn


class Transformer(Module):
    """Stack of pre-norm attention + feed-forward blocks with residual
    connections (Fig. c, right panel)."""

    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        num_residual_streams=4,
    ):
        super().__init__()
        # NOTE: `num_residual_streams` is kept as a constructor argument for
        # interface compatibility with the original implementation, but it
        # is currently unused inside the block construction below (reserved
        # for a future multi-stream residual variant). Behaviour is
        # unchanged from the original code.
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(
                ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                        PreNorm(dim, FeedForward(dim, dropout=ff_dropout)),
                    ]
                )
            )

    def forward(self, x, return_attn=False):
        post_softmax_attns = []
        for attn, ff in self.layers:
            out, attn_map = attn(x)
            x = x + out
            post_softmax_attns.append(attn_map)
            x = x + ff(x)
        if not return_attn:
            return x
        return x, torch.stack(post_softmax_attns)


class MLP(Module):
    """Simple prediction head applied to the pooled [CLS] token."""

    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            layers.append(nn.Linear(dim_in, dim_out))
            if is_last:
                continue
            act = default(act, nn.ReLU())
            layers.append(act)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class PatchEmbed(Module):
    """
    Multi-scale 1D patch embedding.

    Runs several parallel Conv1d branches with different kernel sizes over
    the 1D feature sequence, resamples every branch to a common length, and
    concatenates them along the embedding dimension. This is the "1D conv
    patch kernel_size 1..4 -> linear interpolation -> concat" block shown
    in Fig. c.
    """

    def __init__(
        self,
        dim_in=2,
        dim_out=768,
        kernel_sizes=(5, 13, 47, 89),
        stride=16,
        padding_mode="same",
    ):
        super().__init__()
        self.scales = nn.ModuleList()
        for k in kernel_sizes:
            pad = k // 2 if padding_mode == "same" else 0
            branch_stride = max(1, k // 2)
            conv = nn.Conv1d(
                in_channels=dim_in,
                out_channels=dim_out // len(kernel_sizes),
                kernel_size=k,
                stride=branch_stride,
                padding=pad,
            )
            self.scales.append(conv)

    def forward(self, x):
        # x: [B, C_in, L]
        feats = [conv(x) for conv in self.scales]  # list of [B, D_i, L_i]
        max_len = max(f.shape[-1] for f in feats)
        feats = [F.interpolate(f, size=max_len, mode="linear", align_corners=False) for f in feats]
        x = torch.cat(feats, dim=1)  # [B, D, L]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb


class OmicFormer(Module):
    """
    OmicFormer: a statistical-priors-informed Transformer for omics
    prediction (Fig. c).

    The model consumes two 1D channels of the same feature set, both
    produced by the Dual Statistical Prior module (see
    ``channel_generator.py`` / Fig. b):

      ``x_label_sorted`` [B, 1, F]: raw (z-scored) feature values, ordered
      by feature-label correlation ranking.

      ``x_self_corr``    [B, 1, F]: the same feature values re-ordered by
      the Gromov-Wasserstein optimal-transport module, so that features
      with high feature-feature correlation are placed close together
      along the sequence.

    The two channels are combined with a learnable softmax gate
    (``self.channel_gate``) before being fed into the multi-scale 1D patch
    embedding and the Transformer encoder.
    """

    def __init__(
        self,
        *,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head=16,
        dim_out=1,
        mlp_hidden_mults=(4, 2),
        mlp_act=None,
        continuous_mean_std=None,
        attn_dropout=0.0,
        ff_dropout=0.0,
        num_residual_streams=4,
        patch_size=32,
        patch_stride=16,
        kernel_sizes=(5, 13, 47, 89),
    ):
        super().__init__()
        assert num_continuous > 0
        self.num_continuous = num_continuous
        self.dim = dim

        # optional external standardization (mean/std computed offline)
        if exists(continuous_mean_std):
            assert continuous_mean_std.shape == (num_continuous, 2)
            self.register_buffer("continuous_mean_std", continuous_mean_std)

        # multi-scale patch embedding; dim_in=2 -> [label-sorted, self-corr] channels
        self.patch_embed = PatchEmbed(dim_in=2, dim_out=dim, kernel_sizes=kernel_sizes)

        def calc_num_patches(seq_len, kernel, stride):
            if seq_len <= 0:
                return 0
            eff_len = max(seq_len, kernel)
            return max(1, math.ceil((eff_len - kernel) / stride) + 1)

        self.num_patches = calc_num_patches(num_continuous, patch_size, patch_stride)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            num_residual_streams=num_residual_streams,
        )

        hidden_dimensions = [dim * t for t in mlp_hidden_mults]
        all_dimensions = [dim, *hidden_dimensions, dim_out]
        self.mlp = MLP(all_dimensions, act=mlp_act)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # learnable gate that fuses the two statistical-prior channels
        # (Fig. b, "Learnable gate"). Shape assumes 2 input channels.
        self.channel_gate = nn.Parameter(torch.zeros(1, 2, 1))

    def forward(self, x_label_sorted, x_self_corr=None, return_attn=False, return_cls=False):
        """
        Args:
            x_label_sorted: [B, 1, F] feature-label-correlation-sorted channel.
            x_self_corr: [B, 1, F] or [B, 1, H, W] self-correlation-ordered
                channel produced by the optimal-transport reordering module.
                If None, only the label-sorted channel is used (the gate
                will simply down-weight the missing channel via training,
                but for a clean 2-channel forward pass this argument is
                expected to be provided).
            return_attn: also return the stacked per-layer attention maps.
            return_cls: (extension, backwards-compatible) also return the [CLS]
                token feature [B, dim] before the MLP head, for downstream
                embedding-concat fusion. Default False keeps the original
                return contract unchanged.
        """
        x = x_label_sorted
        if exists(getattr(self, "continuous_mean_std", None)):
            mean, std = self.continuous_mean_std.unbind(dim=-1)
            x = (x - mean) / std

        if x_self_corr is not None:
            if x_self_corr.ndim == 3:
                # [B, C, L], same layout as x -> concat on channel dim
                x = torch.cat([x, x_self_corr], dim=1)
            elif x_self_corr.ndim == 4:
                # [B, 1, H, W] -> flatten spatial dims to a 1D sequence first
                b, c, h, w = x_self_corr.shape
                x_self_corr = x_self_corr.view(b, c, h * w)
                x = torch.cat([x, x_self_corr], dim=1)
            else:
                raise ValueError(f"Unexpected x_self_corr shape: {x_self_corr.shape}")

        # learnable channel-fusion gate (softmax over the 2 channels)
        gate = torch.softmax(self.channel_gate, dim=1)  # [1, 2, 1], sums to 1
        x = x * gate

        x = self.patch_embed(x)  # [B, N, D]

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.dim, np.arange(x.shape[1], dtype=np.float32))
        pos_embed = torch.from_numpy(pos_embed).to(x.device).unsqueeze(0).float()
        x = x + pos_embed

        x, attns = self.transformer(x, return_attn=True)
        cls_out = x[:, 0, :]

        # The MLP head's input dimension is fixed at construction time
        # (dim -> mlp_hidden_mults -> dim_out), matching the [CLS] token
        # width, so no dynamic re-shaping is required here.
        logits = self.mlp(cls_out)

        if return_cls:
            if return_attn:
                return logits, cls_out, attns
            return logits, cls_out
        if not return_attn:
            return logits
        return logits, attns
