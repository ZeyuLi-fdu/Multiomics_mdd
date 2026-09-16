"""
Unified model factory (trimmed) -- keeps only the two branches needed by
the MDD OmicFormer embedding-concatenation fusion model.

Corresponds to LateFusionOmicEmbedding in MDD_omicformer_embedding_fusion.py:
  * covar                 -> CovarMLP        (covariate MLP branch)
  * prs / cli / met / pro -> OmicFormerBranch (OmicFormer branch, aligned
                                                with the official
                                                Omicformer/model.py)

All branches follow the same return format (logits, features, aux):
  logits   [B, dim_out]  branch-independent prediction (used for the deep
                          supervision / auxiliary loss)
  features [B, dim]      omic representation vector (used for
                          embedding-concatenation fusion)
  aux      None           placeholder
"""
import numpy as np
import torch
import torch.nn as nn

from OmicFormer import OmicFormer
from channel_generator import SelfCorrelationReorder
from utils import BWAS_correlation


# ============================================================
# Covariate branch: a simple MLP
# ============================================================
class CovarMLP(nn.Module):
    """Covariate branch: a simple MLP, output format aligned to
    (logits, features, aux)."""

    def __init__(self, input_dim: int, dim: int, dim_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, dim * 2), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(dim, dim_out)

    def forward(self, x):
        feat = self.net(x)
        logits = self.classifier(feat)
        return logits, feat, None


# ============================================================
# OmicFormer branch: aligned with the official Omicformer/model.py
# ============================================================
class OmicFormerBranch(nn.Module):
    """Single-omic branch built on OmicFormer (PatchEmbed + Transformer +
    cls_token).

    Two input channels (aligned with the paper's Fig. b, "Dual Statistical
    Prior"):
      * First channel ``x_label_sorted``: features ordered by
        feature-label correlation (official ``utils.BWAS_correlation``,
        descending by absolute correlation).
      * Second channel ``x_self_corr``: features re-ordered by
        GW-OT self-correlation
        (``channel_generator.SelfCorrelationReorder``), so that
        highly self-correlated features sit next to each other along the
        1D sequence.

    Both channels' column-reordering permutations are fitted once on the
    training set, before training starts, via ``fit_channels()``; val/test
    only apply ``transform``, so there is no data leakage.

    Output (logits, features, None):
      logits   [B, dim_out] branch-independent prediction (used for the
                             deep supervision / auxiliary loss)
      features [B, dim]     [CLS] token feature vector (used for
                             embedding-concatenation fusion)
    """

    def __init__(self, num_features, dim, dim_out, depth, heads,
                 attn_dropout=0., ff_dropout=0.,
                 kernel_sizes=(3, 5, 8, 13)):
        super().__init__()
        self.num_features = num_features
        self.dim = dim
        self.omic = OmicFormer(
            num_continuous=num_features,
            dim=dim, depth=depth, heads=heads,
            dim_head=16, dim_out=dim_out,
            attn_dropout=attn_dropout, ff_dropout=ff_dropout,
            kernel_sizes=kernel_sizes,
        )
        # Column-reordering permutations for the two channels (fitted
        # before training; stored as plain attributes, not part of the
        # gradient graph).
        self._label_sort_perm = None   # channel 1: feature-label correlation ordering
        self._selfcorr_perm = None     # channel 2: GW-OT self-correlation ordering

    def fit_channels(self, x_np, y_np):
        """Fit the column-reordering permutations for both channels on the
        training set.

        Args:
            x_np: [N, F] training feature matrix (after StandardScaler).
            y_np: [N] training labels (0/1).
        """
        x_np = np.asarray(x_np, dtype=np.float64)
        y_np = np.asarray(y_np, dtype=np.float64)

        # Channel 1: feature-label correlation ordering (descending by
        # absolute value).
        corr = BWAS_correlation(x_np, y_np.reshape(-1, 1))[:, 0]  # [F]
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        label_perm = np.argsort(-np.abs(corr))
        self._label_sort_perm = torch.tensor(label_perm, dtype=torch.long)

        # Channel 2: GW-OT self-correlation ordering (official
        # SelfCorrelationReorder).
        reorder = SelfCorrelationReorder(
            metric='correlation', loss_fun='kl_loss', epsilon=0.0)
        reorder.fit(x_np)
        # permutation_matrix_[i, j] = 1 means source feature i is mapped to
        # 1D position j; the column-reorder index is the source-feature
        # row index for each column.
        selfcorr_perm = reorder.permutation_matrix_.argmax(axis=0)
        self._selfcorr_perm = torch.tensor(selfcorr_perm, dtype=torch.long)

    def _apply_perm(self, x, perm):
        if perm is None:
            return x
        return x[:, perm.to(x.device)]

    def forward(self, x):
        # Channel 1 [B, 1, D]: feature-label correlation ordering
        x_label_sorted = self._apply_perm(x, self._label_sort_perm).unsqueeze(1)
        # Channel 2 [B, 1, D]: GW-OT self-correlation ordering
        x_self_corr = self._apply_perm(x, self._selfcorr_perm).unsqueeze(1)
        logits, feat = self.omic(x_label_sorted, x_self_corr, return_cls=True)
        return logits, feat, None
