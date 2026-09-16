"""Small numerical utilities used by the training pipeline."""

import numpy as np
import torch


def g_impute_nan_as_mean(x: np.ndarray) -> np.ndarray:
    """Column-wise mean imputation for NaNs."""
    x_mean = np.nanmean(x, axis=0)
    for i in range(x.shape[1]):
        x[np.isnan(x[:, i]), i] = x_mean[i]
    return x


def nets_zscore(x: np.ndarray) -> np.ndarray:
    """Column-wise z-score standardization.

    x: [n_subjects, n_features] numpy array.
    """
    x_zscore = x - x.mean(axis=0)
    stds = x.std(axis=0)
    zero_std = stds == 0
    if zero_std.sum() > 0:
        stds[zero_std] = 0.1
        print(f"Warning: {int(zero_std.sum())} feature(s) are constant (zero std).")
        print("Normalizing them to all zeros instead of dividing by zero...")
    return x_zscore / stds


def BWAS_correlation(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Fisher-z-style cross-correlation between two feature matrices.

    x1, x2: [n_samples, n_features_1] and [n_samples, n_features_2].
    Returns the [n_features_1, n_features_2] correlation matrix.
    """
    x1 = (x1 - x1.mean(axis=0)) / x1.std(axis=0)
    x2 = (x2 - x2.mean(axis=0)) / x2.std(axis=0)
    r = np.dot(x1.T, x2) / x1.shape[0]
    return r


def get_res(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residualize y with respect to x (ordinary least squares, with an
    added intercept column)."""
    x = np.hstack((x, np.ones((x.shape[0], 1))))
    beta = np.dot(np.linalg.pinv(x), y)
    return y - np.dot(x, beta)


def onehot(label: torch.Tensor, n_classes: int = 2) -> torch.Tensor:
    """One-hot encode an integer label tensor on its own device."""
    return torch.zeros(label.size(0), n_classes, device=label.device).scatter_(1, label.view(-1, 1), 1)


def mixup_data(data: torch.Tensor, targets: torch.Tensor, alpha: float, n_classes: int = 2):
    """Standard mixup augmentation, re-normalizing the mixed sample along
    the feature axis afterwards."""
    device = data.device
    indices = torch.randperm(data.size(0), device=device)
    data2 = data[indices]
    targets2 = targets[indices]

    targets = onehot(targets, n_classes)
    targets2 = onehot(targets2, n_classes)

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)]).to(device)
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    # re-normalize the mixed sample along the feature axis
    data_mean = torch.mean(data, dim=2, keepdim=True)
    data_std = torch.std(data, dim=2, keepdim=True)
    data = (data - data_mean) / data_std

    return data, targets
