"""
Self-correlation-ordered channel generator (Fig. b: the "self-correlation
ordered" channel).

This is a renamed, English-commented, faithful port of the original
``TabMapGenerator`` (+ its ``gromov_wass_solver`` helper). The numerical
algorithm is unchanged from the original implementation:

It computes the pairwise feature-feature distance matrix (default metric:
correlation distance) over the input feature columns. Then it computes
the pairwise distance matrix of an evenly-spaced 1D grid with as many
points as there are features. After that it solves a (linearized)
Gromov-Wasserstein optimal-transport problem between these two structure
matrices to get a soft coupling between "features" and "1D grid slots".
Next it converts the soft coupling into a hard one-to-one feature ->
position assignment via the Hungarian algorithm
(``scipy.optimize.linear_sum_assignment``), yielding a permutation matrix.
Finally ``transform`` simply right-multiplies the (already
feature-selected) input matrix by this fixed permutation matrix, i.e. it
re-orders columns so that features with similar feature-feature distance
profiles end up next to each other along the 1D sequence.

Note: this module does **not** use the label/task at all — it is purely
feature-feature-structure-driven. The complementary "feature-label sorted"
channel (Fig. b, left path) is computed separately, directly in the
training script, from a feature-task correlation ranking (see
``omicformer/utils.py:BWAS_correlation`` and ``train.py``).

Renaming vs. the original implementation

``TabMapGenerator`` is now ``SelfCorrelationReorder``. ``gromov_wass_solver``
is now ``GromovWassersteinSolver``. ``create_space_distributions`` is now
``_uniform_marginals``. ``tensor_product`` is now ``_gw_tensor_product``.
No computation was changed; only names, docstrings, and code style.
"""

from typing import Optional

import numpy as np
import ot
from ot import bregman
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import pairwise_distances


def _uniform_marginals(n_source: int, n_target: int):
    """Uniform marginal distributions over the source and target 1D spaces."""
    return ot.unif(n_source), ot.unif(n_target)


def _gw_tensor_product(const_c: np.ndarray, h_c1: np.ndarray, h_c2: np.ndarray, coupling: np.ndarray) -> np.ndarray:
    """Linearized Gromov-Wasserstein cost tensor for a given coupling matrix."""
    cross_term = -np.dot(h_c1, coupling).dot(h_c2.T)
    return const_c + cross_term


class GromovWassersteinSolver:
    """
    Solves for a transport coupling that minimizes the (linearized)
    Gromov-Wasserstein discrepancy between two structure matrices ``C1``
    (source) and ``C2`` (target).

    With ``epsilon == 0`` (the default, and what ``SelfCorrelationReorder``
    uses), this performs a single linearized step: build the GW cost
    tensor from an initial uniform coupling, then solve one exact
    optimal-transport problem (network-simplex / EMD) under that fixed
    cost.

    With ``epsilon > 0``, it instead runs the standard iterative
    entropic-regularized Gromov-Wasserstein loop: recompute the cost
    tensor from the current coupling, re-solve via Sinkhorn, and repeat
    until the coupling update falls below ``tol`` or ``maxiter`` is
    reached.
    """

    def __init__(self, loss_fun: str = "kl_loss", epsilon: float = 0.0, tol: float = 1e-9, seed: int = 42):
        self.loss_fun = loss_fun
        self.epsilon = epsilon
        self.tol = tol
        np.random.seed(seed)

    def _init_matrices(self, c1: np.ndarray, c2: np.ndarray, u: np.ndarray, v: np.ndarray):
        if self.loss_fun == "kl_loss":
            f1 = lambda a: a * np.log(a + 1e-15) - a
            f2 = lambda b: b
            h1 = lambda a: a
            h2 = lambda b: np.log(b + 1e-15)
        elif self.loss_fun == "square_loss":
            f1 = lambda a: (a ** 2) / 2
            f2 = lambda b: (b ** 2) / 2
            h1 = lambda a: a
            h2 = lambda b: b
        else:
            raise ValueError(f"Unsupported loss_fun for the tensorized GW solver: {self.loss_fun!r}")

        const_c1 = np.dot(np.dot(f1(c1), u.reshape(-1, 1)), np.ones(len(v)).reshape(1, -1))
        const_c2 = np.dot(np.ones(len(u)).reshape(-1, 1), np.dot(v.reshape(1, -1), f2(c2).T))
        const_c = const_c1 + const_c2
        return const_c, h1(c1), h2(c2)

    def solve(self, c1: np.ndarray, c2: np.ndarray, u: np.ndarray, v: np.ndarray,
              maxiter: int = 100, print_every: int = 1, verbose: bool = False) -> np.ndarray:
        c1 = np.asarray(c1, dtype=np.float64)
        c2 = np.asarray(c2, dtype=np.float64)
        c1 = c1 / c1.mean()
        c2 = c2 / c2.mean()

        coupling = np.outer(u, v)

        const_c = h_c1 = h_c2 = None
        if self.loss_fun in ("square_loss", "kl_loss"):
            const_c, h_c1, h_c2 = self._init_matrices(c1, c2, u, v)
            cost = _gw_tensor_product(const_c, h_c1, h_c2, coupling)
        elif self.loss_fun == "sqeuclidean":
            cost = ot.dist(c1, c2, metric=self.loss_fun)
        else:
            raise ValueError(f"Unsupported loss_fun: {self.loss_fun!r}")

        if self.epsilon == 0:
            coupling = ot.lp.emd(u, v, cost, numItermax=int(1e7))
        else:
            it, err = 0, 1.0
            while err > self.tol and it <= maxiter:
                prev_coupling = coupling
                cost = _gw_tensor_product(const_c, h_c1, h_c2, coupling)
                coupling = bregman.sinkhorn(u, v, cost, self.epsilon, numItermax=100000)
                err = np.linalg.norm(coupling - prev_coupling)
                if verbose and it % print_every == 0:
                    print(f"{it:5d}|{err:8e}|")
                it += 1

        return coupling


class SelfCorrelationReorder:
    """
    Re-arranges a feature matrix's columns along a 1D sequence so that
    features with similar pairwise-distance profiles (default metric:
    correlation distance) sit close together, by solving a
    Gromov-Wasserstein optimal-transport problem against a uniform 1D grid
    and converting the resulting transport plan into a hard permutation via
    the Hungarian algorithm. This is the "self-correlation ordered" channel
    in Fig. b.

    Parameters

    metric : str
        Distance metric passed to ``sklearn.metrics.pairwise.
        pairwise_distances`` to build the feature-feature structure
        matrix. Default: "correlation".
    loss_fun : {"kl_loss", "square_loss", "sqeuclidean"}
        Loss function for the Gromov-Wasserstein solver.
    epsilon : float
        Entropic regularization strength. ``0`` (default) uses a single
        linearized exact-OT step; ``> 0`` uses iterative Sinkhorn-GW.
    num_iter : int
        Maximum number of solver iterations (only used when
        ``epsilon > 0``).
    """

    def __init__(self, metric: str = "correlation", loss_fun: str = "kl_loss",
                 epsilon: float = 0.0, num_iter: int = 10):
        self.metric = metric
        self.loss_fun = loss_fun
        self.epsilon = epsilon
        self.num_iter = num_iter

        self.num_points = None
        self.permutation_matrix_: Optional[np.ndarray] = None

    def _feature_distance_matrix(self, x: np.ndarray) -> np.ndarray:
        return pairwise_distances(x.T, metric=self.metric)

    def _grid_distance_matrix(self, num_points: int) -> np.ndarray:
        positions = np.arange(num_points).reshape(-1, 1)
        return pairwise_distances(positions, metric="euclidean")

    def fit(self, x: np.ndarray, num_points: Optional[int] = None) -> "SelfCorrelationReorder":
        """
        Args:
            x: [N, F] feature matrix (already feature-selected, if
                applicable — this module only re-orders whatever columns
                it is given; it does not do feature selection itself).
            num_points: size of the 1D target grid. Defaults to the number
                of features (a full permutation).
        """
        num_features = x.shape[1]
        if num_points is None:
            num_points = num_features
        effective_points = min(num_points, num_features)

        feat_dist = self._feature_distance_matrix(x)
        grid_dist = self._grid_distance_matrix(effective_points)
        u, v = _uniform_marginals(effective_points, effective_points)

        solver = GromovWassersteinSolver(loss_fun=self.loss_fun, epsilon=self.epsilon)
        print("Solving the Gromov-Wasserstein optimal-transport problem...")
        soft_coupling = solver.solve(feat_dist, grid_dist, u, v, maxiter=self.num_iter, print_every=10, verbose=False)

        if soft_coupling.sum() == 0:
            raise ValueError("Optimal-transport solve failed (all-zero coupling); try adjusting epsilon and retry.")

        print("Performing linear sum assignment...")
        row_idx, col_idx = linear_sum_assignment(-soft_coupling)
        permutation = np.zeros_like(soft_coupling)
        permutation[row_idx, col_idx] = 1

        self.num_points = num_points
        self.permutation_matrix_ = permutation
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Applies the fitted feature permutation to a (possibly
        different) set of samples over the same feature columns."""
        if self.permutation_matrix_ is None:
            raise RuntimeError("Call `fit()` before `transform()`.")
        return np.matmul(x, self.permutation_matrix_)  # [N, F]

    def fit_transform(self, x: np.ndarray, num_points: Optional[int] = None) -> np.ndarray:
        self.fit(x, num_points=num_points)
        return self.transform(x)
