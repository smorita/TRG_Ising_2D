#!/usr/bin/env python3
"""Common parts of TRG and HOTRG algorithms for the Ising model on the square lattice"""

from collections.abc import Iterator
from typing import Optional
import numpy as np
import scipy.linalg as spl


def svd(
    a: np.ndarray,
    axes0: Iterator[int],
    axes1: Iterator[int],
    rank: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Singular value decomposition for tensor.

    Args:
        a: A tensor to be decomposed.
        axes0: Axes which connect to U.
        axes1: Axes which connect to VT.
        rank: The maximum number of singular values to be returned.
            If it is None (default), there is no truncation.

    Returns:
        U: A isometric tensor containing the left singular vectors.
            The last index corresponds to the singular values.
        S: The singular values.
        VT: A isometric tensor containing the right singular vectors.
            The first index corresponds to the singular values.

    Note:
        The function calculates the full SVD and then truncates it.
        Therefore, it is not expected to reduce the computational cost.
    """
    shape = np.array(a.shape)
    shape_row = [shape[i] for i in axes0]
    shape_col = [shape[i] for i in axes1]
    n_row = np.prod(shape_row)
    n_col = np.prod(shape_col)

    chi = min(n_row, n_col)
    if (rank is not None) and (rank < chi):
        chi = rank

    mat = np.reshape(np.transpose(a, axes0 + axes1), (n_row, n_col))
    u, s, vt = spl.svd(mat, full_matrices=False)

    if chi < len(s):
        u = u[:, 0:chi]
        s = s[0:chi]
        vt = vt[0:chi, :]

    n = len(s)
    return u.reshape(shape_row + [n]), s, vt.reshape([n] + shape_col)


def initial_TN(temp: float) -> tuple[np.ndarray, float, float]:
    """Initial tensor of the ising model on the square lattice.

    Args:
        temp: Temperature

    Returns:
        a: Initial 4-leg tensor. [top, right, bottom, left]
        log_factor: Logarithm of the normalization factor.
        n_spin: The number of spins which contained the initial tensor.
    """
    shape = (2, 2, 2, 2)
    a = np.zeros(shape, dtype=float)  # [top, right, bottom, left]
    c = np.cosh(1.0 / temp)
    s = np.sinh(1.0 / temp)
    for idx in np.ndindex(shape):
        if sum(idx) == 0:
            a[idx] = 2 * c * c
        elif sum(idx) == 2:
            a[idx] = 2 * c * s
        elif sum(idx) == 4:
            a[idx] = 2 * s * s

    # normalize
    val = np.einsum("ijij", a)
    a /= val
    log_factor = np.log(val)

    n_spin = 1.0  # An initial tensor has one spin.
    return (a, log_factor, n_spin)


def initial_BWTN(temp: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Initial bond-weighted tensor of the ising model on the square lattice.

    Args:
        temp: Temperature

    Returns:
        a: Initial 4-leg tensor. [top, right, bottom, left]
        w0: bond weight on the vertical bond
        w1: bond weight on the horizontal bond
        n_spin: The number of spins which contained the initial tensor.
    """
    shape = (2, 2, 2, 2)
    a = np.zeros(shape, dtype=float)  # [top, right, bottom, left]
    for idx in np.ndindex(shape):
        if sum(idx) % 2 == 0:
            a[idx] = 0.5

    c = np.cosh(1.0 / temp)
    s = np.sinh(1.0 / temp)
    w0 = 2.0 * np.array([c, s])
    w1 = 2.0 * np.array([c, s])

    n_spin = 1.0  # An initial tensor has one spin.
    return (a, w0, w1, n_spin)
