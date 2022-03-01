#!/usr/bin/env python3
"""The HOTRG algorithm for the Ising model on the square lattice

Reference:
    Z. Y. Xie, et al.: Phys. Rev. B 86, 045139 (2012)
"""

import numpy as np

import ising_2d as ising
import common
from trg import TRG


class HOTRG(TRG):
    def __init__(self, temp, chi):
        super().__init__(temp, chi)
        self.method = "HOTRG"

    def update(self):
        self.update_hotrg("x")
        self.update_hotrg("y")
        self.step += 1

    def update_hotrg(self, direction):
        if direction == "y":
            axes_trans = (1, 2, 3, 0)
            axes_inv = (3, 0, 1, 2)
        else:
            axes_trans = (0, 1, 2, 3)
            axes_inv = (0, 1, 2, 3)

        a = np.transpose(self.A, axes_trans)
        u, _, _ = common.svd(contract_a4(a), [0, 2], [1, 3], self.chi)
        # The original HOTRG computes the isometry of both sides and decides which
        # one to use based on the error comparison. In this code, we assume the
        # symmetry and omit the calculation of one of the isometries.
        a = contract_a2u2(a, u)
        self.A = np.transpose(a, axes_inv)

        # normalize
        factor = self.trace()
        self.A /= factor

        self.log_factors.append(np.log(factor))
        self.n_spins.append(2 * self.n_spins[-1])


def contract_a4(a):
    return np.tensordot(
        np.tensordot(
            a, a, ([2, 3], [2, 3])
        ), np.tensordot(
            a, a, ([1, 2], [1, 2])
        ), ([1, 3], [1, 3])
    )


def contract_a2u2(a, u):
    shape = a.shape
    n = u.shape[2]
    a_new = np.zeros((n, shape[1], n, shape[3]), dtype=float)

    # loop blocking to reduce memory usage
    for j0, j1 in get_block(shape[1]):
        a_new += np.tensordot(
            np.tensordot(
                u, a[:, :, :, j0:j1], ([1], [0])
            ), np.tensordot(
                u, a[:, j0:j1, :, :], ([0], [2])
            ), ([0, 3, 4], [2, 0, 3])
        )

    return a_new


def get_block(n):
    block_size = 4
    block_list = [(i, i + block_size) for i in range(0, n, block_size)]
    if n % block_size != 0:
        block_list[-1] = ((n // block_size) * block_size, n)
    return block_list


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HOTRG simulation of the 2D Ising model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("chi", type=int, default=8, nargs="?", help="Bond dimension")
    parser.add_argument("step", type=int, default=16, nargs="?", help="HOTRG steps")
    parser.add_argument("T", type=float, default=ising.T_C, nargs="?", help="Temperature")
    args = parser.parse_args()

    Chi = args.chi
    Step = args.step
    T = args.T

    HOTRG(T, Chi).run(Step)
