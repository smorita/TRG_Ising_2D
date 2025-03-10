#!/usr/bin/env python3
"""The bond-weighted TRG algorithm for the Ising model on the square lattice

Reference:
    D. Adachi, T. Okkubo and S. Todo: Phys. Rev. B 105, L060402 (2022)
"""

import textwrap
import numpy as np

import ising_2d as ising
import common
from trg import TRG

class BWTRG(TRG):
    def __init__(self, temp: float, chi: int, k: float) -> None:
        super().__init__(temp, chi)
        self.method = "BWTRG"
        self.k = k

        a, w0, w1, n_spin = common.initial_BWTN(self.temp)
        self.A = a
        self.w0 = w0
        self.w1 = w1

        factor = self.normalize()
        self.log_factors = [np.log(factor)]

        self.n_spins = [n_spin]
        self.step = 0

    def trace(self) -> float:
        return np.einsum("ijij, i, j ->", self.A, self.w0, self.w1)

    def normalize(self) -> float:
        w0_max = np.max(self.w0)
        w1_max = np.max(self.w1)
        self.w0 /= w0_max
        self.w1 /= w1_max

        trace = self.trace()
        self.A /= trace

        return w0_max * w1_max * trace

    def update(self) -> None:
        w0_k = safe_power(self.w0, k)
        w1_k = safe_power(self.w1, k)
        w0_kp = np.sqrt(self.w0 / w0_k)
        w1_kp = np.sqrt(self.w1 / w1_k)

        a = np.einsum(
            "ijkl, i, j, k, l -> ijkl",
            self.A,
            w0_kp,
            w1_kp,
            w0_kp,
            w1_kp,
            optimize=True,
        )

        # SVD (left, top) - (right, bottom)
        u0, s0, vt0 = common.svd(a, [3, 0], [1, 2], self.chi)
        c2 = np.einsum("lia, i -> lia", u0, w0_k)
        c0 = np.einsum("ajk, k -> ajk", vt0, w0_k)
        self.w0 = s0

        # SVD (top, right) - (bottom, left)
        u1, s1, vt1 = common.svd(a, [0, 1], [2, 3], self.chi)
        c3 = np.einsum("ija, j -> ija", u1, w1_k)
        c1 = np.einsum("akl, l -> akl", vt1, w1_k)
        self.w1 = s1

        # Contraction
        self.A = np.einsum("ail, bji, kjc, lkd -> abcd", c0, c1, c2, c3, optimize=True)

        # normalize
        factor = self.normalize()

        self.log_factors.append(np.log(factor))
        self.n_spins.append(2 * self.n_spins[-1])
        self.step += 1

    def print_preamble(self) -> None:
        output = f"""\
            # {self.method} for Ising model on the square lattice
            # chi = {self.chi}
            # T = {self.temp}
            # k = {self.k}
            # f_exact = {self.f_exact:.12e}"""
        print(textwrap.dedent(output))


def safe_power(w: np.ndarray, alpha: float, eps: float = 1e-12) -> np.ndarray:
    if alpha >= 0:
        return np.power(w, alpha)
    elif w[-1] > eps:
        return np.power(w, alpha)
    else:
        return np.power(w + eps, alpha)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BWTRG simulation of the 2D Ising model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-k", type=float, default=-0.5, help="BWTRG hyperparameter")
    parser.add_argument("chi", type=int, default=8, nargs="?", help="Bond dimension")
    parser.add_argument("step", type=int, default=16, nargs="?", help="TRG steps")
    parser.add_argument(
        "T", type=float, default=ising.T_C, nargs="?", help="Temperature"
    )
    args = parser.parse_args()

    chi = args.chi
    step = args.step
    temp = args.T
    k = args.k

    BWTRG(temp, chi, k).run(step)
