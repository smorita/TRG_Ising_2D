#!/usr/bin/env python3
"""The TRG algorithm for the Ising model on the square lattice

Reference:
    M. Levin, C. P. Nave: Phys. Rev. Lett. 99, 120601 (2007)
"""

import textwrap
import numpy as np

import ising_2d as ising
import common


class TRG:
    def __init__(self, temp, chi):
        self.method = "TRG"
        self.temp = temp
        self.chi = chi
        self.f_exact = ising.exact_free_energy(temp)

        a, log_factor, n_spin = common.initial_TN(self.temp)
        self.A = a
        self.log_factors = [log_factor]
        self.n_spins = [n_spin]
        self.step = 0

    def trace(self):
        return np.einsum("ijij->", self.A)

    def log_Z(self):
        trace_a = self.trace()
        # if trace_a < 0.0:
        #     logging.warning("Negative trace_a %e (%d)", trace_a, self.step)
        log_z = np.sum(np.array(self.log_factors) / np.array(self.n_spins))
        log_z += np.log(abs(trace_a)) / self.n_spins[-1]
        return log_z

    def free_energy(self):
        return -self.temp * self.log_Z()

    def update(self):
        # SVD (top, right) - (bottom, left)
        u, s, vt = common.svd(self.A, [0, 1], [2, 3], self.chi)
        sqrt_s = np.sqrt(s)
        c3 = u * sqrt_s[None, None, :]
        c1 = vt * sqrt_s[:, None, None]

        # SVD (top, left) - (right, bottom)
        u, s, vt = common.svd(self.A, [0, 3], [1, 2], self.chi)
        sqrt_s = np.sqrt(s)
        c2 = u * sqrt_s[None, None, :]
        c0 = vt * sqrt_s[:, None, None]

        # Contraction
        self.A = np.tensordot(
            np.tensordot(c0, c1, (1, 2)),
            np.tensordot(c2, c3, (1, 1)),
            ((1, 3), (2, 0)))

        # normalize
        factor = self.trace()
        self.A /= factor

        self.log_factors.append(np.log(factor))
        self.n_spins.append(2 * self.n_spins[-1])
        self.step += 1

    def print_legend(self):
        output = f"""\
            # {self.method} for Ising model on the square lattice
            # chi= {self.chi}
            # T= {self.temp}
            # f_exact= {self.f_exact:.12e}
            # 1: step
            # 2: N_spin
            # 3: free energy
            # 4: Relative error in the free energy, (f-f_exact)/f_exact"""
        print(textwrap.dedent(output))

    def print_results(self):
        n_spin = self.n_spins[-1]
        f = self.free_energy()
        f_err = (f - self.f_exact) / self.f_exact
        results = [f"{self.step:04d}",
                   f"{n_spin:.12e}",
                   f"{f:.12e}",
                   f"{f_err:.12e}"]
        print(" ".join(results))

    def run(self, step):
        self.print_legend()
        self.print_results()
        for i in range(step):
            self.update()
            self.print_results()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TRG simulation of the 2D Ising model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("chi", type=int, default=8, nargs="?", help="Bond dimension")
    parser.add_argument("step", type=int, default=16, nargs="?", help="TRG steps")
    parser.add_argument("T", type=float, default=ising.T_C, nargs="?", help="Temperature")
    args = parser.parse_args()

    Chi = args.chi
    Step = args.step
    T = args.T

    TRG(T, Chi).run(Step)
