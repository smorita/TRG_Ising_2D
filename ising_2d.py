#!/usr/bin/env python3
"""Exact results of the Ising model on the square lattice"""

from math import sqrt, log, cos, cosh, sinh, tanh, isnan, isinf, pi
import scipy.integrate as integrate
from scipy.special import ellipk, ellipe

T_C = 2.0 / log(1.0 + sqrt(2.0))
Beta_C = 1.0 / T_C


def exact_free_energy(temp, error=False):
    beta = 1.0 / temp
    cc, ss = cosh(2.0 * beta), sinh(2.0 * beta)
    k = 2.0 * ss / cc**2

    def integrant(x):
        return log(1.0 + sqrt(abs(1.0 - k * k * cos(x)**2)))
    integral, err = integrate.quad(integrant, 0, 0.5 * pi, epsabs=1e-13, epsrel=1e-13)
    result = integral / pi + log(cc) + 0.5 * log(2.0)
    if not error:
        return -result / beta
    else:
        return -result / beta, err / beta


def exact_energy(temp):
    beta = 1.0 / temp
    cc, ss, tt = cosh(2.0 * beta), sinh(2.0 * beta), tanh(2.0 * beta)
    k = 2.0 * ss / cc**2
    kk = ellipk(k * k)
    if (isnan(kk) or isinf(kk)) and (1.0 - 2.0 * tt * tt) < 1e-15:
        return -(1.0 / tt)
    return -(1.0 / tt) * (1.0 - (1.0 - 2.0 * tt * tt) * (2.0 / pi) * kk)


def exact_specific_heat(temp):
    beta = 1.0 / temp
    cc, ss, tt = cosh(2.0 * beta), sinh(2.0 * beta), tanh(2.0 * beta)
    k = 2.0 * ss / cc**2
    kk = ellipk(k * k)
    ek = ellipe(k * k)
    val = kk - ek - (pi / 2.0 - (1.0 - 2.0 * tt * tt) * kk) / (cc * cc)
    return 4.0 * beta * beta * val / (pi * tt * tt)


def exact_magnetization(temp):
    beta = 1.0 / temp
    if beta <= Beta_C:
        return 0
    else:
        return (1.0 - sinh(2.0 * beta)**(-4.0))**0.125


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Exact results of the Ising model on the square lattice",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("T", type=float, default=T_C, nargs="?", help="Temperature")
    args = parser.parse_args()
    temp = args.T

    if temp == T_C:
        print(f"T_C = 2.0/log(1.0+sqrt(2.0)) = {T_C:.14f}")
    else:
        print(f"T = {temp:.14f}")

    f = exact_free_energy(temp, error=False)
    e = exact_energy(temp)
    c = exact_specific_heat(temp)
    m = exact_magnetization(temp)

    print(f"Free energy = {f:.14e}")
    print(f"Energy = {e:.14e}")
    print(f"Specific heat = {c:.14e}")
    print(f"Magnetization = {m:.14e}")
