# TRG_Ising_2D

The tensor renormalization group method (TRG) and its variants are efficient computational methods for contracting a tensor network of classical statistical systems. This repository provides very simple Python codes of TRG, higher-order TRG (HOTRG) and bond-weighted TRG (BWTRG) for the Ising model on the square lattice.

## Usage

```
python3 src/trg.py [chi] [step] [temperature]
python3 src/hotrg.py [chi] [step] [temperature]
python3 src/bwtrg.py [-k k] [chi] [step] [temperature]
```

Use `-h` option for more details.

## Requirements

- Python3
- NumPy
- SciPy (`scipy.linalg`, `scipy.special`, `scipy.integrate`)

## Notes

- The computational cost of TRG and HOTRG scale as O(chi^6) and O(chi^7), respectively.
  For O(chi^5) TRG algorithm, please check our PRE paper.
- The memory usage in both methods is O(chi^4).
  In HOTRG, the loop blocking technique is used to reduce memory usage.
- The codes do not support the external magnetic field.
- Animations of TRG and HOTRG are available on https://smorita.github.io/TN_animation/.
- We use [uv](https://docs.astral.sh/uv/) to manage this project.

## References

- M. Levin, C. P. Nave, Phys. Rev. Lett. 99, 120601 (2007)
- Z. Y. Xie, et al., Phys. Rev. B 86, 045139 (2012)
- D. Adachi, T. Okkubo and S. Todo, Phys. Rev. B 105, L060402 (2022)
- S. Morita, R. Igarashi, H.-H. Zhao, and N. Kawashima, Phys. Rev. E 97, 033310 (2018)
