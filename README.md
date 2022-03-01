# TRG_Ising_2D

The tensor renormalization group method (TRG) and the higher-order TRG method (HOTRG) are efficient computational methods to contract the tensor network for classical statistical systems. This repository provides very simple Python codes of TRG and HOTRG for the Ising model on the square lattice.

## Usage

```
./trg.py [chi] [step] [temperature]
./hotrg.py [chi] [step] [temperature]
```

## Requirements

- Python3
- Numpy
- Scipy

## Notes

- The computational cost of TRG and HOTRG scale as O(chi^6) and O(chi^7), respectively.
  For O(chi^5) TRG algorithm, please check our PRE paper.
- The memory usage in both methods is O(chi^4).
- The codes do not support the external magnetic field.

## References

- M. Levin, C. P. Nave: Phys. Rev. Lett. 99, 120601 (2007)
- Z. Y. Xie, et al.: Phys. Rev. B 86, 045139 (2012)
- S. Morita, R. Igarashi, H.-H. Zhao, and N. Kawashima: Phys. Rev. E 97, 033310 (2018)
