Developed for the course "Electrónica Física"
University of Granada (UGR)
Academic Year 2025–2026

# MOSFET-SiSiGe-Leakage-Optimization

Simulation and optimization of Si/SiGe superlattice structures for leakage-current reduction in nanoscale NMOS transistors.

## Project Description

This project studies the introduction of Si/SiGe quantum-well structures inside the channel of a 15 nm NMOS transistor in order to reduce source-to-drain tunnelling and leakage current in the OFF state.

The channel is modeled using a one-dimensional effective-mass approximation. Quantum transport is calculated through the Transfer Matrix Method (TMM), while the leakage current is estimated using the Landauer formalism.

The geometrical dimensions of the Si and SiGe regions are optimized using the Sequential Least Squares Programming (SLSQP) algorithm.

## Physical Model

Operating conditions:

- Channel length: 15 nm
- Temperature: 300 K
- OFF state:
  - VGS = 0 V
  - VDS = 0.7 V

Materials:

- Silicon (Si):
  - Effective channel potential: 0.30 eV
- Silicon-Germanium (SiGe):
  - Effective barrier potential: 0.45 eV

Transport model:

- Effective mass approximation
- Transfer Matrix Method
- Landauer current calculation
- Fermi–Dirac statistics

## Simulated Structures

The following configurations are analyzed:

- N = 0 (conventional Si channel)
- N = 3 (3 SiGe barriers)
- N = 5 (5 SiGe barriers)

The optimization algorithm determines the optimal lengths of each Si and SiGe segment while maintaining a fixed total channel length of 15 nm.

## Optimization

Optimization method:

- SLSQP (Sequential Least Squares Programming)

Objective:

- Minimize OFF-state leakage current

Constraints:

- Total channel length = 15 nm
- Positive segment lengths
- Minimum segment length = 0.5 nm
- Maximum segment length = 8 nm

## Generated Results

The code produces:

- Optimized channel geometries
- Potential profiles
- Transmission spectra
- Landauer current estimates
- Leakage-current reduction percentages

## Repository Contents

```text
main.py
README.md
resultados_SiSiGe_perfil_potencial.png
resultados_SiSiGe_transmision.png
resultados_SiSiGe_tabla.png
```

## Authors

- Marcos Poza Marcos
- José Domínguez González
- José Rosa Girona

## References

1. J. H. Davies,
   *The Physics of Low-Dimensional Semiconductors: An Introduction*,
   Cambridge University Press, 1998.

2. nanoHUB.org,
   *Periodic Complex Potential Barrier Tool (PCPBT)*,
   https://nanohub.org/tools/pcpbt

3. D. Kraft,
   *A Software Package for Sequential Quadratic Programming*,
   DLR, 1988.
