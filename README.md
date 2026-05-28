# MOSFET-SiSiGe-Leakage-Optimization

Developed for the course **Electrónica Física**  
University of Granada (UGR)  
Academic Year 2025–2026

---

## Project Overview

This project investigates the reduction of OFF-state leakage current in nanoscale NMOS transistors through the introduction of Si/SiGe heterostructures inside the conduction channel.

The channel is modeled using a one-dimensional quantum transport approach based on the effective mass approximation. Electron transport is computed through the Transfer Matrix Method (TMM) using BenDaniel–Duke boundary conditions, while the leakage current is evaluated through the Landauer formalism combined with Fermi–Dirac statistics.

The geometrical dimensions of the Si and SiGe regions are automatically optimized using the Sequential Least Squares Programming (SLSQP) algorithm in order to minimize OFF-state source-to-drain tunnelling.

---

## Physical Model

### Device Parameters

- NMOS channel length:
  - 15 nm

- Operating regime:
  - OFF state

- Applied voltages:
  - \(V_{GS} = 0\ \mathrm{V}\)
  - \(V_{DS} = 0.7\ \mathrm{V}\)

- Temperature:
  - 358 K

---

## Materials

| Material | Effective Mass | Band Offset |
|---|---|---|
| Si | \(0.19\,m_0\) | \(0.00\ \mathrm{eV}\) |
| SiGe | \(0.15\,m_0\) | \(-0.15\ \mathrm{eV}\) |

Global electrostatic barrier:

\[
U_{\mathrm{OFF}} = 0.40\ \mathrm{eV}
\]

The SiGe regions behave as quantum wells due to the conduction-band offset relative to Si.

---

## Transport Model

The simulation includes:

- Effective mass approximation
- Position-dependent effective masses
- BenDaniel–Duke boundary conditions
- Transfer Matrix Method (TMM)
- Fermi–Dirac statistics
- Landauer current calculation

The transmission coefficient is obtained by solving the stationary Schrödinger equation layer-by-layer across the channel.

The leakage current is calculated through:

\[
I = \frac{2e^2}{h}
\int
\left[
f(E,\mu_S)-f(E,\mu_D)
\right]
T(E)\, dE
\]

where:

- \(T(E)\) is the transmission coefficient
- \(f(E,\mu)\) are the Fermi–Dirac distributions
- \(\mu_S\) and \(\mu_D\) are the source and drain chemical potentials

---

## Simulated Structures

The following channel configurations are studied:

| Configuration | Structure |
|---|---|
| \(N=0\) | Conventional Si channel |
| \(N=3\) | 4 Si regions + 3 SiGe wells |
| \(N=5\) | 6 Si regions + 5 SiGe wells |

The optimization algorithm determines the optimal segment lengths while maintaining:

\[
\sum_i L_i = 15\ \mathrm{nm}
\]

---

## Optimization

### Algorithm

- SLSQP (Sequential Least Squares Programming)

### Objective

- Minimize OFF-state leakage current

### Constraints

- Fixed total channel length
- Positive segment lengths
- Minimum segment size:
  - 0.5 nm
- Maximum segment size:
  - 15 nm

### Numerical Strategy

The optimization is repeated using multiple random initializations in order to reduce trapping in local minima.

---

## Generated Results

The program automatically generates:

- Optimized Si/SiGe geometries
- Potential-energy profiles
- Quantum transmission spectra
- Landauer current estimates
- Leakage-current reduction percentages
- Publication-ready figures

Generated files:

```text
resultados_SiSiGe_tabla2.png
resultados_SiSiGe_perfil_potencial2.png
resultados_SiSiGe_transmision2.png
```

---

## Repository Structure

```text
sige_nmos_optimizer_2.py
README.md
fotos/
├── resultados_SiSiGe_tabla2.png
├── resultados_SiSiGe_perfil_potencial2.png
└── resultados_SiSiGe_transmision2.png
```

---

## Main Features of the Code

- Vectorized Transfer Matrix implementation
- Different effective masses in Si and SiGe
- Automatic constrained optimization
- Numerical stabilization for evanescent states
- Automatic figure generation
- Fully reproducible simulations

---

## Authors

- Marcos Poza Marcos
- José Domínguez González
- José Rosa Girona

---

## References

1. J. H. Davies,  
   *The Physics of Low-Dimensional Semiconductors: An Introduction*,  
   Cambridge University Press, 1998.

2. nanoHUB.org,  
   *Periodic Complex Potential Barrier Tool (PCPBT)*,  
   https://nanohub.org/tools/pcpbt

3. D. Kraft,  
   *A Software Package for Sequential Quadratic Programming*,  
   DLR German Aerospace Center (DLR), 1988.

4. S. Datta,  
   *Electronic Transport in Mesoscopic Systems*,  
   Cambridge University Press, 1995.
