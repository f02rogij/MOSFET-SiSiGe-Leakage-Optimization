# MOSFET-SiSiGe-Leakage-Optimization

Developed for the course **Electrónica Física**  
University of Granada (UGR)  
Academic Year 2025–2026

---

## Project Overview

This project investigates the reduction of OFF-state leakage current in nanoscale NMOS transistors through the introduction of Si/SiGe superlattice structures inside the channel.

The device is modeled using a one-dimensional quantum transport approach based on the effective mass approximation. Electron transmission through the channel is computed using the Transfer Matrix Method (TMM), while the leakage current is evaluated using the Landauer formalism with Fermi–Dirac statistics.

The geometrical dimensions of the Si and SiGe regions are automatically optimized using the Sequential Least Squares Programming (SLSQP) algorithm.

---

## Physical Model

### Device Parameters

- NMOS channel length: 15 nm
- OFF-state operation:
  - \(V_{GS} = 0\) V
  - \(V_{DS} = 0.7\) V
- Temperature:
  - 358 K (85 °C)

### Materials

| Material | Effective Potential | Effective Mass |
|---|---|---|
| Si | 0.30 eV | \(0.19\,m_0\) |
| SiGe | 0.45 eV | \(0.22\,m_0\) |

The SiGe regions are modeled as conduction-band barriers with an effective offset of:

\[
\Delta E_c = 0.15\ \mathrm{eV}
\]

relative to the Si regions.

---

## Quantum Transport Model

The simulation includes:

- Effective mass approximation
- Position-dependent effective masses
- BenDaniel–Duke boundary conditions
- Transfer Matrix Method (TMM)
- Continuous \(V_{DS}\) potential ramp
- Landauer transport formalism
- Fermi–Dirac statistics

The drain-source voltage is not modeled as a simple step potential. Instead, the channel is subdivided into multiple thin slices in order to reproduce a continuous linear voltage drop along the transport direction.

This ensures a physically consistent comparison between the conventional channel and the Si/SiGe superlattice structures.

---

## Simulated Structures

The following configurations are analyzed:

| Configuration | Structure |
|---|---|
| \(N=0\) | Conventional Si channel |
| \(N=3\) | 4 Si regions + 3 SiGe barriers |
| \(N=5\) | 6 Si regions + 5 SiGe barriers |

The optimization algorithm determines the optimal segment lengths while enforcing:

\[
\sum_i L_i = 15\ \mathrm{nm}
\]

---

## Optimization

### Algorithm

- SLSQP (Sequential Least Squares Programming)

### Objective

Minimize the OFF-state leakage current:

\[
I_N = \frac{2e^2}{h}
\int
\left[
f(E,\mu_S)-f(E,\mu_D)
\right]
T(E)\,dE
\]

### Constraints

- Total channel length fixed to 15 nm
- Positive segment lengths
- Minimum segment size: 0.5 nm
- Multiple random initializations to reduce local-minimum trapping

---

## Generated Results

The program automatically generates:

- Optimized Si/SiGe geometries
- Potential profiles \(U(x)\)
- Quantum transmission spectra \(T(E)\)
- Landauer current estimates
- Leakage-current reduction percentages
- Publication-ready figures

---

## Repository Structure

```text
sige_nmos_optimizer_3.py
README.md
fotos/
├── resultados_SiSiGe_tabla.png
├── resultados_SiSiGe_perfil_potencial.png
└── resultados_SiSiGe_transmision.png
```

---

## Main Features of the Code

- Vectorized TMM implementation
- Different effective masses in each material
- Continuous electrostatic potential profile
- Automatic constrained optimization
- Numerical stability for evanescent states
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
   DLR, 1988.
