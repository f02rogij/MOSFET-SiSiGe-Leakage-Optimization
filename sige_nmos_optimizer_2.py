# ==============================================================================
# Optimización de Estructura Si/SiGe en Canal NMOS (Staircase)
# Método TMM con BenDaniel-Duke + Formalismo de Landauer
# ==============================================================================

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
import warnings
import time

warnings.filterwarnings("ignore")

# ==============================================================================
# CONSTANTES FÍSICAS (SI)
# ==============================================================================
HBAR  = 1.054571817e-34        # ħ  [J·s]
M0    = 9.1093837015e-31       # m₀ [kg]
Q_E   = 1.602176634e-19        # e  [C]
KB_EV = 8.617333262e-5         # k_B [eV/K]
G0    = Q_E**2 / (np.pi * HBAR)    # = 2e²/h ≈ 7.748×10⁻⁵ S

# ==============================================================================
# PARÁMETROS DEL DISPOSITIVO
# ==============================================================================
M_STAR_SI   = 0.19       # Masa efectiva relativa del Si
M_STAR_SIGE = 0.15       # Masa efectiva relativa del SiGe

L_NM     = 15.0      # longitud total del canal [nm]
T_KELVIN = 358.0     # temperatura [K]

# Polarización externa
MU_S     = 0.0       # potencial químico de la fuente μS [eV]
MU_D     = -0.7      # potencial químico del drenador μD [eV]
V_DS     = MU_S - MU_D # Caída de potencial total (0.7 V)

# Definición del perfil de bandas (Estado OFF)
U_OFF       = 0.40   # Altura de la barrera base inducida por la puerta [eV]
OFFSET_SI   = 0.00   # Nivel de referencia para el pozo del Si [eV]
OFFSET_SIGE = -0.15  # Discontinuidad de la banda en SiGe (pozo) [eV]
U_LEAD      = 0.00   # Potencial de los contactos (Source en 0 eV) [eV]

# Integración y optimización
E_MIN = 0.0
E_MAX = 1.0
L_SEG_MIN = 0.5
L_SEG_MAX = 15.0
N_E_OPT  = 500
N_E_PLOT = 2000
N_STARTS = 40
RNG_SEED = 42

SAVE_PATH = Path(__file__).resolve().parent / "fotos"
SAVE_PATH.mkdir(exist_ok=True)

# ==============================================================================
# 1. MÓDULO DE ESTRUCTURA (STAIRCASE)
# ==============================================================================

def build_structure(N, lengths):
    n_seg = 2 * N + 1
    lengths = np.asarray(lengths, dtype=float)
    L_total = lengths.sum()
    
    structure = []
    x_current = 0.0
    
    for i, d_nm in enumerate(lengths):
        is_si = (i % 2 == 0)
        offset = OFFSET_SI if is_si else OFFSET_SIGE
        m_eff  = (M_STAR_SI if is_si else M_STAR_SIGE) * M0
        
        x_mid_nm = x_current + (d_nm / 2.0)
        caida_V = (x_mid_nm / L_total) * V_DS
        U_segmento = U_OFF - caida_V + offset
        
        structure.append((float(U_segmento), float(d_nm) * 1e-9, float(m_eff)))
        x_current += d_nm
        
    return structure

# ==============================================================================
# 2. MATRICES DE TRANSFERENCIA (TMM con BenDaniel-Duke)
# ==============================================================================

def _prop_matrix_batch(k_arr, d_m, m_eff):
    kd = k_arr * d_m
    c  = np.cos(kd)
    s  = np.sin(kd)

    n_E = len(k_arr)
    Nj  = np.zeros((n_E, 2, 2), dtype=complex)

    k_safe = np.where(np.abs(k_arr) < 1e3, 1.0 + 0j, k_arr)
    k_tilde = k_safe / m_eff

    Nj[:, 0, 0] =  c
    Nj[:, 0, 1] =  s / k_tilde
    Nj[:, 1, 0] = -k_tilde * s
    Nj[:, 1, 1] =  c

    tiny = np.abs(k_arr) < 1e3
    if np.any(tiny):
        Nj[tiny, 0, 0] = 1.0
        Nj[tiny, 0, 1] = d_m * m_eff
        Nj[tiny, 1, 0] = 0.0
        Nj[tiny, 1, 1] = 1.0

    return Nj

def _compute_T_batch(E_grid_eV, structure, m_lead=M_STAR_SI*M0, U_lead=U_LEAD):
    n_E  = len(E_grid_eV)
    E_J  = E_grid_eV * Q_E
    UL_J = U_lead * Q_E

    kL = np.sqrt((2.0 * m_lead * (E_J - UL_J) / HBAR**2).astype(complex))
    valid = kL.real > 1e4
    kL_tilde = kL / m_lead

    N_tot = np.broadcast_to(np.eye(2, dtype=complex), (n_E, 2, 2)).copy()

    for U_eV, d_m, m_eff in structure:
        U_J = U_eV * Q_E
        k_j = np.sqrt((2.0 * m_eff * (E_J - U_J) / HBAR**2).astype(complex))
        Nj  = _prop_matrix_batch(k_j, d_m, m_eff)
        N_tot = Nj @ N_tot

    n11, n12 = N_tot[:, 0, 0], N_tot[:, 0, 1]
    n21, n22 = N_tot[:, 1, 0], N_tot[:, 1, 1]

    denom = 1j * kL_tilde * n11 + 1j * kL_tilde * n22 + kL_tilde**2 * n12 - n21
    T_raw = np.where(valid, np.real(4.0 * kL_tilde.real**2 / (np.abs(denom)**2 + 1e-300)), 0.0)
    return np.clip(T_raw, 0.0, 1.0).astype(float)

# ==============================================================================
# 3. FERMI-DIRAC Y LANDAUER
# ==============================================================================

def fermi(E_eV, mu_eV, T_K=T_KELVIN):
    x = np.clip((E_eV - mu_eV) / (KB_EV * T_K), -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(x))

def landauer_current(E_grid, T_E, muS=MU_S, muD=MU_D, T_K=T_KELVIN):
    fS = fermi(E_grid, muS, T_K)
    fD = fermi(E_grid, muD, T_K)
    return float(G0 * np.trapz((fS - fD) * T_E, E_grid))

# ==============================================================================
# 4. OPTIMIZACIÓN
# ==============================================================================

def objective(lengths, N):
    structure = build_structure(N, lengths)
    E_grid    = np.linspace(E_MIN, E_MAX, N_E_OPT)
    T_E       = _compute_T_batch(E_grid, structure)
    return landauer_current(E_grid, T_E)

def optimize_structure(N, seed=RNG_SEED):
    n_seg       = 2 * N + 1
    bounds      = [(L_SEG_MIN, L_SEG_MAX)] * n_seg
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - L_NM}]
    rng         = np.random.default_rng(seed)

    best_I, best_lengths = np.inf, None
    converged_count = 0

    for trial in range(N_STARTS):
        raw = rng.uniform(L_SEG_MIN, L_SEG_MAX, n_seg)
        x0  = raw / raw.sum() * L_NM
        x0  = np.clip(x0, L_SEG_MIN, L_SEG_MAX)
        x0  = x0 / x0.sum() * L_NM

        try:
            res = optimize.minimize(
                objective, x0, args=(N,), method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'ftol': 1e-14, 'maxiter': 2000, 'disp': False}
            )
            if res.success:
                converged_count += 1
                if res.fun < best_I:
                    best_I       = float(res.fun)
                    best_lengths = res.x.copy()
        except Exception:
            pass

    if best_lengths is None:
        best_lengths = np.full(n_seg, L_NM / n_seg)
        best_I       = float(objective(best_lengths, N))
        print(f"      - Fallback a distribución uniforme (0 convergen)")
    else:
        print(f"      - {converged_count}/{N_STARTS} intentos convergidos")

    return best_lengths, best_I

# ==============================================================================
# 5. EXPORTACIÓN Y VISUALIZACIÓN
# ==============================================================================

def export_nanohub_profile_txt(results):
    """Genera un archivo .txt con las capas exactas y TODOS los valores T(E)."""
    file_path = SAVE_PATH / "perfiles_nanohub.txt"
    E_grid = np.linspace(E_MIN, E_MAX, N_E_PLOT)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=== VALORES OPTIMIZADOS PARA SIMULADOR NANOHUB ===\n")
        f.write("Nota: Los contactos (Source/Drain) no están incluidos en estas tablas.\n")
        f.write("Configúralos en nanoHUB como semi-infinitos o dales una longitud grande.\n\n")
        
        # 1. IMPRIMIR LOS PERFILES DE CAPAS
        for N, lengths, _ in results:
            structure = build_structure(N, lengths)
            f.write(f"--- [ PERFIL N={N} ] ---\n")
            f.write(f"{'Capa':<8} | {'Material':<8} | {'Grosor (nm)':<12} | {'Energía U (eV)':<14}\n")
            f.write("-" * 55 + "\n")
            
            for i, (U_eV, d_m, m_eff) in enumerate(structure):
                material = "Si" if i % 2 == 0 else "SiGe"
                d_nm = d_m * 1e9
                f.write(f"Capa {i+1:<3} | {material:<8} | {d_nm:<12.3f} | {U_eV:<14.4f}\n")
            f.write("\n")
            
        # 2. IMPRIMIR TODOS LOS VALORES DE TRANSMISIÓN T(E)
        f.write("=== VALORES DE TRANSMISIÓN T(E) ===\n")
        
        # Calcular T(E) para cada caso y guardarlo en una lista
        T_E_all = []
        headers = ["Energia (eV)"]
        for N, lengths, _ in results:
            structure = build_structure(N, lengths)
            T_E = _compute_T_batch(E_grid, structure)
            T_E_all.append(T_E)
            headers.append(f"T(E) N={N}")
        
        # Cabecera de la tabla T(E)
        header_str = f"{headers[0]:<15} | " + " | ".join([f"{h:<15}" for h in headers[1:]])
        f.write(header_str + "\n")
        f.write("-" * len(header_str) + "\n")
        
        # Escribir las 2000 filas
        for i in range(N_E_PLOT):
            row_str = f"{E_grid[i]:<15.5f} | " + " | ".join([f"{T_E_all[j][i]:<15.4e}" for j in range(len(results))])
            f.write(row_str + "\n")
            
    print(f"   ✓ Perfiles y valores T(E) guardados en el TXT: {file_path}")

def export_and_print_TE(results):
    """Mantiene el CSV por si lo necesitas para graficar rápido."""
    E_grid = np.linspace(E_MIN, E_MAX, N_E_PLOT)
    data_export = np.zeros((N_E_PLOT, len(results) + 1))
    data_export[:, 0] = E_grid
    header = "Energia_[eV]"
    for idx, (N, lengths, _) in enumerate(results):
        structure = build_structure(N, lengths)
        T_E = _compute_T_batch(E_grid, structure)
        data_export[:, idx + 1] = T_E
        header += f",T_N{N}"
    file_path = SAVE_PATH / "datos_transmision_TE.csv"
    np.savetxt(file_path, data_export, delimiter=",", header=header, comments="")
    print(f"   ✓ Curvas de Transmisión T(E) exportadas en CSV: {file_path}")

def _compact_lengths(arr, per_line=3):
    parts = [f"{v:.2f}" for v in arr]
    lines = []
    for i in range(0, len(parts), per_line):
        lines.append(", ".join(parts[i:i + per_line]))
    return "\n".join(lines)

def build_table(ax, results):
    ax.axis("off")
    I_ref = results[0][2]

    col_labels = [
            "N\n(SiGe segs.)", "Longitudes Si [nm]", "Longitudes SiGe [nm]",
            "I_N [nA]\n(2e²/h)·∫T·Δf·dε", "Reducción\nvs N=0 (%)"
        ]

    rows = []
    for N, lengths, I_N in results:
        si_l = lengths[0::2]
        sg_l = lengths[1::2]
        red  = (1.0 - I_N / I_ref) * 100.0 if I_ref > 1e-30 else 0.0

        rows.append([
            str(N), _compact_lengths(si_l, per_line=3),
            _compact_lengths(sg_l, per_line=3) if len(sg_l) > 0 else "-",
            f"{I_N * 1e9:.4f}", f"{red:.3f} %"
        ])

    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.2)
    tbl.scale(1.0, 2.7)

    for j in range(len(col_labels)):
        cell = tbl[(0, j)]
        cell.set_facecolor("#1a3464")
        cell.set_text_props(color="white", fontweight="bold")

    row_bgs = ["#d9e6f5", "#f4f8ff", "#d9e6f5"]
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            tbl[(i, j)].set_facecolor(row_bgs[(i - 1) % len(row_bgs)])

def plot_potential(ax, results):
    palette = ["#1a6faf", "#cc4125", "#2e7d32"]
    labels  = ["N=0  (1 Si)", "N=3  (4 Si + 3 SiGe)", "N=5  (6 Si + 5 SiGe)"]
    lws     = [2.8, 2.2, 1.8]

    for idx, (N, lengths, _) in enumerate(results):
        structure = build_structure(N, lengths)
        n_seg = len(structure)
        edges = np.concatenate(([0.0], np.cumsum(lengths)))

        x_plot, U_plot = [], []
        for i in range(n_seg):
            U = structure[i][0]
            x_plot += [edges[i], edges[i + 1]]
            U_plot += [U, U]

        x_full = [-1.5, 0]      + x_plot + [L_NM, L_NM + 1.5]
        U_full = [U_LEAD, U_LEAD] + U_plot + [MU_D, MU_D] 

        ax.plot(x_full, U_full, color=palette[idx], lw=lws[idx],
                label=labels[idx], zorder=3 + idx)

    ax.axhline(MU_S, color="#003366", ls=":", lw=1.2, alpha=0.75, label=f"μS = {MU_S:.1f} eV")
    ax.axhline(MU_D, color="#8b0000", ls=":", lw=1.2, alpha=0.75, label=f"μD = {MU_D:.1f} eV")

    for xb in [0.0, L_NM]:
        ax.axvline(xb, color="dimgray", ls="--", lw=1.0, alpha=0.55, zorder=2)

    ax.set_xlabel("Posición x [nm]", fontsize=11)
    ax.set_ylabel("Energía Potencial U(x) [eV]", fontsize=11)
    ax.set_xlim(-1.5, L_NM + 1.5)
    ax.legend(fontsize=8.0, loc="upper right", ncol=1, framealpha=0.85)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_facecolor("#f7f9fc")

def plot_transmission(ax, results):
    palette = ["#1a6faf", "#cc4125", "#2e7d32"]
    labels  = ["N=0  (1 Si)", "N=3  (4 Si + 3 SiGe)", "N=5  (6 Si + 5 SiGe)"]
    lws     = [2.8, 2.2, 1.8]

    E_grid = np.linspace(E_MIN, E_MAX, N_E_PLOT)
    fermi_w = fermi(E_grid, MU_S) - fermi(E_grid, MU_D)
    ax.fill_between(E_grid, fermi_w, alpha=0.22, color="#e57373", label=r"Ventana Fermi", zorder=1)

    for idx, (N, lengths, _) in enumerate(results):
        structure = build_structure(N, lengths)
        T_E = _compute_T_batch(E_grid, structure)
        ax.plot(E_grid, T_E, color=palette[idx], lw=lws[idx], label=labels[idx], zorder=3 + idx)

    ax.set_xlabel("Energía E [eV]", fontsize=11)
    ax.set_ylabel("Coeficiente de transmisión T(E)", fontsize=11)
    ax.set_xlim(E_MIN, E_MAX)
    ax.set_ylim(-0.02, 1.08)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_facecolor("#f7f9fc")

# ==============================================================================
# 6. FUNCIÓN PRINCIPAL
# ==============================================================================

def main():
    t_start = time.time()

    print("=================================================================")
    print("   Optimizador Si/SiGe OFF NMOS (Staircase Model)")
    print("=================================================================\n")

    results = []

    print("-> [N=0] Canal Si uniforme (referencia)...")
    l0 = np.array([L_NM])
    s0 = build_structure(0, l0)
    E_plot = np.linspace(E_MIN, E_MAX, N_E_PLOT)
    T0 = _compute_T_batch(E_plot, s0)
    I0 = landauer_current(E_plot, T0)
    print(f"         I_0 = {I0:.8f}")
    results.append((0, l0, I0))

    print("-> [N=3] Optimizando (4 Si + 3 SiGe)...")
    l3, I3 = optimize_structure(3)
    results.append((3, l3, I3))
    print(f"         I_3 = {I3:.8f} (Reduccion {(1-I3/I0)*100:.2f}%)")

    print("-> [N=5] Optimizando (6 Si + 5 SiGe)...")
    l5, I5 = optimize_structure(5)
    results.append((5, l5, I5))
    print(f"         I_5 = {I5:.8f} (Reduccion {(1-I5/I0)*100:.2f}%)")

    print("\n-> Generando archivos y figuras...")

    # Exportar datos a TXT y CSV
    export_nanohub_profile_txt(results)
    export_and_print_TE(results)

    # Figura 1: Tabla
    fig_tbl, ax_tbl = plt.subplots(figsize=(12, 4))
    build_table(ax_tbl, results)
    fig_tbl.savefig(SAVE_PATH / "resultados_tabla.png", dpi=150, bbox_inches="tight")

    # Figura 2: Perfil U(x)
    fig_pot, ax_pot = plt.subplots(figsize=(9, 5.5))
    plot_potential(ax_pot, results)
    fig_pot.savefig(SAVE_PATH / "resultados_perfil_potencial.png", dpi=150, bbox_inches="tight")

    # Figura 3: Transmisión T(E)
    fig_tra, ax_tra = plt.subplots(figsize=(9, 5.5))
    plot_transmission(ax_tra, results)
    fig_tra.savefig(SAVE_PATH / "resultados_transmision.png", dpi=150, bbox_inches="tight")

    print(f"\n   - Proceso completado en: {time.time()-t_start:.1f} s\n")
    
    # Mostrar por pantalla
    plt.show()

if __name__ == "__main__":
    main()