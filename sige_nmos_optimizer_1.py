"""
╔══════════════════════════════════════════════════════════════════════════╗
║        Optimización de Estructura Si/SiGe en Canal NMOS                  ║
║        Método de Matrices de Transferencia + Formalismo de Landauer      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Busca automáticamente las longitudes óptimas de regiones Si y SiGe      ║
║  alternadas dentro del canal para minimizar la corriente de fuga OFF.    ║
║                                                                          ║
║  Casos:                                                                  ║
║    N=0 → 1 Si  (canal uniforme, referencia)                              ║
║    N=3 → 4 Si + 3 SiGe  (7 segmentos)                                    ║
║    N=5 → 6 Si + 5 SiGe  (11 segmentos)                                   ║
║                                                                          ║
║  Física:                                                                 ║
║    T(E): Método TMM — matrices de propagación 1D por tramos              ║
║    I_N : Fórmula de Landauer con distribuciones de Fermi-Dirac           ║
║    Opt : scipy.optimize.minimize (SLSQP) con arranques aleatorios        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════
# IMPORTACIONES
# ══════════════════════════════════════════════════════════════════════════
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
import warnings
import time

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS (SI)
# ══════════════════════════════════════════════════════════════════════════
HBAR  = 1.054571817e-34        # ħ  [J·s]
M0    = 9.1093837015e-31       # m₀ [kg]
Q_E   = 1.602176634e-19        # e  [C]
KB_EV = 8.617333262e-5         # k_B [eV/K]
G0 = Q_E**2 / (np.pi * HBAR)   # = 2e²/h ≈ 7.748×10⁻⁵ S

# ══════════════════════════════════════════════════════════════════════════
# PARÁMETROS DEL DISPOSITIVO — Modificar aquí fácilmente
# ══════════════════════════════════════════════════════════════════════════
M_STAR   = 0.19      # masa efectiva relativa m*/m₀
M_EFF    = M_STAR * M0   # masa efectiva [kg]

L_NM     = 15.0      # longitud total del canal [nm]
T_KELVIN = 358.0     # temperatura [K]
MU_S     = 0.0       # potencial químico de la fuente μS [eV]
MU_D     = -0.7      # potencial químico del drenador μD [eV]
U_SI     = -0.30     # posición BC del Si  [eV]
U_SIGE   = -0.45     # posición BC del SiGe [eV]
U_LEAD   = 0.00      # potencial de los contactos (= Si) [eV]

# Rango de integración en energía
E_MIN    = 0.0       # [eV]
E_MAX    = 1.0       # [eV]

# Restricciones geométricas de cada segmento
L_SEG_MIN = 0.5      # longitud mínima [nm]
L_SEG_MAX = 15.0     # longitud máxima [nm]

# Resolución de energía
N_E_OPT  = 500       # puntos durante la optimización (velocidad)
N_E_PLOT = 2000      # puntos para las gráficas finales (resolución)

# Optimización
N_STARTS = 40        # intentos con puntos iniciales aleatorios

# Semilla para reproducibilidad
RNG_SEED = 42

# Directorio de salida de imágenes
SAVE_PATH = Path(__file__).resolve().parent / "fotos"
SAVE_PATH.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# 1. MÓDULO DE ESTRUCTURA
# ══════════════════════════════════════════════════════════════════════════

def build_structure(N, lengths, U_Si_val=None, U_SiGe_val=None):
    """
    Construye la estructura del canal como lista de pares (U [eV], d [m]).

    La alternancia es siempre:  Si – SiGe – Si – SiGe – … – Si
    (empieza y termina en Si).

    Parámetros
    ----------
    N          : int — número de segmentos SiGe
    lengths    : array-like de longitudes [nm], tamaño 2N+1
    U_Si_val   : float [eV] — potencial del Si   (None → U_SI global)
    U_SiGe_val : float [eV] — potencial del SiGe (None → U_SIGE global)

    Retorna
    -------
    structure : list de tuplas (U_eV [float], d_m [float])
    """
    if U_Si_val   is None: U_Si_val   = U_SI
    if U_SiGe_val is None: U_SiGe_val = U_SIGE

    n_seg = 2 * N + 1
    lengths = np.asarray(lengths, dtype=float)

    if len(lengths) != n_seg:
        raise ValueError(
            f"build_structure: se esperan {n_seg} longitudes para N={N}, "
            f"pero se recibieron {len(lengths)}."
        )

    structure = []
    for i, d_nm in enumerate(lengths):
        U = U_Si_val if (i % 2 == 0) else U_SiGe_val   # par=Si, impar=SiGe
        structure.append((float(U), float(d_nm) * 1e-9))  # nm → m
    return structure


# ══════════════════════════════════════════════════════════════════════════
# 2. MATRICES DE TRANSFERENCIA (TMM)
# ══════════════════════════════════════════════════════════════════════════
#
# Ecuación de Schrödinger 1D con potencial constante por tramos:
#
#   ψ_j(x) = A_j e^{ik_j x} + B_j e^{-ik_j x}
#   k_j = sqrt(2m*(E - U_j)) / ħ   (complejo si E < U_j)
#
# Matriz de propagación de la capa j (mapea estado en x_L → x_R):
#
#   N_j = ⎡  cos(k_j d_j)       sin(k_j d_j)/k_j ⎤
#          ⎣ -k_j sin(k_j d_j)   cos(k_j d_j)     ⎦
#
# Matriz total (capas de izquierda a derecha):
#   N_tot = N_n @ N_{n-1} @ … @ N_1
#
# Coeficiente de transmisión (derivado de condiciones de contorno):
#
#   T(E) = 4 k_L k_R / |i k_R n₁₁ + i k_L n₂₂ + k_L k_R n₁₂ − n₂₁|²
#
# donde k_L = k_R = sqrt(2m(E − U_lead)) / ħ  (contactos Si simétricos).
#
# Verificación: para N=0 (canal Si uniforme), N_tot = N_Si(L),
#   denom = 2ik e^{−ikL}  ⟹  T = 4k²/(4k²) = 1  ✓
# ──────────────────────────────────────────────────────────────────────────

def _prop_matrix_batch(k_arr, d_m):
    """
    Calcula las matrices de propagación N_j para un array de k.

    Parámetros
    ----------
    k_arr : ndarray complex (n_E,) — vectores de onda [m⁻¹]
    d_m   : float — espesor de la capa [m]

    Retorna
    -------
    Nj : ndarray complex (n_E, 2, 2)
    """
    kd = k_arr * d_m
    c  = np.cos(kd)
    s  = np.sin(kd)

    n_E = len(k_arr)
    Nj  = np.zeros((n_E, 2, 2), dtype=complex)

    # Denominador seguro para sin/k (evita 0/0 cuando k≈0)
    k_safe = np.where(np.abs(k_arr) < 1e3, 1.0 + 0j, k_arr)

    Nj[:, 0, 0] =  c
    Nj[:, 0, 1] =  s / k_safe
    Nj[:, 1, 0] = -k_safe * s
    Nj[:, 1, 1] =  c

    # Límite k→0: N_j → [[1, d], [0, 1]]
    tiny = np.abs(k_arr) < 1e3
    if np.any(tiny):
        Nj[tiny, 0, 0] = 1.0
        Nj[tiny, 0, 1] = d_m
        Nj[tiny, 1, 0] = 0.0
        Nj[tiny, 1, 1] = 1.0

    return Nj


def _compute_T_batch(E_grid_eV, structure, m_eff=None, U_lead=None):
    """
    Calcula T(E) de forma vectorizada para todo el grid de energías.

    Versión eficiente de transfer_matrix_transmission usando numpy broadcasting
    para procesar todas las energías en paralelo con matmul por lotes.

    Parámetros
    ----------
    E_grid_eV : ndarray (n_E,) [eV]
    structure  : list de (U_eV, d_m)
    m_eff      : float [kg]  (None → M_EFF global)
    U_lead     : float [eV]  (None → U_LEAD global)

    Retorna
    -------
    T_arr : ndarray float (n_E,), valores en [0, 1]
    """
    if m_eff  is None: m_eff  = M_EFF
    if U_lead is None: U_lead = U_LEAD

    n_E  = len(E_grid_eV)
    E_J  = E_grid_eV * Q_E
    UL_J = U_lead * Q_E

    # Vector de onda del contacto
    kL   = np.sqrt((2.0 * m_eff * (E_J - UL_J) / HBAR**2).astype(complex))
    # Energías para las que el contacto es propagante (k real > 0)
    valid = kL.real > 1e4

    # Matriz de transferencia total: (n_E, 2, 2)
    N_tot = np.broadcast_to(np.eye(2, dtype=complex), (n_E, 2, 2)).copy()

    for U_eV, d_m in structure:
        U_J  = U_eV * Q_E
        k_j  = np.sqrt((2.0 * m_eff * (E_J - U_J) / HBAR**2).astype(complex))
        Nj   = _prop_matrix_batch(k_j, d_m)
        # Multiplicación de matrices por lotes: N_tot = N_j @ N_tot
        N_tot = Nj @ N_tot

    n11 = N_tot[:, 0, 0]
    n12 = N_tot[:, 0, 1]
    n21 = N_tot[:, 1, 0]
    n22 = N_tot[:, 1, 1]

    # Fórmula de transmisión (contactos simétricos k_L = k_R = kL)
    denom = 1j * kL * n11 + 1j * kL * n22 + kL**2 * n12 - n21
    T_raw = np.where(
        valid,
        np.real(4.0 * kL.real**2 / (np.abs(denom)**2 + 1e-300)),
        0.0
    )
    return np.clip(T_raw, 0.0, 1.0).astype(float)


def transfer_matrix_transmission(E_eV, structure, m_eff=None, U_lead=None):
    """
    Calcula T(E) para un único valor de energía (versión escalar).

    Implementa el método TMM con la derivación analítica:

      denom = i·k_R·n₁₁ + i·k_L·n₂₂ + k_L·k_R·n₁₂ − n₂₁
      T     = 4·k_L·k_R / |denom|²

    Parámetros
    ----------
    E_eV      : float — energía [eV]
    structure : list de (U_eV, d_m) — estructura del canal
    m_eff     : float [kg]  (None → M_EFF global)
    U_lead    : float [eV]  (None → U_LEAD global)

    Retorna
    -------
    T : float en [0, 1]
    """
    if m_eff  is None: m_eff  = M_EFF
    if U_lead is None: U_lead = U_LEAD

    E_J  = E_eV  * Q_E
    UL_J = U_lead * Q_E

    kL = np.sqrt(complex(2.0 * m_eff * (E_J - UL_J) / HBAR**2))
    if kL.real < 1e4:         # modo evanescente en el contacto
        return 0.0

    # Acumular matriz total de izquierda a derecha
    N_tot = np.eye(2, dtype=complex)
    for U_eV, d_m in structure:
        U_J  = U_eV * Q_E
        k_j  = np.sqrt(complex(2.0 * m_eff * (E_J - U_J) / HBAR**2))
        kd   = k_j * d_m

        if abs(k_j) < 1e4:    # límite k→0
            Nj = np.array([[1.0, d_m], [0.0, 1.0]], dtype=complex)
        else:
            c, s = np.cos(kd), np.sin(kd)
            Nj   = np.array([[c, s / k_j], [-k_j * s, c]], dtype=complex)

        N_tot = Nj @ N_tot

    n11, n12 = N_tot[0, 0], N_tot[0, 1]
    n21, n22 = N_tot[1, 0], N_tot[1, 1]

    denom = 1j * kL * n11 + 1j * kL * n22 + kL**2 * n12 - n21
    T = float(np.real(4.0 * kL.real**2 / (abs(denom)**2 + 1e-300)))
    return max(0.0, min(1.0, T))


# ══════════════════════════════════════════════════════════════════════════
# 3. DISTRIBUCIONES DE FERMI-DIRAC Y CORRIENTE DE LANDAUER
# ══════════════════════════════════════════════════════════════════════════

def fermi(E_eV, mu_eV, T_K=None):
    """
    Distribución de Fermi-Dirac.

      f(E, μ) = 1 / (1 + exp((E − μ) / (k_B T)))

    Parámetros
    ----------
    E_eV  : float o ndarray [eV] — energía
    mu_eV : float [eV] — potencial químico
    T_K   : float [K]  (None → T_KELVIN global)

    Retorna
    -------
    f : float o ndarray, valores en (0, 1)
    """
    if T_K is None: T_K = T_KELVIN
    x = np.clip((E_eV - mu_eV) / (KB_EV * T_K), -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(x))


def landauer_current(E_grid, T_E, muS=None, muD=None, T_K=None):
    """
    Corriente de Landauer en Amperes:

      I_N = (2e²/h) · ∫ [f(E,μS) − f(E,μD)] · T_N(E) dE   [A]

    El prefactor G₀ = 2e²/h es el cuanto de conductancia (≈ 77.48 μS).
    La integral se evalúa en eV; el factor G₀ convierte eV → A.

    Parámetros
    ----------
    E_grid : ndarray [eV]
    T_E    : ndarray — coeficiente de transmisión
    muS    : float [eV]  (None → MU_S)
    muD    : float [eV]  (None → MU_D)
    T_K    : float [K]   (None → T_KELVIN)

    Retorna
    -------
    I : float [A]
    """
    if muS is None: muS = MU_S
    if muD is None: muD = MU_D
    if T_K is None: T_K = T_KELVIN
    fS = fermi(E_grid, muS, T_K)
    fD = fermi(E_grid, muD, T_K)
    return float(G0 * np.trapezoid((fS - fD) * T_E, E_grid))


# ══════════════════════════════════════════════════════════════════════════
# 4. FUNCIÓN OBJETIVO Y OPTIMIZACIÓN
# ══════════════════════════════════════════════════════════════════════════

def objective(lengths, N, U_Si_val=None, U_SiGe_val=None):
    """
    Función objetivo para scipy.optimize: retorna I_N (a minimizar).

    Construye la estructura, calcula T(E) y aplica la fórmula de Landauer.
    Usa N_E_OPT puntos para mayor velocidad durante la búsqueda.

    Parámetros
    ----------
    lengths    : ndarray [nm] — longitudes de los 2N+1 segmentos
    N          : int — número de segmentos SiGe
    U_Si_val   : float [eV]  (None → U_SI global)
    U_SiGe_val : float [eV]  (None → U_SIGE global)

    Retorna
    -------
    I : float — valor proporcional a la corriente OFF
    """
    if U_Si_val   is None: U_Si_val   = U_SI
    if U_SiGe_val is None: U_SiGe_val = U_SIGE

    structure = build_structure(N, lengths, U_Si_val, U_SiGe_val)
    E_grid    = np.linspace(E_MIN, E_MAX, N_E_OPT)
    T_E       = _compute_T_batch(E_grid, structure)
    return landauer_current(E_grid, T_E)


def optimize_structure(N, U_Si_val=None, U_SiGe_val=None, seed=RNG_SEED):
    """
    Optimiza las longitudes de segmentos para minimizar la corriente OFF.

    Algoritmo: SLSQP con N_STARTS puntos iniciales aleatorios.

    Restricción de igualdad:  Σ lengths = L_NM  (= 15 nm)
    Cotas:                    L_SEG_MIN ≤ l_i ≤ L_SEG_MAX

    Los puntos iniciales se generan como distribuciones aleatorias
    uniformes re-normalizadas para cumplir la restricción de suma.

    Parámetros
    ----------
    N          : int — número de segmentos SiGe
    U_Si_val   : float [eV]  (None → U_SI)
    U_SiGe_val : float [eV]  (None → U_SIGE)
    seed       : int — semilla aleatoria

    Retorna
    -------
    best_lengths : ndarray [nm] — longitudes óptimas
    best_I       : float — corriente mínima encontrada
    """
    if U_Si_val   is None: U_Si_val   = U_SI
    if U_SiGe_val is None: U_SiGe_val = U_SIGE

    n_seg       = 2 * N + 1
    bounds      = [(L_SEG_MIN, L_SEG_MAX)] * n_seg
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - L_NM}]
    rng         = np.random.default_rng(seed)

    best_I, best_lengths = np.inf, None
    converged_count = 0

    for trial in range(N_STARTS):
        # Punto inicial: distribución aleatoria que suma L_NM
        raw = rng.uniform(L_SEG_MIN, L_SEG_MAX, n_seg)
        x0  = raw / raw.sum() * L_NM
        x0  = np.clip(x0, L_SEG_MIN, L_SEG_MAX)
        x0  = x0 / x0.sum() * L_NM   # re-normalizar tras clip

        try:
            res = optimize.minimize(
                objective,
                x0,
                args=(N, U_Si_val, U_SiGe_val),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-14, 'maxiter': 2000, 'disp': False}
            )
            if res.success:
                converged_count += 1
                if res.fun < best_I:
                    best_I       = float(res.fun)
                    best_lengths = res.x.copy()
        except Exception:
            pass

    # Fallback: distribución uniforme si ningún intento converge
    if best_lengths is None:
        best_lengths = np.full(n_seg, L_NM / n_seg)
        best_I       = float(objective(best_lengths, N))
        print(f"      ⚠ Fallback a distribución uniforme (0 convergen de {N_STARTS})")
    else:
        print(f"      ✓ {converged_count}/{N_STARTS} intentos convergidos")

    return best_lengths, best_I


# ══════════════════════════════════════════════════════════════════════════
# 5. VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════

def _compact_lengths(arr, per_line=3):
    """Formatea un array de floats en cadenas multi-línea (per_line valores/línea)."""
    parts = [f"{v:.2f}" for v in arr]
    lines = []
    for i in range(0, len(parts), per_line):
        lines.append(", ".join(parts[i:i + per_line]))
    return "\n".join(lines)


def build_table(ax, results):
    """
    Dibuja la tabla de resultados en el Axes proporcionado.

    Columnas: N_SiGe | Longitudes Si | Longitudes SiGe | I_N | Reducción%

    Parámetros
    ----------
    ax      : matplotlib Axes (debe estar con axis('off'))
    results : list de tuplas (N, lengths_nm, I_N)
    """
    ax.axis("off")
    I_ref = results[0][2]   # corriente de referencia (N=0)

    col_labels = [
            "N\n(SiGe segs.)",
            "Longitudes Si [nm]",
            "Longitudes SiGe [nm]",
            "I_N [nA]\n(2e²/h)·∫T·Δf·dε",
            "Reducción\nvs N=0 (%)"
        ]

    rows = []
    for N, lengths, I_N in results:
        n_seg = 2 * N + 1
        si_l  = lengths[0::2]
        sg_l  = lengths[1::2]
        red   = (1.0 - I_N / I_ref) * 100.0 if I_ref > 1e-30 else 0.0

        rows.append([
            str(N),
            _compact_lengths(si_l, per_line=3),
            _compact_lengths(sg_l, per_line=3) if len(sg_l) > 0 else "—",
            f"{I_N * 1e9:.4f}",      # A → nA
            f"{red:.3f} %"
        ])

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.2)
    tbl.scale(1.0, 2.7)

    # Estilo del encabezado
    HEADER_COLOR = "#1a3464"
    for j in range(len(col_labels)):
        cell = tbl[(0, j)]
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)

    # Colores alternos de filas
    row_bgs = ["#d9e6f5", "#f4f8ff", "#d9e6f5"]
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            tbl[(i, j)].set_facecolor(row_bgs[(i - 1) % len(row_bgs)])


def plot_potential(ax, results, U_Si_val=None, U_SiGe_val=None):
    """
    Dibuja el perfil de potencial U(x) para los tres casos optimizados.
    Sin sombreado de bandas, sin título.

    Parámetros
    ----------
    ax      : matplotlib Axes
    results : list de (N, lengths_nm, I_N)
    """
    if U_Si_val   is None: U_Si_val   = U_SI
    if U_SiGe_val is None: U_SiGe_val = U_SIGE

    palette = ["#1a6faf", "#cc4125", "#2e7d32"]
    labels  = ["N=0  (1 Si)", "N=3  (4 Si + 3 SiGe)", "N=5  (6 Si + 5 SiGe)"]
    lws     = [2.8, 2.2, 1.8]

    # Trazar perfil escalonado de cada caso (sin sombreado de fondo)
    for idx, (N, lengths, _) in enumerate(results):
        n_seg = 2 * N + 1
        edges = np.concatenate(([0.0], np.cumsum(lengths)))

        x_plot, U_plot = [], []
        for i in range(n_seg):
            U = U_Si_val if (i % 2 == 0) else U_SiGe_val
            x_plot += [edges[i], edges[i + 1]]
            U_plot += [U, U]

        # Extender a los contactos (Si)
        x_full = [-1.5, 0]      + x_plot + [L_NM, L_NM + 1.5]
        U_full = [U_LEAD, U_LEAD] + U_plot + [U_LEAD, U_LEAD]

        ax.plot(x_full, U_full, color=palette[idx], lw=lws[idx],
                label=labels[idx], zorder=3 + idx)

    # Niveles de potencial químico
    ax.axhline(MU_S, color="#003366", ls=":", lw=1.2, alpha=0.75,
               label=f"μS = {MU_S:.1f} eV (Fuente)")
    ax.axhline(MU_D, color="#8b0000", ls=":", lw=1.2, alpha=0.75,
               label=f"μD = {MU_D:.1f} eV (Drenador)")

    # Fronteras del canal
    for xb in [0.0, L_NM]:
        ax.axvline(xb, color="dimgray", ls="--", lw=1.0, alpha=0.55, zorder=2)

    ax.text(0.01, 0.96, "← Fuente", transform=ax.transAxes,
            fontsize=8, color="dimgray", va="top")
    ax.text(0.99, 0.96, "Drenador →", transform=ax.transAxes,
            fontsize=8, color="dimgray", va="top", ha="right")

    ax.set_xlabel("Posición x [nm]", fontsize=11)
    ax.set_ylabel("Potencial U(x) [eV]", fontsize=11)
    ax.set_xlim(-1.5, L_NM + 1.5)
    U_margin = 0.08
    ax.set_ylim(min(U_SiGe_val, MU_D) - U_margin, max(U_Si_val, MU_S) + U_margin + 0.05)
    ax.legend(fontsize=8.0, loc="lower right", ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_facecolor("#f7f9fc")


def plot_transmission(ax, results):
    """
    Dibuja T(E) para los tres casos y la ventana de Fermi  [f(μS) − f(μD)].

    Parámetros
    ----------
    ax      : matplotlib Axes
    results : list de (N, lengths_nm, I_N)
    """
    palette = ["#1a6faf", "#cc4125", "#2e7d32"]
    labels  = ["N=0  (1 Si)", "N=3  (4 Si + 3 SiGe)", "N=5  (6 Si + 5 SiGe)"]
    lws     = [2.8, 2.2, 1.8]

    E_grid = np.linspace(E_MIN, E_MAX, N_E_PLOT)

    # Ventana de Fermi
    fS     = fermi(E_grid, MU_S)
    fD     = fermi(E_grid, MU_D)
    fermi_w = fS - fD
    ax.fill_between(E_grid, fermi_w, alpha=0.22, color="#e57373",
                    label=r"Ventana Fermi  $f(\mu_S) - f(\mu_D)$", zorder=1)

    for idx, (N, lengths, _) in enumerate(results):
        structure = build_structure(N, lengths)
        T_E = _compute_T_batch(E_grid, structure)
        ax.plot(E_grid, T_E, color=palette[idx], lw=lws[idx],
                label=labels[idx], zorder=3 + idx)

    ax.set_xlabel("Energía E [eV]", fontsize=11)
    ax.set_ylabel("Coeficiente de transmisión T(E)", fontsize=11)
    ax.set_xlim(E_MIN, E_MAX)
    ax.set_ylim(-0.02, 1.08)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_facecolor("#f7f9fc")


# ══════════════════════════════════════════════════════════════════════════
# 6. FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

def main():
    """
    Flujo principal:
      1. Calcula I₀ para el canal Si uniforme (N=0, referencia).
      2. Optimiza N=3: encuentra las longitudes que minimizan I₃.
      3. Optimiza N=5: ídem para I₅.
      4. Genera la figura PNG con tabla, U(x) y T(E).
    """
    t_start = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Optimizador Si/SiGe — Corriente OFF en Canal NMOS          ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  m* = {M_STAR} m₀  ·  L = {L_NM} nm  ·  T = {T_KELVIN:.0f} K                  ║")
    print(f"║  μS = {MU_S} eV    ·  μD = {MU_D} eV                              ║")
    print(f"║  U_Si = {U_SI} eV  ·  U_SiGe = {U_SIGE} eV                        ║")
    print(f"║  E ∈ [{E_MIN}, {E_MAX}] eV  ·  N_starts = {N_STARTS}                         ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    results = []   # lista de (N, optimal_lengths [nm], I_N)
    E_plot  = np.linspace(E_MIN, E_MAX, N_E_PLOT)

    # ─── N=0: caso de referencia (canal Si puro) ──────────────────────────
    print("► [N=0] Canal Si uniforme (referencia)…")
    l0 = np.array([L_NM])              # único segmento Si de 15 nm
    s0 = build_structure(0, l0)
    T0 = _compute_T_batch(E_plot, s0)  # T(E) = 1 para toda E (sin interfaces)
    I0 = landauer_current(E_plot, T0)
    print(f"         I₀ = {I0:.8f}  [unidades: eV · (2e/h)⁻¹]")
    results.append((0, l0, I0))

    # ─── N=3: 4 Si + 3 SiGe ──────────────────────────────────────────────
    print(f"► [N=3] Optimizando con {N_STARTS} puntos iniciales…")
    t3 = time.time()
    l3, I3 = optimize_structure(3)
    red3   = (1.0 - I3 / I0) * 100.0
    print(f"         Longitudes Si   [nm]: {np.round(l3[0::2], 3)}")
    print(f"         Longitudes SiGe [nm]: {np.round(l3[1::2], 3)}")
    print(f"         Σ longitudes = {l3.sum():.6f} nm  (debe ser {L_NM} nm)")
    print(f"         I₃ = {I3:.8f}")
    print(f"         Reducción = {red3:.3f} %")
    print(f"         Tiempo: {time.time()-t3:.1f} s\n")
    results.append((3, l3, I3))

    # ─── N=5: 6 Si + 5 SiGe ──────────────────────────────────────────────
    print(f"► [N=5] Optimizando con {N_STARTS} puntos iniciales…")
    t5 = time.time()
    l5, I5 = optimize_structure(5)
    red5   = (1.0 - I5 / I0) * 100.0
    print(f"         Longitudes Si   [nm]: {np.round(l5[0::2], 3)}")
    print(f"         Longitudes SiGe [nm]: {np.round(l5[1::2], 3)}")
    print(f"         Σ longitudes = {l5.sum():.6f} nm  (debe ser {L_NM} nm)")
    print(f"         I₅ = {I5:.8f}")
    print(f"         Reducción = {red5:.3f} %")
    print(f"         Tiempo: {time.time()-t5:.1f} s\n")
    results.append((5, l5, I5))

    # ─── Resumen ──────────────────────────────────────────────────────────
    print("══════════════════════════════════════════════════════════════")
    print("  RESUMEN FINAL")
    print("══════════════════════════════════════════════════════════════")
    for N, lengths, I_N in results:
        red = (1.0 - I_N / I0) * 100.0
        print(f"  N={N}:  I = {I_N:.8f}   Reducción = {red:.3f} %")
    print(f"\n  Tiempo total: {time.time()-t_start:.1f} s")
    print("══════════════════════════════════════════════════════════════\n")

    # ─── Tres figuras independientes ─────────────────────────────────────
    print("► Generando figuras independientes…")

    # ── Figura 1: Tabla de resultados ─────────────────────────────────────
    fig_tbl, ax_tbl = plt.subplots(figsize=(14, 4.5))
    fig_tbl.patch.set_facecolor("white")
    ax_tbl.set_facecolor("white")
    build_table(ax_tbl, results)
    path_tbl = SAVE_PATH / "resultados_SiSiGe_tabla.png"
    fig_tbl.savefig(path_tbl, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_tbl)
    print(f"   ✓ Tabla guardada en:              '{path_tbl}'")

    # ── Figura 2: Perfil de potencial U(x) ───────────────────────────────
    fig_pot, ax_pot = plt.subplots(figsize=(9, 5.5))
    fig_pot.patch.set_facecolor("white")
    plot_potential(ax_pot, results)
    path_pot = SAVE_PATH / "resultados_SiSiGe_perfil_potencial.png"
    fig_pot.savefig(path_pot, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_pot)
    print(f"   ✓ Perfil U(x) guardado en:        '{path_pot}'")

    # ── Figura 3: Espectro de transmisión T(E) ────────────────────────────
    fig_tra, ax_tra = plt.subplots(figsize=(9, 5.5))
    fig_tra.patch.set_facecolor("white")
    plot_transmission(ax_tra, results)
    path_tra = SAVE_PATH / "resultados_SiSiGe_transmision.png"
    fig_tra.savefig(path_tra, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_tra)
    print(f"   ✓ Transmisión T(E) guardada en:   '{path_tra}'")


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
