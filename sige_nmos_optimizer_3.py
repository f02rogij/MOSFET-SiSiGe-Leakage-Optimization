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
║    - Masa efectiva distinta en Si (0.19 m₀) y SiGe (0.15 m₀)            ║
║    - TMM con condición de contorno de BenDaniel-Duke (m*(x) variable)    ║
║    - Caída de VDS CONTINUA: cada segmento de material se subdivide en    ║
║      N_VDS_SLICES sub-láminas finas; cada una lleva su propio valor de   ║
║      la rampa lineal U_OFF − (x/L)·VDS + offset_material evaluado en    ║
║      su centro exacto. Así N=0 también ve la rampa completa y la         ║
║      comparación entre casos es justa.                                   ║
║    - T(E): Método TMM vectorizado por lotes (batch)                      ║
║    - I_N : Fórmula de Landauer con distribuciones de Fermi-Dirac         ║
║    - Opt : scipy.optimize.minimize (SLSQP) con arranques aleatorios      ║
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
G0    = Q_E**2 / (np.pi * HBAR)   # cuanto de conductancia 2e²/h ≈ 7.748×10⁻⁵ S

# ══════════════════════════════════════════════════════════════════════════
# PARÁMETROS DEL DISPOSITIVO — Modificar aquí fácilmente
# ══════════════════════════════════════════════════════════════════════════

# ── Masas efectivas (dirección de transporte en el plano, canal [001]) ──
M_STAR_SI   = 0.190    # masa efectiva transversal Si, valle Δ₂ [adim.]
M_STAR_SIGE = 0.220    # masa efectiva de transporte SiGe x=0.3 [adim.]
M_EFF_SI    = M_STAR_SI   * M0   # [kg]
M_EFF_SIGE  = M_STAR_SIGE * M0   # [kg]

# ── Geometría ────────────────────────────────────────────────────────────
L_NM     = 15.0      # longitud total del canal [nm]

# ── Condiciones de contorno eléctrico ────────────────────────────────────
T_KELVIN = 358.0     # temperatura [K]  (85 °C, condición típica de prueba)
MU_S     =  0.0      # potencial químico de la fuente μS [eV]  (referencia)
MU_D     = -0.7      # potencial químico del drenador μD [eV]  (VDS = 0.7 V)
V_DS     = MU_S - MU_D  # caída de potencial total = 0.7 V (valor positivo)

# ── Perfil de bandas en estado OFF ───────────────────────────────────────
#   U(x) = U_OFF − (x_centro/L)·V_DS + offset_material
#
#   U_OFF       : altura de la barrera que fija la puerta en OFF [eV]
#   OFFSET_SI   : nivel de referencia de la banda de conducción del Si
#   OFFSET_SIGE : discontinuidad de banda ΔEc entre Si y SiGe
#   U_LEAD      : potencial del contacto (fuente, = 0 V de referencia)
U_OFF       =  0.30   # barrera de puerta en OFF [eV]
OFFSET_SI   =  0.00   # referencia Si [eV]
OFFSET_SIGE = +0.15   # barrera de potencial efectiva SiGe sobre Si [eV]
U_LEAD      =  0.00   # potencial del contacto (= lado fuente) [eV]

# ── Rango de integración en energía ─────────────────────────────────────
E_MIN    = 0.0        # [eV]
E_MAX    = 0.8        # [eV]

# ── Restricciones geométricas ────────────────────────────────────────────
L_SEG_MIN = 0.5       # longitud mínima por segmento [nm]
L_SEG_MAX = 15.0      # longitud máxima por segmento [nm]

# ── Resolución numérica ──────────────────────────────────────────────────
N_E_OPT  = 500        # puntos de E durante la optimización (velocidad)
N_E_PLOT = 2000       # puntos de E para las gráficas finales (resolución)

# N_VDS_SLICES: sub-láminas por segmento para la rampa continua de VDS.
# Cada segmento de material se subdivide en N_VDS_SLICES capas finas;
# cada una lleva el valor exacto de U_OFF - (x_centro/L)*V_DS + offset.
# Con 20 sub-láminas el error de discretización es < 2 meV.
N_VDS_SLICES = 20     # sub-láminas por segmento para la rampa VDS

# ── Optimización ─────────────────────────────────────────────────────────
N_STARTS = 40         # intentos con puntos iniciales aleatorios
RNG_SEED = 42         # semilla para reproducibilidad

# ── Directorio de salida de imágenes ─────────────────────────────────────
SAVE_PATH = Path(__file__).resolve().parent / "fotos"
SAVE_PATH.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# 1. MÓDULO DE ESTRUCTURA
# ══════════════════════════════════════════════════════════════════════════

def build_structure(N, lengths, n_slices=None):
    """
    Construye la estructura del canal como lista de tuplas
        (U_eV [float], d_m [float], m_eff [float])
    con una entrada por sub-lámina.

    Alternancia de materiales: Si – SiGe – Si – … – Si.

    Rampa VDS continua
    ------------------
    La caída de potencial debida a VDS es continua en el canal:
        U(x) = U_OFF - (x / L_total) * V_DS + offset_material(x)

    Para que el TMM la aproxime fielmente en TODOS los casos (incluyendo
    N=0), cada segmento de material se subdivide en n_slices sub-láminas
    uniformes de grosor d_nm/n_slices.  En el centro de cada sub-lámina
    se evalúa la rampa lineal exacta, de modo que la escalera resultante
    converge a la rampa continua cuando n_slices → ∞.

    Con n_slices = N_VDS_SLICES = 20 el error máximo de discretización
    de la rampa es ΔU < V_DS / (2 * n_slices) < 2 meV, despreciable.

    Esto garantiza que N=0 también muestre la rampa descendente completa
    (no un valor constante central), haciendo la comparación justa.

    Parámetros
    ----------
    N        : int            — número de segmentos SiGe (0, 3, 5, …)
    lengths  : array-like [nm] — longitudes de los 2N+1 segmentos de material
    n_slices : int             — sub-láminas por segmento (None → N_VDS_SLICES)

    Retorna
    -------
    structure : list de (U_eV, d_m, m_eff)
        Puede tener hasta (2N+1)*n_slices entradas.
    """
    if n_slices is None:
        n_slices = N_VDS_SLICES

    lengths = np.asarray(lengths, dtype=float)
    L_total = lengths.sum()          # debe ser ≈ L_NM [nm]
    n_seg   = 2 * N + 1

    if len(lengths) != n_seg:
        raise ValueError(
            f"build_structure: se esperan {n_seg} longitudes para N={N}, "
            f"pero se recibieron {len(lengths)}."
        )

    structure = []
    x_current = 0.0                  # posición acumulada [nm]

    for i, d_nm in enumerate(lengths):
        is_si  = (i % 2 == 0)
        offset = OFFSET_SI   if is_si else OFFSET_SIGE
        m_eff  = M_EFF_SI    if is_si else M_EFF_SIGE

        # Subdividir el segmento en n_slices capas finas iguales
        d_slice_nm = d_nm / n_slices
        d_slice_m  = d_slice_nm * 1e-9

        for s in range(n_slices):
            # Centro exacto de esta sub-lámina en el canal
            x_mid_nm = x_current + (s + 0.5) * d_slice_nm
            # Rampa lineal de VDS evaluada en el centro
            caida_V  = (x_mid_nm / L_total) * V_DS
            # Potencial total de la sub-lámina
            U_slice  = U_OFF - caida_V + offset
            structure.append((float(U_slice), float(d_slice_m), float(m_eff)))

        x_current += d_nm

    return structure


# ══════════════════════════════════════════════════════════════════════════
# 2. MATRICES DE TRANSFERENCIA (TMM con BenDaniel-Duke)
# ══════════════════════════════════════════════════════════════════════════
#
# Con masa efectiva m*(x) variable por región, la condición de contorno
# correcta en las interfaces es la de BenDaniel-Duke:
#
#   ψ continua  y   (1/m*) dψ/dx continua
#
# Esto modifica la matriz de propagación: en lugar de k se usa k̃ = k/m*
# en los elementos fuera de la diagonal.  La fórmula de transmisión resulta:
#
#   denom = i·k̃_L·n₁₁ + i·k̃_L·n₂₂ + k̃_L²·n₁₂ − n₂₁
#   T     = 4·k̃_L² / |denom|²
#
# donde k̃_L = k_L / m_lead  (contactos de Si → m_lead = M_EFF_SI).
#
# Verificación: para N=0 (canal Si uniforme, k_j = k_L):
#   N_tot = N_Si(L)  →  denom = 2i k̃_L e^{−ik_L L}  →  T = 1  ✓
# ──────────────────────────────────────────────────────────────────────────

def _prop_matrix_batch(k_arr, d_m, m_eff):
    """
    Matrices de propagación BenDaniel-Duke para un array de k.

    Con masa efectiva m_eff, la matriz de la capa j es:

        N_j = ⎡  cos(kd)       sin(kd) / k̃  ⎤
               ⎣ −k̃ sin(kd)   cos(kd)        ⎦

    donde k̃ = k / m_eff.

    Parámetros
    ----------
    k_arr : ndarray complex (n_E,) — vectores de onda [m⁻¹]
    d_m   : float  — espesor [m]
    m_eff : float  — masa efectiva [kg]

    Retorna
    -------
    Nj : ndarray complex (n_E, 2, 2)
    """
    kd    = k_arr * d_m
    c     = np.cos(kd)
    s     = np.sin(kd)
    n_E   = len(k_arr)
    Nj    = np.zeros((n_E, 2, 2), dtype=complex)

    # Umbral bajo el cual tratamos k ≈ 0 (modo evanescente muy suave o k nulo)
    KTINY = 1e3   # [m⁻¹]

    # k̃ = k / m_eff  (con k protegido contra 0)
    k_safe  = np.where(np.abs(k_arr) < KTINY, 1.0 + 0j, k_arr)
    k_tilde = k_safe / m_eff

    Nj[:, 0, 0] =  c
    Nj[:, 0, 1] =  s / k_tilde
    Nj[:, 1, 0] = -k_tilde * s
    Nj[:, 1, 1] =  c

    # Límite k→0: N_j → [[1, d·m_eff], [0, 1]]
    tiny = np.abs(k_arr) < KTINY
    if np.any(tiny):
        Nj[tiny, 0, 0] = 1.0
        Nj[tiny, 0, 1] = d_m * m_eff
        Nj[tiny, 1, 0] = 0.0
        Nj[tiny, 1, 1] = 1.0

    return Nj


def _compute_T_batch(E_grid_eV, structure,
                     m_lead=None, U_lead=None):
    """
    Calcula T(E) de forma vectorizada para todo el grid de energías.

    Parámetros
    ----------
    E_grid_eV : ndarray (n_E,) [eV]
    structure  : list de (U_eV, d_m, m_eff)
    m_lead     : float [kg]  — masa efectiva del contacto (None → M_EFF_SI)
    U_lead     : float [eV]  — potencial del contacto    (None → U_LEAD)

    Retorna
    -------
    T_arr : ndarray float (n_E,), valores en [0, 1]
    """
    if m_lead is None: m_lead = M_EFF_SI
    if U_lead is None: U_lead = U_LEAD

    n_E  = len(E_grid_eV)
    E_J  = E_grid_eV * Q_E
    UL_J = U_lead * Q_E

    # Número de onda del contacto (Si)
    kL      = np.sqrt((2.0 * m_lead * (E_J - UL_J) / HBAR**2).astype(complex))
    # Solo son físicamente relevantes las energías para las que el contacto
    # es propagante (parte real de kL > umbral)
    valid   = kL.real > 1e4
    k_tilde_L = kL / m_lead   # BenDaniel-Duke: k̃_L = kL / m*_contacto

    # Inicializar la matriz de transferencia total como identidad
    N_tot = np.broadcast_to(np.eye(2, dtype=complex), (n_E, 2, 2)).copy()

    for U_eV, d_m, m_eff in structure:
        U_J = U_eV * Q_E
        k_j = np.sqrt((2.0 * m_eff * (E_J - U_J) / HBAR**2).astype(complex))
        Nj  = _prop_matrix_batch(k_j, d_m, m_eff)
        # Acumulación de izquierda a derecha: N_tot = N_j @ N_tot
        N_tot = Nj @ N_tot

    n11 = N_tot[:, 0, 0]
    n12 = N_tot[:, 0, 1]
    n21 = N_tot[:, 1, 0]
    n22 = N_tot[:, 1, 1]

    # Fórmula de Landauer-Büttiker con BenDaniel-Duke
    # (contactos Si simétricos → k̃_L = k̃_R)
    denom = (1j * k_tilde_L * n11
           + 1j * k_tilde_L * n22
           + k_tilde_L**2 * n12
           - n21)

    T_raw = np.where(
        valid,
        np.real(4.0 * np.abs(k_tilde_L)**2 / (np.abs(denom)**2 + 1e-300)),
        0.0
    )
    return np.clip(T_raw, 0.0, 1.0).astype(float)


# ══════════════════════════════════════════════════════════════════════════
# 3. DISTRIBUCIONES DE FERMI-DIRAC Y CORRIENTE DE LANDAUER
# ══════════════════════════════════════════════════════════════════════════

def fermi(E_eV, mu_eV, T_K=None):
    """
    Distribución de Fermi-Dirac.

      f(E, μ) = 1 / (1 + exp((E − μ) / (k_B T)))

    Parámetros
    ----------
    E_eV  : float o ndarray [eV]
    mu_eV : float [eV]
    T_K   : float [K]  (None → T_KELVIN global)

    Retorna
    -------
    f : float o ndarray ∈ (0, 1)
    """
    if T_K is None: T_K = T_KELVIN
    x = np.clip((E_eV - mu_eV) / (KB_EV * T_K), -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(x))


def landauer_current(E_grid, T_E, muS=None, muD=None, T_K=None):
    """
    Corriente de Landauer en Amperes:

      I_N = (2e²/h) · ∫ [f(E,μS) − f(E,μD)] · T_N(E) dE   [A]

    El prefactor G₀ = 2e²/h ≈ 7.748×10⁻⁵ S convierte eV → A.

    Parámetros
    ----------
    E_grid : ndarray [eV]
    T_E    : ndarray — coeficiente de transmisión ∈ [0, 1]
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
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return float(G0 * _trapz((fS - fD) * T_E, E_grid))


# ══════════════════════════════════════════════════════════════════════════
# 4. FUNCIÓN OBJETIVO Y OPTIMIZACIÓN
# ══════════════════════════════════════════════════════════════════════════

def objective(lengths, N):
    """
    Función objetivo para scipy.optimize: retorna I_N (a minimizar).

    Parámetros
    ----------
    lengths : ndarray [nm] — longitudes de los 2N+1 segmentos
    N       : int — número de segmentos SiGe

    Retorna
    -------
    I : float — corriente OFF proporcional (a minimizar)
    """
    structure = build_structure(N, lengths)
    E_grid    = np.linspace(E_MIN, E_MAX, N_E_OPT)
    T_E       = _compute_T_batch(E_grid, structure)
    return landauer_current(E_grid, T_E)


def optimize_structure(N, seed=RNG_SEED):
    """
    Optimiza las longitudes de segmentos para minimizar la corriente OFF.

    Algoritmo: SLSQP con N_STARTS puntos iniciales aleatorios.
    Restricción de igualdad: Σ lengths = L_NM.
    Cotas:                   L_SEG_MIN ≤ l_i ≤ L_SEG_MAX.

    Parámetros
    ----------
    N    : int — número de segmentos SiGe
    seed : int — semilla aleatoria

    Retorna
    -------
    best_lengths : ndarray [nm]
    best_I       : float [A]
    """
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
                objective,
                x0,
                args=(N,),
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

    if best_lengths is None:
        best_lengths = np.full(n_seg, L_NM / n_seg)
        best_I       = float(objective(best_lengths, N))
        print(f"      ⚠ Fallback a distribución uniforme (0/{N_STARTS} convergen)")
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

    Columnas: N_SiGe | Longitudes Si | Longitudes SiGe | I_N [nA] | Reducción%

    Parámetros
    ----------
    ax      : matplotlib Axes
    results : list de (N, lengths_nm, I_N)
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
        si_l = lengths[0::2]
        sg_l = lengths[1::2]
        red  = (1.0 - I_N / I_ref) * 100.0 if I_ref > 1e-30 else 0.0

        rows.append([
            str(N),
            _compact_lengths(si_l, per_line=3),
            _compact_lengths(sg_l, per_line=3) if len(sg_l) > 0 else "—",
            f"{I_N * 1e9:.4f}",
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

    HEADER_COLOR = "#1a3464"
    for j in range(len(col_labels)):
        cell = tbl[(0, j)]
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)

    row_bgs = ["#d9e6f5", "#f4f8ff", "#d9e6f5"]
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            tbl[(i, j)].set_facecolor(row_bgs[(i - 1) % len(row_bgs)])


def plot_potential(ax, results):
    """
    Dibuja el perfil de potencial U(x) para los tres casos optimizados.

    Para N=0 se muestra la rampa lineal continua (un solo segmento Si
    con la caída de VDS).  Para N>0 cada segmento tiene su valor central
    de potencial, produciendo la escalera escalonada.

    Se añade la rampa lineal de referencia U_OFF − (x/L)·VDS + OFFSET_SI
    como línea de puntos para facilitar la comparación visual.

    Parámetros
    ----------
    ax      : matplotlib Axes
    results : list de (N, lengths_nm, I_N)
    """
    palette = ["#1a6faf", "#cc4125", "#2e7d32"]
    labels  = ["N=0  (1 Si)", "N=3  (4 Si + 3 SiGe)", "N=5  (6 Si + 5 SiGe)"]
    lws     = [2.8, 2.2, 1.8]

    for idx, (N, lengths, _) in enumerate(results):
        # Reconstruir la escalera fina (n_slices sub-láminas) para visualizar
        # la rampa continua tal como la ve el TMM.  Se usa n_slices=N_VDS_SLICES
        # para que la gráfica sea coherente con el cálculo.
        structure = build_structure(N, lengths)

        # Calcular bordes de cada sub-lámina a partir de las longitudes originales
        lengths_arr = np.asarray(lengths, dtype=float)
        n_seg_mat   = len(lengths_arr)
        edges_sub   = []        # bordes izquierdo de cada sub-lámina [nm]
        x_cur       = 0.0
        for d_nm in lengths_arr:
            d_sl = d_nm / N_VDS_SLICES
            for s in range(N_VDS_SLICES):
                edges_sub.append(x_cur + s * d_sl)
            x_cur += d_nm
        edges_sub.append(L_NM)  # borde derecho del último sub-segmento

        x_plot, U_plot = [], []
        for k, (U, d_m, _) in enumerate(structure):
            xl = edges_sub[k]
            xr = edges_sub[k + 1]
            x_plot += [xl, xr]
            U_plot += [U, U]

        # Extender a los contactos
        x_full = [-1.5, 0.0]      + x_plot + [L_NM, L_NM + 1.5]
        U_full = [U_LEAD, U_LEAD] + U_plot + [MU_D,  MU_D]

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
    U_all = [U_OFF + OFFSET_SI, U_OFF + OFFSET_SIGE - V_DS, MU_D, MU_S, U_LEAD]
    U_margin = 0.08
    ax.set_ylim(min(U_all) - U_margin, max(U_all) + U_margin + 0.05)
    ax.legend(fontsize=8.5, loc="lower right", ncol=1, framealpha=0.85)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_facecolor("#f7f9fc")


def plot_transmission(ax, results):
    """
    Dibuja T(E) para los tres casos y la ventana de Fermi [f(μS) − f(μD)].

    Parámetros
    ----------
    ax      : matplotlib Axes
    results : list de (N, lengths_nm, I_N)
    """
    palette = ["#1a6faf", "#cc4125", "#2e7d32"]
    labels  = ["N=0  (1 Si)", "N=3  (4 Si + 3 SiGe)", "N=5  (6 Si + 5 SiGe)"]
    lws     = [2.8, 2.2, 1.8]

    E_grid  = np.linspace(E_MIN, E_MAX, N_E_PLOT)
    fS      = fermi(E_grid, MU_S)
    fD      = fermi(E_grid, MU_D)
    fermi_w = fS - fD

    ax.fill_between(E_grid, fermi_w, alpha=0.22, color="#e57373",
                    label=r"Ventana Fermi  $f(\mu_S) - f(\mu_D)$", zorder=1)

    for idx, (N, lengths, _) in enumerate(results):
        structure = build_structure(N, lengths)
        T_E       = _compute_T_batch(E_grid, structure)
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
      2. Optimiza N=3: longitudes que minimizan I₃.
      3. Optimiza N=5: ídem para I₅.
      4. Genera tres figuras PNG independientes (tabla, U(x), T(E)).
    """
    t_start = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Optimizador Si/SiGe — Corriente OFF en Canal NMOS          ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  m*_Si = {M_STAR_SI} m₀  ·  m*_SiGe = {M_STAR_SIGE} m₀                      ║")
    print(f"║  L = {L_NM} nm  ·  T = {T_KELVIN:.0f} K                                  ║")
    print(f"║  μS = {MU_S} eV  ·  μD = {MU_D} eV  ·  VDS = {V_DS:.2f} V              ║")
    print(f"║  U_OFF = {U_OFF} eV  ·  ΔEc(SiGe) = {OFFSET_SIGE} eV                    ║")
    print(f"║  E ∈ [{E_MIN}, {E_MAX}] eV  ·  N_starts = {N_STARTS}                         ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    results = []
    E_plot  = np.linspace(E_MIN, E_MAX, N_E_PLOT)

    # ─── N=0: caso de referencia ──────────────────────────────────────────
    print("► [N=0] Canal Si uniforme (referencia)…")
    l0 = np.array([L_NM])
    s0 = build_structure(0, l0)
    T0 = _compute_T_batch(E_plot, s0)
    I0 = landauer_current(E_plot, T0)
    print(f"         I₀ = {I0:.6e} A  ({I0*1e9:.4f} nA)")
    results.append((0, l0, I0))

    # ─── N=3: 4 Si + 3 SiGe ──────────────────────────────────────────────
    print(f"\n► [N=3] Optimizando con {N_STARTS} puntos iniciales…")
    t3 = time.time()
    l3, I3 = optimize_structure(3)
    red3   = (1.0 - I3 / I0) * 100.0
    print(f"         Longitudes Si   [nm]: {np.round(l3[0::2], 3)}")
    print(f"         Longitudes SiGe [nm]: {np.round(l3[1::2], 3)}")
    print(f"         Σ = {l3.sum():.4f} nm  (objetivo: {L_NM} nm)")
    print(f"         I₃ = {I3:.6e} A  ({I3*1e9:.4f} nA)")
    print(f"         Reducción = {red3:.3f} %")
    print(f"         Tiempo: {time.time()-t3:.1f} s")
    results.append((3, l3, I3))

    # ─── N=5: 6 Si + 5 SiGe ──────────────────────────────────────────────
    print(f"\n► [N=5] Optimizando con {N_STARTS} puntos iniciales…")
    t5 = time.time()
    l5, I5 = optimize_structure(5)
    red5   = (1.0 - I5 / I0) * 100.0
    print(f"         Longitudes Si   [nm]: {np.round(l5[0::2], 3)}")
    print(f"         Longitudes SiGe [nm]: {np.round(l5[1::2], 3)}")
    print(f"         Σ = {l5.sum():.4f} nm  (objetivo: {L_NM} nm)")
    print(f"         I₅ = {I5:.6e} A  ({I5*1e9:.4f} nA)")
    print(f"         Reducción = {red5:.3f} %")
    print(f"         Tiempo: {time.time()-t5:.1f} s")
    results.append((5, l5, I5))

    # ─── Resumen ──────────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════════════════════════")
    print("  RESUMEN FINAL")
    print("══════════════════════════════════════════════════════════════")
    for N, lengths, I_N in results:
        red = (1.0 - I_N / I0) * 100.0
        print(f"  N={N}:  I = {I_N*1e9:.4f} nA   Reducción = {red:.3f} %")
    print(f"\n  Tiempo total: {time.time()-t_start:.1f} s")
    print("══════════════════════════════════════════════════════════════\n")

    # ─── Figuras ──────────────────────────────────────────────────────────
    print("► Generando figuras…")

    # Figura 1: Tabla de resultados
    fig_tbl, ax_tbl = plt.subplots(figsize=(14, 4.5))
    fig_tbl.patch.set_facecolor("white")
    ax_tbl.set_facecolor("white")
    build_table(ax_tbl, results)
    path_tbl = SAVE_PATH / "resultados_SiSiGe_tabla.png"
    fig_tbl.savefig(path_tbl, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_tbl)
    print(f"   ✓ Tabla guardada:          '{path_tbl}'")

    # Figura 2: Perfil de potencial U(x)
    fig_pot, ax_pot = plt.subplots(figsize=(9, 5.5))
    fig_pot.patch.set_facecolor("white")
    plot_potential(ax_pot, results)
    path_pot = SAVE_PATH / "resultados_SiSiGe_perfil_potencial.png"
    fig_pot.savefig(path_pot, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_pot)
    print(f"   ✓ Perfil U(x) guardado:    '{path_pot}'")

    # Figura 3: Espectro de transmisión T(E)
    fig_tra, ax_tra = plt.subplots(figsize=(9, 5.5))
    fig_tra.patch.set_facecolor("white")
    plot_transmission(ax_tra, results)
    path_tra = SAVE_PATH / "resultados_SiSiGe_transmision.png"
    fig_tra.savefig(path_tra, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_tra)
    print(f"   ✓ Transmisión T(E) guardada: '{path_tra}'")

    print("\n  (Mostrando figuras en pantalla…)")
    # Re-crear figuras para plt.show()
    fig_tbl2, ax_tbl2 = plt.subplots(figsize=(14, 4.5))
    build_table(ax_tbl2, results)
    fig_pot2, ax_pot2 = plt.subplots(figsize=(9, 5.5))
    plot_potential(ax_pot2, results)
    fig_tra2, ax_tra2 = plt.subplots(figsize=(9, 5.5))
    plot_transmission(ax_tra2, results)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
