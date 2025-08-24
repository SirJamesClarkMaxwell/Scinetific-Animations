from manimlib import *

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# --- Physical constants ---
q_e = 1.602176634e-19  # C
kB_eV = 8.617333262145e-5  # eV/K   (for plotting F(E) on an eV axis)
kB_J = 1.380649e-23  # J/K    (for the SI DOS prefactor)
hbar = 1.054571817e-34  # J*s
m0 = 9.1093837015e-31  # kg
CM3_PER_M3 = 1e-6  # m^-3 -> cm^-3

# Optional: keep your Nc if you want to compare to MB
Nc = 1e19  # cm^-3


def _integrand_eV(
    E_eV: float,
    Ec_eV: float,
    Ef_eV: float,
    T: float,
    m_eff_rel: float,
    valley_degeneracy: int,
) -> float:
    """Integrand expressed on an eV axis; returns SI density contribution per dE_eV.
    We convert to Joules internally and include a factor q_e so that
    ∫ integrand(E_eV) dE_eV -> m^-3 (SI).
    """
    EcJ = Ec_eV * q_e
    EJ = E_eV * q_e
    EfJ = Ef_eV * q_e
    kT = kB_J * T

    if EJ <= EcJ:
        return 0.0

    m_eff = m_eff_rel * m0
    pref = (valley_degeneracy / (2.0 * np.pi**2)) * ((2.0 * m_eff) ** 1.5) / (hbar**3)

    # Fermi-Dirac with simple guards for extreme tails
    x = (EJ - EfJ) / kT
    if x > 40.0:  # safely ~0
        f = 0.0
    elif x < -40.0:
        f = 1.0
    else:
        f = 1.0 / (1.0 + np.exp(x))

    # q_e converts dE[eV] -> dE[J], giving the correct units after integration
    return pref * np.sqrt(EJ - EcJ) * f * q_e  # -> m^-3 per eV


def calculate_concentration(
    Ec_eV: float,
    Ef_eV: float,
    T: float,
    m_eff_rel: float = 1.0,
    valley_degeneracy: int = 1,
    quad_limit: int = 200,
) -> float:
    """n = ∫_{Ec}^{∞} g_c(E) f_FD(E) dE  (returns cm^-3)."""
    val_m3, _err = quad(
        _integrand_eV,
        Ec_eV,
        np.inf,
        args=(Ec_eV, Ef_eV, T, m_eff_rel, valley_degeneracy),
        limit=quad_limit,
    )
    return val_m3 * CM3_PER_M3  # cm^-3



