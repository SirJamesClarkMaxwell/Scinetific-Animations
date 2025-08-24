from manimlib import *
from scipy.integrate import quad
from typing import Callable

# sys.path.append(os.path.dirname(__file__))
# from utils import TranslucentBand, Slider  # ignore

# --- Physical constants ---
q_e = 1.602176634e-19  # C
kB_eV = 8.617333262145e-5  # eV/K   (for plotting F(E) on an eV axis)
kB_J = 1.380649e-23  # J/K    (for the SI DOS prefactor)
hbar = 1.054571817e-34  # J*s
m0 = 9.1093837015e-31  # kg
CM3_PER_M3 = 1e-6  # m^-3 -> cm^-3

Nc = 1e19  # cm^-3


class ConcentrationDependence(Scene):
    def construct(self):
        main_axes = Axes(
            x_range=[0, 10, 2], y_range=[0, 1.2, 2], height=3, width=6
        ).to_corner(UL)
        T = ValueTracker(3500)
        numbs = main_axes.add_coordinate_labels()
        fermi_plot =  main_axes.get_graph(
            lambda E: self.fermi_function(E, 5, T.get_value()),
            x_range=[0, 10, 0.001],
            color=RED,
            discontinuities=[5],
            use_smoothing=False,
        )
        # fermi_plot.set_color(RED)
        self.add(main_axes.x_axis, fermi_plot, numbs)
        # self.play(T.animate.set_value(3500))
        self.wait()

    def calculate_fermi_level(self, Nd: float, Ed: float, Ec: float, T: float) -> float:

        denominator = Nc * (
            1 + np.sqrt(1 + 4 * (Nd / Nc) * np.exp((Ec - Ed) / (kB_J * T)))
        )
        return Ec + np.log(2 * (Nd / denominator))

    def calculate_concentration(
        self,
        Ec_eV: float,
        Ef_eV: float,
        T: float,
        m_eff_rel: float = 1.0,
        valley_degeneracy: int = 1,
        quad_limit: int = 200,
    ) -> float:
        """n = ∫_{Ec}^{∞} g_c(E) f_FD(E) dE  (returns cm^-3)."""
        val_m3, _err = quad(
            self.__integrand_eV,
            Ec_eV,
            np.inf,
            args=(Ec_eV, Ef_eV, T, m_eff_rel, valley_degeneracy),
            limit=quad_limit,
        )
        return val_m3 * CM3_PER_M3  # cm^-3

    def fermi_function(self, E, mu, T) -> float:
        x = (E - mu) / kB_J * T
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(x))

    def __integrand_eV(
        self,
        E_eV: float,
        Ec_eV: float,
        Ef_eV: float,
        T: float,
        m_eff_rel: float,
        valley_degeneracy: int,
    ) -> float:
        """
        Integrand expressed on an eV axis; returns SI density contribution per dE_eV.
        We convert to Joules internally and include a factor q_e so that
        ∫ integrand(E_eV) dE_eV -> m^-3 (SI).
        """
        EcJ = Ec_eV * q_e
        EJ = E_eV * q_e
        EfJ = Ef_eV * q_e

        if EJ <= EcJ:
            return 0.0

        m_eff = m_eff_rel * m0
        pref = (
            (valley_degeneracy / (2.0 * np.pi**2)) * ((2.0 * m_eff) ** 1.5) / (hbar**3)
        )

        f = self.fermi_function(EJ, EfJ, T)
        return pref * np.sqrt(EJ - EcJ) * f * q_e  # -> m^-3 per eV
