from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Tuple, Optional, Dict

from matplotlib import axes
from scipy.integrate import quad

from manim import *
from manim.utils.color import interpolate_color
import os, sys


sys.path.append(os.path.dirname(__file__))
from utils import Slider, TranslucentBand
from scipy.optimize import fsolve

# Boltzmann constant in eV/K
k_B = 8.617333262145e-5

## Fermi-Dirac Distribution
T_MIN, T_MAX, T_STEP = 0.0, 5500.0, 500.0
GRADIENT_STOPS: List[Tuple[float, ManimColor]] = [
    (0.0, BLUE),
    (0.5, YELLOW),
    (1.0, RED),
]


def multi_stop_color(
    value: float, vmin: float, vmax: float, stops: List[Tuple[float, ManimColor]]
) -> ManimColor:
    """Return a color for value in [vmin, vmax] using multi-stop linear interpolation."""
    if vmax <= vmin:
        return stops[-1][1]
    a = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
    # find segment [p_i, p_{i+1}] that contains a
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        if a <= p1 or i == len(stops) - 2:
            # local interpolation factor
            t = 0.0 if p1 <= p0 else (a - p0) / (p1 - p0)
            t = float(np.clip(t, 0.0, 1.0))
            return interpolate_color(c0, c1, t)
    return stops[-1][1]


def temp_to_color(T: float) -> ManimColor:
    return multi_stop_color(T, T_MIN, T_MAX, GRADIENT_STOPS)


class FermiDiracDistribution(Scene):
    def fermi_dirac(self, E: float, mu: float, T: float) -> float:
        # Stable evaluation (avoid overflow at very low T)
        x = (E - mu) / (k_B * T)
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(x))

    def construct(self):
        x_bounds = [0, 10]
        axes = (
            Axes(
                x_range=[*x_bounds, 2],
                x_length=8,
                y_range=[0, 1.2, 0.2],
                y_length=5,
                x_axis_config={"tip_shape": StealthTip, "tip_height": 0.2},
                y_axis_config={"tip_shape": StealthTip, "tip_height": 0.2},
            )
            .add_coordinates()
            .to_edge(LEFT)
        )

        labels = axes.get_axis_labels("E", MathTex(r"\mathbb{P}"))

        # Parameters
        mu = 5.0  # eV
        T_min, T_max, T_step = 1e-8, 3500, 500
        temps = np.arange(T_min, T_max, T_step)
        temperature = ValueTracker(T_min)  # start at 0 K

        def generate_fermi_dirac_plot(T: ValueTracker, scene: Scene) -> VMobject:
            plot = axes.plot(
                lambda E: self.fermi_dirac(E, mu, T.get_value()),
                x_range=[*x_bounds, 0.001],
                color=temp_to_color(T.get_value()),
                stroke_width=4,
                use_smoothing=False,
                discontinuities=[mu],
                dt=0.0001,
            )
            if T.get_value() % 500 == 0:
                scene.add(plot)
            return plot

        # Live curve (color changes with T)
        live_curve = always_redraw(lambda: generate_fermi_dirac_plot(temperature, self))

        # --- Temperature slider UI ---
        slider_width, slider_height = 0.3, 5.0
        slider_rect = Rectangle(
            width=slider_width, height=slider_height, stroke_color=WHITE, stroke_width=1
        ).next_to(axes, RIGHT, LARGE_BUFF, aligned_edge=DOWN)

        # Fill with a BLUE→RED gradient
        slider_rect.set_fill([BLUE, YELLOW, RED], opacity=1.0)
        # slider_rect.next_to(axes, DOWN, buff=0.6)

        # Triangle marker that follows temperature and matches its color
        def make_marker():
            T = temperature.get_value()
            a = np.clip(T / (T_max - T_step), 0.0, 1.0)
            x = slider_rect.get_right()[0] + slider_width * 0.6
            y = slider_rect.get_bottom()[1] + a * slider_height
            tri = Triangle().rotate(PI / 2).scale(0.2)
            tri.set_fill(temp_to_color(T), opacity=1.0).set_stroke(WHITE, 1)
            tri.move_to(np.array([x, y, 0.0]))
            return tri

        marker = always_redraw(make_marker)

        temperature_text = Variable(
            temperature.get_value(),
            label=MathTex("T [K]", font_size=30),
            var_type=Integer,
        )

        temperature_text.tracker = temperature

        def temperature_text_updator() -> Callable:
            def updator(mob, dt) -> None:
                mob.next_to(marker, RIGHT, buff=SMALL_BUFF)
                mob.value.set_color(temp_to_color(temperature.get_value()))

            return updator

        temperature_text.add_updater(temperature_text_updator())
        self.add(axes, labels, slider_rect, marker, temperature_text, live_curve)

        for T in temps:
            self.play(
                temperature.animate.set_value(int(T)), run_time=1.5, rate_func=linear
            )
            self.wait()
        self.wait(0.5)


## ConcentrationDependency
q_e = 1.602176634e-19  # C
kB_eV = 8.617333262145e-5  # eV/K   (for plotting F(E) on an eV axis)
kB_J = 1.380649e-23  # J/K    (for the SI DOS prefactor)
hbar = 1.054571817e-34  # J*s
m0 = 9.1093837015e-31  # kg
CM3_PER_M3 = 1e-6  # m^-3 -> cm^-3
Nc = 1e19  # cm^-3
Nv = 1e19  # cm^-3


class ConcentrationDependency(ZoomedScene):
    def __init__(self, **kwargs):
        self.zoom_tracker = ValueTracker(1)
        self.zoom_display_width = ValueTracker(3)

        self.zoom_display_height = ValueTracker(6)
        ZoomedScene.__init__(
            self,
            zoom_factor=self.zoom_tracker.get_value(),
            zoomed_display_height=3,
            zoomed_display_width=7,
            image_frame_stroke_width=20,
            zoomed_camera_config={
                "default_frame_stroke_width": 2,
            },
            **kwargs,
        )

        self.T = ValueTracker(300)
        self.Eg = ValueTracker(1.2)
        self.Nd = ValueTracker(1e16)
        self.Ed = ValueTracker(1.0)

    def setup(self):
        super().setup()
        zoomed_display = self.zoomed_display
        zoomed_display_border = zoomed_display.display_frame
        frame = self.zoomed_camera.frame
        self.zoomed_objs = [zoomed_display, zoomed_display_border, frame]

    def change_stroke_width_zoomed_display(self):
        mob = Mobject()
        init_stroke = self.zoomed_camera.cairo_line_width_multiple

        def updater(_):
            scale = self.zoom_tracker.get_value()
            self.zoomed_camera.cairo_line_width_multiple = scale * init_stroke

        mob.add_updater(updater)
        return mob

    def zoom(self, zoom_level: float, animate: bool = True):
        if animate:
            return self.zoomed_objs[-1].animate.scale(
                1 / zoom_level
            ), self.zoom_tracker.animate.set_value(
                self.zoom_tracker.get_value() * 1 / zoom_level
            )

    def construct(self):
        mob = self.change_stroke_width_zoomed_display()
        self.add(mob)
        x_bounds = [-1, 3.5]
        axes = (
            Axes(
                x_range=[*x_bounds, 0.5],
                x_length=5,
                y_range=[0, 1.2, 0.2],
                y_length=3,
                x_axis_config={
                    "tip_shape": StealthTip,
                    "tip_height": 0.2,
                    "numbers_to_include": np.arange(*x_bounds, 0.5),
                    "font_size": 30,
                },
                y_axis_config={"tip_shape": StealthTip, "tip_height": 0.2},
            )
            .to_corner(UL, buff=SMALL_BUFF * 5)
            .add_coordinates(np.arange(*x_bounds, 0.5))
            .to_edge(UP, SMALL_BUFF)
        )

        x_label = axes.get_x_axis_label("E")

        # Live curve (color changes with T)
        fermi_dirac_function = always_redraw(
            lambda: axes.plot(
                lambda E: self.fermi_dirac(E, self.Ef),
                x_range=[*x_bounds, 0.001],
                color=temp_to_color(self.T.get_value()),
                # stroke_width=4,
                use_smoothing=False,
                discontinuities=[self.Ef],
                dt=0.0001,
                stroke_width=2,
            )
        )
        fermi_level = always_redraw(
            lambda: DashedVMobject(
                vmobject=VMobject()
                .set_points(
                    [
                        *axes.plot_parametric_curve(
                            function=lambda t: [self.Ef, t],
                            t_range=[0, 1.1],
                            stroke_width=3,
                            color=YELLOW,
                        ).get_all_points()
                    ]
                )
                .set_color(YELLOW),
                color=YELLOW,
            )
        )
        conduction_band_dos = always_redraw(
            lambda: axes.plot(
                lambda E: self.DOS(self.Eg.get_value(), E),
                x_range=[self.Eg.get_value(), self.Eg.get_value() + 1, 0.001],
                color=BLUE,
                stroke_width=2,
            )
        )
        valenced_band_dos = always_redraw(
            lambda: axes.plot(
                lambda E: self.DOS(0, -E),
                x_range=[-1, 0, 0.001],
                color=RED,
                stroke_width=2,
            )
        )
        concetration_function = always_redraw(
            lambda: axes.plot(
                lambda E: self.DOS(self.Eg.get_value(), E)
                * self.fermi_dirac(E, self.Ef),
                x_range=[self.Eg.get_value(), x_bounds[1], 0.001],
                color=PURPLE,
            )
        )
        concentration_area = always_redraw(
            lambda: axes.get_area(
                concetration_function,
                x_range=[self.Eg.get_value(), x_bounds[1]],
                color=PURPLE,
                opacity=0.5,
            )
        )
        donor_level = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda t: [self.Ed.get_value(), t], t_range=[0, 0.5], color=PINK
            )
        )

        # Sliders
        sliders_length = 2
        numberline_kwargs = {
            "rotation": PI / 2,
            "label_direction": LEFT,
            "include_numbers": True,
            "font_size": 30,
        }
        Eg_slider = Slider(
            x_range=[0, 2, 0.2],
            length=sliders_length,
            tracker=self.Eg,
            color=WHITE,
            label=f"E_g = ",
            post_label=" [eV]",
            numberline_kwargs=numberline_kwargs,
        )
        Ed_slider = Slider(
            x_range=[0, 2, 0.2],
            length=sliders_length,
            tracker=self.Ed,
            color=PURPLE,
            label=f"E_d = ",
            post_label="[eV]",
            numberline_kwargs=numberline_kwargs,
        )
        Nd_slider = Slider(
            x_range=[12, 19, 1],
            length=sliders_length,
            color=ORANGE,
            tracker=self.Nd,
            label=r"N_d = ",
            post_label=r" cm^{-3}",
            numberline_kwargs={
                "scaling": LogBase(),
                **numberline_kwargs,
            },
        )
        Nd_slider.label[1] = always_redraw(
            lambda: MathTex(rf"{Nd_slider.tracker.get_value():.1e}", font_size=20)
            .next_to(Nd_slider.numberline, DOWN, SMALL_BUFF)
            .set_color(ORANGE)
        )
        Nd_slider.label.add_updater(
            lambda mob: mob.arrange(RIGHT, SMALL_BUFF)
            .next_to(Nd_slider.numberline, DOWN, SMALL_BUFF * 1.3, aligned_edge=DOWN)
            .set_color(ORANGE)
        )

        T_Slider = Slider(
            x_range=[0, 1600, 300],
            length=sliders_length,
            tracker=self.T,
            color=RED,
            label=r"T = ",
            post_label=r" [K]",
            numberline_kwargs=numberline_kwargs,
        )

        sliders = Group(Eg_slider, Ed_slider, Nd_slider, T_Slider).add_updater(
            lambda mob: mob.arrange(RIGHT).to_corner(DR)
        )

        second_axes = (
            Axes(
                x_range=[0, 3.5, 1],
                x_length=5,
                y_range=[-1, 2.5, 0.5],
                y_length=3,
                y_axis_config={
                    "numbers_to_include": np.arange(0, 2.6, 0.5),
                    "font_size": 30,
                },
                x_axis_config={"include_ticks": False},
                tips=False,
            ).to_corner(DL)
            # .add_coordinates(np.arange(*x_bounds, 0.5))
        )
        second_axes_labels = second_axes.get_axis_labels(
            Text("x", font_size=30), MathTex(r"E [eV]", font_size=30)
        )

        valenced_band = TranslucentBand(
            second_axes,
            x0=0,
            x1=3,
            y0=-1,
            y1=0,
            color_bottom=RED,
            color_top=RED,
            opacity_bottom=0.3,
            opacity_top=0.9,
        )
        conduction_band = always_redraw(
            lambda: TranslucentBand(
                second_axes,
                x0=0,
                x1=3,
                y0=self.Eg.get_value(),
                y1=self.Eg.get_value() + 1,
                color_bottom=BLUE,
                color_top=BLUE,
                opacity_bottom=0.9,
                opacity_top=0.3,
            )
        )
        second_fermi_level = always_redraw(
            lambda: DashedVMobject(
                VMobject(color=YELLOW).set_points(
                    [
                        *second_axes.plot_parametric_curve(
                            lambda t: [t, self.Ef], t_range=[0, 1.5]
                        ).get_all_points()
                    ]
                ),
                color=YELLOW,
            )
        )
        second_donor_level = always_redraw(
            lambda: second_axes.plot_parametric_curve(
                lambda t: [t, self.Ed.get_value()], t_range=[1.5, 3], color=PURPLE
            )
        )

        self.add(
            axes.get_x_axis(),
            fermi_dirac_function,
            x_label,
            conduction_band_dos,
            valenced_band_dos,
            concetration_function,
            concentration_area,
            fermi_level,
            donor_level,
            sliders,
            second_axes,
            second_axes_labels,
            conduction_band,
            valenced_band,
            second_fermi_level,
            second_donor_level,
        )
        zoomed_display: ImageMobjectFromCamera = self.zoomed_objs[0]
        zoomed_display.next_to(
            axes, RIGHT, LARGE_BUFF, aligned_edge=DOWN
        )  # ,submobject_to_align=axes.coordinate_labels[-1])
        zoomed_display_border: SurroundingRectangle = self.zoomed_objs[1]
        frame: ScreenRectangle = self.zoomed_objs[2]
        frame.add_updater(
            lambda mob: mob.move_to(axes.get_x_axis().n2p(self.Eg.get_value())).shift(
                0.6 * UP * frame.get_height() / 2 + 0.8 * RIGHT * frame.get_width() / 2
            )
        )

        self.play(self.zoom(70))
        self.activate_zooming()
        self.play(self.get_zoomed_display_pop_out_animation())
        self.play(self.T.animate.set_value(1600), run_time=2)
        self.wait()
        self.play(self.Nd.animate.set_value(1e18), run_time=2)
        self.play(self.Eg.animate.set_value(2), run_time=2)
        self.wait()
        self.play(self.Ed.animate.set_value(1.9), run_time=2)
        self.wait()
        self.play(self.T.animate.set_value(1500), self.zoom(0.1), run_time=2)
        self.wait()

    def fermi_dirac(self, E: float, mu: float) -> float:
        # Stable evaluation (avoid overflow at very low T)
        x = (E - mu) / (k_B * self.T.get_value())
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(x))

    def calculate_fermi_level(self, Nd: float, Ed: float, Ec: float, T: float) -> float:
        first = np.exp((Ec - Ed) / (k_B * T))
        second = np.sqrt(1 + 4 * (Nd / Nc) * first)
        denominator = Nc * (1 + second)
        final = Ec + k_B * T * np.log(2 * (Nd / denominator))
        return final

    @property
    def Ef(self) -> float:
        return self.calculate_fermi_level(
            self.Nd.get_value(),
            self.Ed.get_value(),
            self.Eg.get_value(),
            self.T.get_value(),
        )

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

    def DOS(self, EcJ, EJ) -> float:
        if EJ <= EcJ:
            return 0
        return np.sqrt(EJ - EcJ)

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

        f = self.fermi_dirac(EJ, EfJ)
        return pref * np.sqrt(EJ - EcJ) * f * q_e  # -> m^-3 per eV


# config.write_to_movie = False
## Fermi level dependence
class FermiLevelDependence(Scene):
    def setup(self):
        self.Nd = ValueTracker(13)
        self.Na = ValueTracker(0)
        self.Ed = ValueTracker(1.0)
        self.Eg = ValueTracker(1.2)
        return super().setup()

    def construct(self):

        # parameters
        T_MIN, T_MAX, T_STEP = 50, 1000, 100

        T_ARHHENIUS_STEP = T_STEP * 10
        T_range = [T_MIN, T_MAX, 1]
        main_axes, arrhenius_axes, semiconductor_axes = self.generate_axes(
            T_MIN, T_MAX, T_STEP
        )
        sliders: Dict[str, Slider] = self.generate_slider()
        tmp_sliders = (
            Group(*sliders.values())
            .arrange(RIGHT, SMALL_BUFF, aligned_edge=DOWN)
            .next_to(arrhenius_axes, RIGHT, SMALL_BUFF)
            .to_edge(RIGHT, SMALL_BUFF)
        )
        semiconductor_bands = self.generate_semiconductors_look(
            main_axes, semiconductor_axes, T_MIN, T_MAX
        )

        def generate_Ef_T_plot(Nd: ValueTracker, scene: Scene):
            plot = main_axes.plot(
                function=lambda T: scene.Ef(T),
                x_range=T_range,
                color=YELLOW,
                stroke_width=3
            )
            # if Nd.get_value() % 10 in range(1, 9):
            #     scene.add(plot)
            return plot

        def generate_arrhenius_plot(Nd: ValueTracker, scene: Scene):
            plot = arrhenius_axes.plot(
                # x is 1000/T ⇒ T = 1000/x
                function=lambda invT1000: scene.calculate_concentration(
                    1000.0 / invT1000
                ),
                x_range=[invT_min, invT_max, (invT_max - invT_min) / 300],
                color=GREEN,
                use_smoothing=False,
                stroke_width=3,
            )
            # if Nd.get_value() % 10 in range(1, 9):
            #     scene.add(plot)
            return plot

        Ef_T = always_redraw(lambda: generate_Ef_T_plot(self.Nd, self))

        invT_min = 1000.0 / T_MAX
        invT_max = 1000.0 / max(T_MIN, 1.0)

        arrhenius_plot = always_redraw(lambda: generate_arrhenius_plot(self.Nd,self))
        self.add(
            semiconductor_bands,
            main_axes,
            arrhenius_axes,
            semiconductor_axes,
            tmp_sliders,
            Ef_T,
            arrhenius_plot,
        )
        self.play(self.Eg.animate.set_value(2), run_time=3)
        self.wait()
        self.play(self.Eg.animate.set_value(1.2), run_time=3)
        self.wait()
        self.play(self.Eg.animate.set_value(2),self.Ed.animate.set_value(1.8), run_time=3)

        self.wait(3)
        self.Na.set_value(1e14)
        self.play(self.Na.animate.set_value(1e16), run_time=5)
        self.wait(3)
        self.play(self.Ed.animate.set_value(1.8), run_time=5)
        self.wait()
        self.play(self.Nd.animate.set_value(18), run_time=5)
        self.wait()

    def generate_slider(self):
        slider_length = 3
        number_line_kwargs = {
            "rotation": PI / 2,
            "label_direction": LEFT,
            "include_numbers": True,
            "font_size": 25,
        }

        donor_concentration_slider = Slider(
            x_range=[0, 20, 4],
            length=slider_length,
            tracker=self.Nd,
            color=BLUE,
            label=r"N_D = ",
            number_label_scientific=True,
            numberline_kwargs={**number_line_kwargs},
        )
        acceptor_concentration_slider = Slider(
            x_range=[0, 20, 4],
            length=slider_length,
            tracker=self.Na,
            color=RED,
            label=r"N_A = ",
            number_label_scientific=True,
            numberline_kwargs={**number_line_kwargs},
        )
        defect_energy_slider = Slider(
            x_range=[0, 2.4, 0.4],
            length=slider_length,
            tracker=self.Ed,
            color=GOLD,
            label=r"E_d = ",
            post_label=r"[eV]",
            numberline_kwargs=number_line_kwargs,
        )
        band_gap_slider = Slider(
            x_range=[0, 2.4, 0.4],
            length=slider_length,
            tracker=self.Eg,
            color=YELLOW,
            label=r"E_g = ",
            post_label=r"[eV]",
            numberline_kwargs=number_line_kwargs,
        )
        return {
            "Nd": donor_concentration_slider,
            "Na": acceptor_concentration_slider,
            "Ed": defect_energy_slider,
            "Eg": band_gap_slider,
        }

    def generate_axes(self, T_MIN, T_MAX, T_STEP):
        x_length = 5
        main_axes = (
            Axes(
                x_range=[T_MIN, T_MAX + 1, T_STEP],
                x_length=x_length,
                y_range=[-1, 3, 0.5],
                y_length=3,
                y_axis_config={
                    "tip_shape": StealthTip,
                    "tip_height": 0.1,
                    "font_size": 25,
                },
                x_axis_config={
                    "tip_shape": StealthTip,
                    "tip_height": 0.1,
                    "font_size": 25,
                    "numbers_to_exclude": [T_MIN],
                },
            )
            .add_coordinates()
            .to_corner(UL, SMALL_BUFF * 3)
        )
        main_axes += main_axes.get_axis_labels(Text("T [K]",font_size=25), Text("E [eV]",font_size=25))#, font_size=25)

        invT_min = 1000.0 / T_MAX
        invT_max = 1000.0 / max(T_MIN, 1.0)
        invT_step = 1000 / T_STEP
        arrhenius_axes = (
            Axes(
                x_range=[invT_min, invT_max, invT_step],
                x_length=x_length,
                y_range=[4, 16, 4],
                y_length=3,
                y_axis_config={
                    "tip_shape": StealthTip,
                    "tip_height": 0.2,
                    "scaling": LogBase(),
                    "font_size": 25,
                },
                x_axis_config={
                    "tip_shape": StealthTip,
                    "tip_height": 0.2,
                    "font_size": 25,
                    "decimal_number_config": {"num_decimal_places": 1},
                },
            )
            .add_coordinates()
            # .next_to(main_axes,DOWN,LARGE_BUFF*0.75)
            .to_corner(DL, SMALL_BUFF)
            .shift(UP * SMALL_BUFF)
        )
        arrhenius_axes += arrhenius_axes.get_axis_labels(
            MathTex(r"1000/T [K^{-1}]", font_size=25),
            MathTex(r"log(n) [cm^{-3}]", font_size=25),
        )

        semiconductor_axes = (
            Axes(
                x_range=[0, 3, 1],
                x_length=5,
                y_range=[-1, 3, 0.5],
                y_length=3,
                y_axis_config={
                    "tip_shape": StealthTip,
                    "tip_height": 0.2,
                    "font_size": 30,
                    "include_numbers": True,
                },
                x_axis_config={
                    "tip_shape": StealthTip,
                    "tip_height": 0.2,
                    "font_size": 30,
                    "include_ticks": False,
                },
            )
            .next_to(main_axes, RIGHT)
            .to_edge(RIGHT, LARGE_BUFF)
        )
        semiconductor_axes += semiconductor_axes.get_axis_labels(
            Text("x", font_size=25), Text("E [eV]", font_size=25)
        )

        return main_axes, arrhenius_axes, semiconductor_axes

    def generate_semiconductors_look(
        self, main_axes, semiconductor_axes, T_MIN, T_MAX
    ) -> VGroup:
        valenced_band_kw = {
            "color_bottom": RED_A,
            "color_top": PURE_RED,
            "opacity_bottom": 0.2,
            "opacity_top": 0.9,
        }
        conduction_band_kw = {
            "color_bottom": PURE_BLUE,
            "color_top": BLUE_A,
            "opacity_bottom": 0.9,
            "opacity_top": 0.2,
        }
        main_valenced_band = TranslucentBand(
            main_axes, x0=T_MIN, x1=T_MAX, y0=-1, y1=0, **valenced_band_kw
        )
        main_conduction_band = always_redraw(
            lambda: TranslucentBand(
                main_axes,
                x0=T_MIN,
                x1=T_MAX,
                y0=self.Eg.get_value(),
                y1=self.Eg.get_value() + 0.5,
                **conduction_band_kw,
            )
        )

        semiconductor_axes_valened_band = TranslucentBand(
            axes=semiconductor_axes, x0=0, x1=2.5, y0=-1, y1=0, **valenced_band_kw
        )
        semiconductor_axes_conduction_band = always_redraw(
            lambda: TranslucentBand(
                axes=semiconductor_axes,
                x0=0,
                x1=2.5,
                y0=self.Eg.get_value(),
                y1=self.Eg.get_value() + 0.5,
                **conduction_band_kw,
            )
        )

        return VGroup(
            main_valenced_band,
            main_conduction_band,
            semiconductor_axes_valened_band,
            semiconductor_axes_conduction_band,
        )

    def calculate_concentration(self, T: float) -> float:
        """Return n(T) in cm^-3 using the same EF(T) neutrality solution."""
        T = max(float(T), 1e-6)
        kT = k_B * T
        Eg = float(self.Eg.get_value())
        NC = 1e19
        EF = self.Ef(T)
        expo = np.clip((EF - Eg) / kT, -700.0, 700.0)
        n = NC * np.exp(expo)
        # keep strictly positive for log y-axis
        return float(max(n, 1e-30))

    def Ef(self, T_K: float) -> float:
        Nd = 10 ** self.Nd.get_value()
        Na = 10 ** self.Na.get_value()
        Ec = self.Eg.get_value()
        Ed = self.Ed.get_value()

        kT = k_B * T_K
        ni2 = Nc * Nv * np.exp(-Ec / kT)  # ni^2 = Nc Nv exp(-Eg/kT), Eg=Ec

        def neutrality(Ef: float) -> float:
            n = Nc * np.exp(-(Ec - Ef) / kT)
            p = ni2 / n
            Nd_plus = Nd / (1.0 + np.exp((Ef - Ed) / kT))  # g_D = 1
            return p + Nd_plus - n - Na

        Ef0 = Ec - 0.2  # simple initial guess
        return float(fsolve(neutrality, x0=Ef0, xtol=1e-12, maxfev=200)[0])


