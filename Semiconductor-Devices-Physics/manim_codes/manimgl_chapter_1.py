from typing import Callable, List, Tuple
from manimlib import *
import numpy as np
from manimlib.utils.color import interpolate_color

# Boltzmann constant in eV/K
k_B = 8.617333262145e-5

T_MIN, T_MAX, T_STEP = 0.0, 3000.0, 500.0
GRADIENT_STOPS: List[Tuple[float, Color]] = [
    (0.0, BLUE),
    (0.5, YELLOW),
    (1.0, RED),
]


class Variable(VMobject):  # , metaclass=ConvertToOpenGL):

    def __init__(
        self,
        var: float,
        label: str | TexText | TexText,
        var_type: DecimalNumber | Integer = DecimalNumber,
        num_decimal_places: int = 2,
        **kwargs,
    ):
        self.label = TexText(label) if isinstance(label, str) else label
        equals = TexText("=").next_to(self.label, RIGHT)
        self.label.add(equals)

        self.tracker = ValueTracker(var)

        if var_type == DecimalNumber:
            self.value = DecimalNumber(
                self.tracker.get_value(),
                num_decimal_places=num_decimal_places,
            )
        elif var_type == Integer:
            self.value = Integer(self.tracker.get_value())

        self.value.add_updater(lambda v: v.set_value(self.tracker.get_value())).next_to(
            self.label,
            RIGHT,
        )

        super().__init__(**kwargs)
        self.add(self.label, self.value)


def multi_stop_color(
    value: float, vmin: float, vmax: float, stops: List[Tuple[float, Color]]
) -> Color:
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


def temp_to_color(T: float) -> Color:
    return multi_stop_color(T, T_MIN, T_MAX, GRADIENT_STOPS)


class FermiDiracDistribution(Scene):
    def fermi_dirac(self, E: float, mu: float, T: float) -> float:
        # Stable evaluation (avoid overflow at very low T)
        x = (E - mu) / (k_B * T)
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(x))

    def construct(self):
        x_bounds = [0, 10]
        axes = Axes(
            x_range=[*x_bounds, 2],
            # x_length=8,
            height=8,
            y_range=[0, 1.2, 0.2],
            # y_length=5,
            width=5,
            # x_axis_config={"tip_shape": StealthTip, "tip_height": 0.2},
            # y_axis_config={"tip_shape": StealthTip, "tip_height": 0.2},
        )
        coord_labels = axes.add_coordinate_labels()
        axes.to_edge(LEFT)

        label_x = axes.get_axis_labels("E", "$\\mathbb{P}$")

        # Parameters
        mu = 5.0  # eV
        T_min, T_max, T_step = 1e-8, 3500, 500
        temps = np.arange(T_min, T_max, T_step)
        temperature = ValueTracker(T_min)  # start at 0 K

        def generate_fermi_dirac_plot(T: ValueTracker, scene: Scene) -> VMobject:
            plot = axes.get_graph(
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
        ).next_to(axes, RIGHT, LARGE_BUFF)

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

        temperature_TexText = Variable(
            temperature.get_value(),
            label=TexText("T [K]", font_size=30),
            var_type=Integer,
        )

        temperature_TexText.tracker = temperature

        def temperature_TexText_updator() -> Callable:
            def updator(mob, dt) -> None:
                mob.next_to(marker, RIGHT, buff=SMALL_BUFF)
                mob.value.set_color(temp_to_color(temperature.get_value()))

            return updator

        temperature_TexText.add_updater(temperature_TexText_updator())
        self.add(
            axes,
            label_x,
            coord_labels,
            slider_rect,
            marker,
            temperature_TexText,
            live_curve,
        )

        # --- Snapshots at 0K, 500K, 1000K (overlaid, semi-transparent) ---
        # def snapshot_at(Tsnap: ValueTracker) -> VMobject:
        #     return generate_fermi_dirac_plot(Tsnap,self)

        # Animate through T and take snapshots “externally into the graph”
        # 0 K snapshot
        for T in temps:
            self.play(
                temperature.animate.set_value(int(T)), run_time=1.5, rate_func=linear
            )
            # self.add(snapshot_at(temperature))
        self.play(slider_rect.animate.next_to(axes, RIGHT, LARGE_BUFF, aligned_edge=UP))
        # # 0 -> 500 K
        # snap500 = snapshot_at(temperature)
        # self.play(Create(snap500), run_time=1.0)

        # # 500 -> 1000 K
        # self.play(temperature.animate.set_value(1000.0), run_time=2.5, rate_func=smooth)
        # snap1000 = snapshot_at(temperature)
        # self.play(Create(snap1000), run_time=1.0)

        self.wait(0.5)
        self.interact()
