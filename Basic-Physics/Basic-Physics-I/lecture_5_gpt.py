from typing import Callable, Tuple
from manim import *
import numpy as np

# ---------- helpers ----------


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v[:2])
    return v / (n if n > eps else 1.0)


def rot90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0], 0.0])


# ---------- moving point ----------


class Point(VGroup):
    def __init__(self, omega: float, color: ManimColor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.omega = omega
        self.dot = Dot(color=color, radius=0.07)
        self.add(self.dot)

    def dot_updater(
        self,
        ellipse_fn: Callable[[float], Tuple[float, float]],
        time_tracker: ValueTracker,
        axes: NumberPlane,
    ):
        def updater(mob: Mobject, dt: float) -> None:
            t = self.omega * time_tracker.get_value()
            x, y = ellipse_fn(t)
            mob.move_to(axes.c2p(x, y))

        return updater


# ---------- vectors with modes ----------


class Vectors(VGroup):
    """
    Pokazuje dwie strzałki (zielona i czerwona) dla wybranego trybu:
    - 'vt_vn'  : v_t, v_n (Frenet)
    - 'at_an'  : a_t, a_n (Frenet)
    - 'vr_vphi': v_r, v_phi (biegunowe)
    - 'ar_aphi': a_r, a_phi (biegunowe)
    """

    def __init__(
        self,
        axes: NumberPlane,
        point: Point,
        a: float,
        b: float,
        omega: float,
        time_tracker: ValueTracker,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.axes = axes
        self.point = point
        self.t = time_tracker
        self.a, self.b, self.omega = a, b, omega

        self.first = Arrow(color=GREEN, buff=0.0, max_tip_length_to_length_ratio=0.18)
        self.second = Arrow(color=RED, buff=0.0, max_tip_length_to_length_ratio=0.18)
        self.add(self.first, self.second)

        # Skale do wygodnego podglądu
        self.scale_v = ValueTracker(1.0)
        self.scale_a = ValueTracker(1.0)

        # aktywny tryb
        self.mode = "vt_vn"

        # główny updater
        self.add_updater(self._main_updater)

    # --- kinematyka elipsy ---

    def _pos(self, t: float) -> np.ndarray:
        th = self.omega * t
        return np.array([self.a * np.cos(th), self.b * np.sin(th), 0.0])

    def _vel(self, t: float) -> np.ndarray:
        th = self.omega * t
        w = self.omega
        return np.array([-self.a * w * np.sin(th), self.b * w * np.cos(th), 0.0])

    def _acc(self, t: float) -> np.ndarray:
        th = self.omega * t
        w2 = self.omega * self.omega
        return np.array([-self.a * w2 * np.cos(th), -self.b * w2 * np.sin(th), 0.0])

    def _er(self, r: np.ndarray) -> np.ndarray:
        return unit(r)

    def _ephi(self, er: np.ndarray) -> np.ndarray:
        return unit(rot90(er))

    def _that(self, v: np.ndarray) -> np.ndarray:
        return unit(v)

    def _nhat(self, that: np.ndarray, a_vec: np.ndarray) -> np.ndarray:
        n_cand = unit(rot90(that))
        sign = 1.0 if np.dot(a_vec[:2], n_cand[:2]) >= 0 else -1.0
        return n_cand * sign

    # --- bezpieczne ustawianie strzałek ---

    def _place_arrow_safe(
        self,
        arrow: Arrow,
        start_xy: np.ndarray,
        vec: np.ndarray,
        fallback_dir: np.ndarray,
        eps: float,
        scale: float,
    ):
        v = vec * scale
        L = np.linalg.norm(v[:2])
        start3 = self.axes.c2p(*start_xy[:2])

        if L < eps:
            # wygaszamy strzałkę zamiast rysować "pchełkę"
            arrow.set_opacity(0.0)
            # ale utrzymujemy geometrię stabilną
            end3 = self.axes.c2p(*(start_xy[:2] + unit(fallback_dir)[:2] * eps))
            arrow.put_start_and_end_on(start3, end3)
        else:
            arrow.set_opacity(1.0)
            end3 = self.axes.c2p(*(start_xy[:2] + v[:2]))
            arrow.put_start_and_end_on(start3, end3)

    # --- API ---

    def set_mode(self, mode: str):
        self.mode = mode

    # --- główny updater ---

    def _main_updater(self, mob: "Vectors", dt: float):
        t = self.t.get_value()
        r = self._pos(t)
        v = self._vel(t)
        a = self._acc(t)

        er = self._er(r)
        ephi = self._ephi(er)
        that = self._that(v)
        nhat = self._nhat(that, a)

        # wektory składowe
        if self.mode == "vt_vn":
            vt = np.dot(v[:2], that[:2]) * that
            vn = np.dot(v[:2], nhat[:2]) * nhat  # zwykle ~0
            s_v = self.scale_v.get_value()
            self._place_arrow_safe(
                self.first, self.point.get_center(), vt, that, eps=1e-3, scale=s_v
            )
            self._place_arrow_safe(
                self.second, self.point.get_center(), vn, nhat, eps=1e-3, scale=s_v
            )

        elif self.mode == "at_an":
            at_ = np.dot(a[:2], that[:2]) * that
            an_ = np.dot(a[:2], nhat[:2]) * nhat
            s_a = self.scale_a.get_value()
            self._place_arrow_safe(
                self.first, self.point.get_center(), at_, that, eps=1e-3, scale=s_a
            )
            self._place_arrow_safe(
                self.second, self.point.get_center(), an_, nhat, eps=1e-3, scale=s_a
            )

        elif self.mode == "vr_vphi":
            vr = np.dot(v[:2], er[:2]) * er
            vphi = np.dot(v[:2], ephi[:2]) * ephi
            s_v = self.scale_v.get_value()
            self._place_arrow_safe(
                self.first, self.point.get_center(), vr, er, eps=1e-3, scale=s_v
            )
            self._place_arrow_safe(
                self.second, self.point.get_center(), vphi, ephi, eps=1e-3, scale=s_v
            )

        elif self.mode == "ar_aphi":
            ar = np.dot(a[:2], er[:2]) * er
            aphi = np.dot(a[:2], ephi[:2]) * ephi
            s_a = self.scale_a.get_value()
            self._place_arrow_safe(
                self.first, self.point.get_center(), ar, er, eps=1e-3, scale=s_a
            )
            self._place_arrow_safe(
                self.second, self.point.get_center(), aphi, ephi, eps=1e-3, scale=s_a
            )


# ---------- Scene ----------


class NormalAndTangentialVelocities(Scene):
    def construct(self):
        range_ = [-2, 2, 0.5]
        length = 7
        axes = (
            NumberPlane(
                x_range=range_, y_range=range_, x_length=length, y_length=length
            )
            .to_edge(LEFT, MED_LARGE_BUFF)
            .add_coordinates()
        )

        a, b = 2.0, 1.0
        ellipse_fn = self.generate_ellipse_function(a, b)

        ellipse = ParametricFunction(
            lambda t: axes.c2p(*ellipse_fn(t)),
            t_range=[0, TAU, 0.01],
            color=YELLOW,
        )

        time = ValueTracker(0.0)

        P = Point(omega=1.0, color=WHITE)
        P.add_updater(P.dot_updater(ellipse_fn, time, axes))

        vecs = Vectors(axes, P, a, b, omega=1.0, time_tracker=time)

        # UI: tytuł/tryb i legenda
        mode_text = always_redraw(
            lambda: Text(
                {
                    "vt_vn": "Frenet: v_t, v_n",
                    "at_an": "Frenet: a_t, a_n",
                    "vr_vphi": "Polar: v_r, v_φ",
                    "ar_aphi": "Polar: a_r, a_φ",
                }[vecs.mode]
            )
            .scale(0.5)
            .to_corner(UR)
            .set_opacity(0.9)
        )
        legend = (
            VGroup(
                VGroup(
                    Line(color=GREEN), Text("1st vector", color=GREEN).scale(0.45)
                ).arrange(RIGHT, buff=0.2),
                VGroup(
                    Line(color=RED), Text("2nd vector", color=RED).scale(0.45)
                ).arrange(RIGHT, buff=0.2),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            .to_corner(UR)
            .shift(DOWN * 0.6)
        )

        self.add(axes, ellipse, P, vecs, mode_text, legend)

        # start: prędkości Freneta
        vecs.set_mode("vt_vn")
        self.play(time.animate.set_value(TAU), run_time=5, rate_func=linear)

        # pokaż skale: najpierw polar v_r,v_φ
        vecs.set_mode("vr_vphi")
        self.wait(0.8)
        self.play(time.animate.set_value(2*TAU), run_time=5, rate_func=linear)

        # przyspieszenia Freneta – ustaw skalę przyspieszeń (często większe)
        vecs.scale_a.set_value(0.8)
        vecs.set_mode("at_an")
        self.wait(0.8)
        self.play(time.animate.set_value(3*TAU), run_time=5, rate_func=linear)

        # przyspieszenia polarne
        vecs.set_mode("ar_aphi")
        self.wait(0.8)
        self.play(time.animate.set_value(4*TAU), run_time=5, rate_func=linear)
        self.wait(0.8)

    def generate_ellipse_function(
        self, a: float = 2.0, b: float = 1.0
    ) -> Callable[[float], Tuple[float, float]]:
        def ellipse_function(t: float) -> Tuple[float, float]:
            return (a * np.cos(t), b * np.sin(t))

        return ellipse_function
