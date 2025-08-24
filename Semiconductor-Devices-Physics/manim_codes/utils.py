from manim import *
import numpy as np
from typing import Iterable, List, Dict, Any, Callable

DEFAULT_MEDIUM_DOT_RADIUS = 0.04

class TranslucentBand(VGroup):
    """
    A rectangle in (x0..x1, y0..y1) with a vertical fade (bottom -> top).

    Parameters
    ----------
    axes : Axes
        Axes used to convert (x,y) to scene coords.
    x0, x1, y0, y1 : float
        Data-space bounds of the band.
    color_bottom : ManimColor
        Base color near the bottom edge.
    color_top : ManimColor | None
        Optional color at the top edge (if None uses color_bottom).
    opacity_bottom : float
        Fill opacity at the bottom edge (0..1).
    opacity_top : float
        Fill opacity at the top edge (0..1), e.g. 0.0 for “fade out”.
    slices : int
        Number of horizontal strips (more = smoother gradient).
    """

    def __init__(
        self,
        axes: Axes,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        color_bottom=RED,
        color_top=None,
        opacity_bottom=0.9,
        opacity_top=0.0,
        slices: int = 120,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.axes = axes
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.color_bottom = color_bottom
        self.color_top = color_top if color_top is not None else color_bottom
        self.opacity_bottom = float(opacity_bottom)
        self.opacity_top = float(opacity_top)
        self.slices = int(max(2, slices))
        self._rebuild()

    # ----- public API -----
    def set_bounds(self, x0: float, x1: float, y0: float, y1: float):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self._rebuild()
        return self

    def set_colors(self, bottom, top=None):
        self.color_bottom = bottom
        self.color_top = top if top is not None else bottom
        self._rebuild()
        return self

    def set_opacities(self, bottom: float, top: float):
        self.opacity_bottom, self.opacity_top = float(bottom), float(top)
        self._rebuild()
        return self

    # ----- internal -----
    def _rebuild(self):
        # self.clear()
        h = (self.y1 - self.y0) / self.slices
        for i in range(self.slices):
            t = i / (self.slices - 1)  # 0 at bottom -> 1 at top
            y_lo = self.y0 + i * h
            y_hi = y_lo + h
            poly = Polygon(
                self.axes.c2p(self.x0, y_lo),
                self.axes.c2p(self.x1, y_lo),
                self.axes.c2p(self.x1, y_hi),
                self.axes.c2p(self.x0, y_hi),
                stroke_width=0,
            )
            col = interpolate_color(self.color_bottom, self.color_top, t)
            opa = (1 - t) * self.opacity_bottom + t * self.opacity_top
            poly.set_fill(col, opacity=float(np.clip(opa, 0.0, 1.0)))
            self.add(poly)


# --- tiny helpers reused from your file ---
def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _interp_color_stops(a: float, stops: List[ManimColor]) -> ManimColor:
    if not stops:
        return WHITE
    if len(stops) == 1:
        return stops[0]
    a = _clamp01(a)
    seg = a * (len(stops) - 1)
    i = int(np.floor(seg))
    t = seg - i
    i = min(i, len(stops) - 2)
    return interpolate_color(stops[i], stops[i + 1], t)


class Slider(Group):
    def __init__(
        self,
        x_range: List[float],
        length: float,
        tracker: ValueTracker,
        color: ManimColor,
        label: str = "",
        post_label: str = "",
        label_font_size: int = 25,
        marker_size: float = 0.1,
        marker_direction=RIGHT,
        label_direction=DOWN,
        attach_label_to_marker: bool = False,
        number_label_scientific: bool = False,
        numberline_kwargs: Dict[str, Any] = {},
    ) -> None:

        self.numberline = NumberLine(x_range, length=length, **numberline_kwargs)

        self.tracker = tracker
        self.marker = self._make_marker(marker_size, marker_direction, color)
        self.label = self._make_label(
            label=label,
            post_label=post_label,
            label_font_size=label_font_size,
            number_label_scientific=number_label_scientific,
            color=color,
            label_direction=label_direction,
            attach_label_to_marker=attach_label_to_marker,
        )

        super().__init__(self.numberline, self.marker, self.label)

    def _make_marker(self, marker_size, marker_direction, color) -> VGroup:
        dot = Dot(
            self.numberline.n2p(self.tracker.get_value()), DEFAULT_MEDIUM_DOT_RADIUS
        ).set_color(color)
        dot.add_updater(
            lambda mob, dt: mob.move_to(self.numberline.n2p(self.tracker.get_value()))
        )

        marker = (
            Triangle()
            .rotate(PI / 2)
            .scale(marker_size)
            .next_to(dot, marker_direction, buff=SMALL_BUFF)
            .set_color(color)
            .set_fill(color, 1)
        )
        marker.add_updater(
            lambda mob, dt: mob.next_to(dot, marker_direction, SMALL_BUFF)
        )

        return Group(dot, marker)

    def _make_label(
        self,
        label,
        post_label,
        label_font_size,
        color: ManimColor,
        number_label_scientific: bool,
        label_direction=DOWN,
        attach_label_to_marker: bool = False,
        decimal_places: int = 1,
    ) -> VGroup:

        basic_label = MathTex(label, font_size=label_font_size)
        if not number_label_scientific:
            number_label = DecimalNumber(
                10**self.tracker.get_value(),
                font_size=label_font_size,
                num_decimal_places=decimal_places,
            )
            # number_label.add_updater(lambda mob: mob.set_value(self.tracker.get_value()))
        else:
            number_label =  MathTex(
                rf"{10**self.tracker.get_value():.1e}", font_size=label_font_size
            )
        post_label = MathTex(post_label, font_size=label_font_size)
        vg = (
            VGroup(basic_label, number_label, post_label)
            .arrange(RIGHT, SMALL_BUFF * 1.4)
            .set_color(color)
        )
        vg.next_to(self.numberline, label_direction, MED_LARGE_BUFF, aligned_edge=DOWN)

        if attach_label_to_marker:
            vg.add_updater(
                lambda mob, dt: mob.next_to(
                    self.marker, label_direction, SMALL_BUFF * 1.3
                )
            )

        return vg
