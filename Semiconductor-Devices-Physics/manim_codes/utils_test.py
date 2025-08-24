from manimlib import *
import sys, os

sys.path.append(os.path.dirname(__file__))
from utils import TranslucentBand, Slider  # ignore


class DemoTranslucentBand(Scene):
    def construct(self):
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5.5, 1],
            width=7,
            height=7,
            axis_config=dict(include_tip=False),
        ).to_edge(LEFT, buff=0.8)
        axes.add_coordinate_labels(font_size=24)
        x_lab = (
            Tex("x").scale(1.0).next_to(axes.x_axis.get_right(), UP, buff=LARGE_BUFF)
        )
        y_lab = (
            Tex("E").scale(1.0).next_to(axes.y_axis.get_top(), LEFT, buff=SMALL_BUFF)
        )
        self.add(axes, x_lab, y_lab)

        # bottom red bar (opaque bottom -> transparent top)
        red_band = TranslucentBand(
            axes,
            x0=0.0,
            x1=3.0,
            y0=0.0,
            y1=1.3,
            color_bottom=RED,
            color_top=RED,
            opacity_top=1,
            opacity_bottom=0.1,
            slices=140,
        )

        # cyan band above (slightly darker top hue + fade)
        cyan_band = TranslucentBand(
            axes,
            x0=0.0,
            x1=3.0,
            y0=2.6,
            y1=3.9,
            color_bottom=TEAL_B,
            color_top=TEAL_D,
            opacity_bottom=1,
            opacity_top=0.1,
            slices=140,
        )

        self.play(FadeIn(red_band, shift=UP * 0.2), FadeIn(cyan_band, shift=UP * 0.2))
        self.wait()


class DemoSlider(Scene):
    def construct(self):
        tracker = ValueTracker(0)
        slider = Slider(
            x_range=[0, 5, 1],
            length=5,
            tracker=tracker,
            label="T [K] = ",
            color = RED,
        )
        self.add(slider)
        self.play(tracker.animate.set_value(5))
        self.wait(0.5)
        self.play(tracker.animate.set_value(0))
        self.wait(0.5)
