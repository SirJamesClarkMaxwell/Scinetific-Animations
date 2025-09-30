from turtle import left
from typing import Callable, List, Sequence, Dict, Optional
from glm import sign
from manim import *
from manim.typing import Vector3D
from manim_slides.slide import Slide
from manim_physics import *


def checking_function(x: float) -> float:
    return (
        0.5
        * x
        * (x - 0.5)
        * (x - 2.5)
        * 0.5
        * (x - 4.0)
        # * (x-3.5)
        * (0.6 + np.exp(-0.8 * (x - 2) ** 2) + 0.04 * (x - 2))
    )


def bisection_one_step(
    a: float, b: float, func: Callable[[float], float], precision: float = 1e-10
) -> Dict[str, float]:
    f_a, f_b = func(a), func(b)
    x0 = (a + b) / 2
    y0 = func(x0)
    to_return = {"x0": x0, "y0": 0, "a": a, "b": b}
    if y0 < precision:
        return to_return
    elif sign(y0) == sign(f_a):
        to_return["a"] = x0
    elif sign(y0) == sign(f_b):
        to_return["b"] = x0
    return to_return


def bisection_one_step_animation() -> List[Animation]:
    return []


def get_T_label(
    axes: CoordinateSystem,
    graph: ParametricFunction,
    value: Optional[float] = None,
    tracker: Optional[ValueTracker] = None,
    variable: Optional[Variable] = None,
    label: str | Mobject | None = None,
    label_position: Vector3D | None = None,
    label_color: ParsableManimColor = WHITE,
    label_buff: float = LARGE_BUFF,
    triangle_size: float = MED_SMALL_BUFF,
    triangle_color: ParsableManimColor = GREEN,
    line_func: type[Line] = DashedLine,
    line_color: ParsableManimColor = YELLOW,
    update: bool = True,
) -> VGroup:

    def creator():
        t_label_group = VGroup()

        triangle = RegularPolygon(n=3, start_angle=np.pi / 2, stroke_width=0).set_fill(
            color=triangle_color,
            opacity=1,
        )
        triangle.height = triangle_size
        if tracker is None and variable is None and value is not None:
            raise ValueError("Tracker/variable/value can't be both None")

        if variable is not None:
            x_val = variable.tracker.get_value()
        elif tracker is not None:
            x_val = tracker.get_value()
        elif value is not None:
            x_val = value
        else:
            x_val = 0

        position = graph.get_point_from_function(x_val)
        triangle.move_to(axes.c2p(x_val, 0), UP)
        if axes.p2c(position)[1] < 0:
            triangle.rotate(PI, about_point=triangle.get_top())

        v_line = axes.get_line_from_axis_to_point(
            index=0,
            point=[*position],
            color=line_color,
            line_func=line_func,
        )
        t_label_text = (
            (label or "") + f"{x_val:1}"
            if isinstance(label, str) or label is None
            else f"{x_val:1}"
        )
        t_label = MathTex(t_label_text, color=label_color)
        if label_position is None:
            direction = UP if axes.p2c(position)[1] < 0 else DOWN
            t_label.next_to(triangle, direction, buff=label_buff)
        else:
            t_label.move_to(label_position)

        t_label_group += triangle
        t_label_group += v_line
        t_label_group += t_label
        return t_label_group

    t_group: VGroup = creator()

    if update:
        t_group.become(creator())

    return t_group


class BisectionMethod(ZoomedScene): 
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

    def change_stroke_width_zoomed_display(self):
        mob = Mobject()
        init_stroke = self.zoomed_camera.cairo_line_width_multiple

        def updater(_):
            scale = self.zoom_tracker.get_value()
            self.zoomed_camera.cairo_line_width_multiple = scale * init_stroke

        mob.add_updater(updater)
        return mob

    def setup(self):
        super().setup()
        zoomed_display = self.zoomed_display
        zoomed_display_border = zoomed_display.display_frame
        frame = self.zoomed_camera.frame
        self.zoomed_objs = [zoomed_display, zoomed_display_border, frame]

    def zoom(self, zoom_level: float, animate: bool = True):
        if animate:
            return self.zoomed_objs[-1].animate.scale(
                1 / zoom_level
            ), self.zoom_tracker.animate.set_value(
                self.zoom_tracker.get_value() * 1 / zoom_level
            )

    def construct(self):
        _, _, frame = self.zoomed_objs
        some_mob = self.change_stroke_width_zoomed_display()
        self.add(some_mob)

        axes = (
            Axes(
                x_range=[0, 4.5, 1],
                y_range=[-3, 3, 1],
                x_length=7,
                y_length=4,
                x_axis_config={"numbers_to_exclude": [0]},
            )
            .add_coordinates()
            .to_corner(UL)
            .shift(LEFT * SMALL_BUFF * 3)
        )

        plot = axes.plot(checking_function, x_range=[0, 5])
        left_bound, right_bound = ValueTracker(1), ValueTracker(3)
        left_bound_label = get_T_label(
            axes,
            plot,
            tracker=left_bound,
            label="a = ",
            label_buff=LARGE_BUFF * 0.5,
            line_func=DashedLine,
            line_color=GREEN,
        )

        right_bound_label = get_T_label(
            axes,
            plot,
            tracker=right_bound,
            label="b = ",
            label_buff=LARGE_BUFF * 0.5,
            triangle_color=RED,
            line_func=DashedLine,
            line_color=RED,
        )
        x0 = ValueTracker().add_updater(
            lambda mob: mob.set_value(
                (left_bound.get_value() + right_bound.get_value()) / 2
            )
        )
        y0 = ValueTracker().add_updater(
            lambda mob: mob.set_value(checking_function(x0.get_value()))
        )
        f_a = ValueTracker().add_updater(
            lambda mob: mob.set_value(checking_function(left_bound.get_value()))
        )
        f_b = ValueTracker().add_updater(
            lambda mob: mob.set_value(checking_function(right_bound.get_value()))
        )

        def gen_code_text(
            a: ValueTracker, b: ValueTracker, y0: ValueTracker, f_a: ValueTracker
        ) -> str:
            def fmt(v: float, digits: int = 3) -> str:
                return f"{v:.{digits}f}"

            return f"""
                a, b = {fmt(a.get_value())}, {fmt(b.get_value())}
                f_a, f_b = f({fmt(a.get_value(),1)}), f({fmt(b.get_value(),1)})
                x0 = (a + b)/2
                y0 = f(x0)
                while abs(y0) > 1e-10:
                    if {int(np.sign(y0.get_value()))} == {int(np.sign(f_a.get_value()))}:
                        a = x0
                    else:
                        b = x0
                """

        bisection_code = always_redraw(
            lambda: Code(
                code_string=gen_code_text(left_bound, right_bound, y0, f_a),
                language="python",
                tab_width=2,
                background="window",
                paragraph_config={"font_size": 15},
            ).to_corner(UR, buff=SMALL_BUFF)
        )
        code_pointer = Arrow(
            start=ORIGIN, end=RIGHT, tip_shape=StealthTip, color=YELLOW
        ).next_to(bisection_code.get_line_numbers()[0], LEFT, SMALL_BUFF)
        self.add(
            axes,
            plot,
            bisection_code,
            left_bound_label,
            right_bound_label,
            code_pointer,
        )
        # self.play(a.animate.set_value(2))
        # self.wait()
        # self.play(b.animate.set_value(3.5))
        # self.wait()


class FalsiRule(Slide):
    def construct(self):
        pass


class SecantSlopeMethod(Slide):
    def construct(self):
        pass


class NewtonMethod(Slide):
    def construct(self):
        pass


class ConstantPointMethod(Slide):
    def construct(self):
        pass


class NonlinearSystemOfEquations(ThreeDScene):
    def construct(self):
        pass


# with tempconfig({"quality": "low_quality", "preview": True}):
#     Test().render()
