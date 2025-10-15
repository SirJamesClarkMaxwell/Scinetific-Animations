from ssl import CertificateError
from typing import Callable
from manim import *
from manim_slides import Slide
import os


ABS_PATH = os.path.abspath(os.curdir)
ant_path = os.path.join(ABS_PATH, "Basic-Physics", "Basic-Physics-I", "ant.svg")


class AntSpiralExample(Slide):
    def construct(self):
        self.next_slide()
        x_len, y_len = 6.5, 6.5
        zoomed_axes = (
            NumberPlane(
                x_range=[0, 1.09, 0.1],
                x_length=x_len,
                y_range=[0, 1.09, 0.1],
                y_length=y_len,
                y_axis_config={"label_direction": LEFT},
            )
            .add_coordinates()
            .to_corner(DL, MED_LARGE_BUFF)
            .shift(UP * SMALL_BUFF * 5)
        )

        ants = VGroup(
            SVGMobject(ant_path).set_color(WHITE).scale_to_fit_height(0.5)
            for _ in range(2)
        ).next_to(zoomed_axes, RIGHT, 3 * LARGE_BUFF)

        velocity_vector = Arrow(
            start=zoomed_axes @ ([1, 0]),
            end=zoomed_axes @ (0.5, 0.5),
            color=YELLOW,
            buff=0,
            stroke_width=3,
            tip_kwargs={"tip_shape": StealthTip},
        )
        velocity_vector_label = MathTex(
            r"\vec{v} = [\vec{v_r}, \vec{v_{\varphi}}}]",
            tex_to_color_map={
                r"\vec{v}": YELLOW,
                r"\vec{v_r}": RED,
                r"\vec{v_{\varphi}}}": GREEN,
            },
        ).next_to(
            Dot(velocity_vector.point_from_proportion(0.5)), LEFT, buff=MED_LARGE_BUFF
        )

        radial_velocity = Arrow(
            start=zoomed_axes @ ([1, 0]),
            end=zoomed_axes @ ([0.5, 0.0]),
            color=RED,
            buff=0,
            stroke_width=3,
            tip_kwargs={"tip_shape": StealthTip},
        )
        radial_velocity_lablel = MathTex(r"\vec{v}_r", color=RED).next_to(
            Dot(radial_velocity.point_from_proportion(0.5)), DOWN, buff=MED_LARGE_BUFF
        )
        transversal_velocity = Arrow(
            start=zoomed_axes @ ([1, 0]),
            end=zoomed_axes @ ([1.0, 0.5]),
            color=GREEN,
            buff=0,
            stroke_width=3,
            tip_kwargs={"tip_shape": StealthTip},
        )
        transversal_velocity_lablel = MathTex(
            r"\vec{v}_{\varphi}}", color=GREEN
        ).next_to(
            Dot(transversal_velocity.point_from_proportion(0.5)),
            RIGHT,
            buff=MED_LARGE_BUFF,
        )
        connecting_line = DashedLine(
            start=zoomed_axes @ (1, 0),
            end=zoomed_axes @ (0, 1),
            color=YELLOW,
            stroke_width=2,
        )
        angle_arc = ArcBetweenPoints(
            zoomed_axes @ (0.9, 0.1), zoomed_axes @ (1.0, 0.15), angle=-TAU / 4
        )
        angle_arc2 = ArcBetweenPoints(
            zoomed_axes @ (0.9, 0.1), zoomed_axes @ (1.0 - 0.15, 0), angle=TAU / 4
        )
        first_angle_label = (
            MathTex(r"\frac{\pi}{4}")
            .move_to(angle_arc.point_from_proportion(0.5))
            .shift((angle_arc.point_from_proportion(0.5) - zoomed_axes @ (1, 0)) * 1.1)
        )
        second_angle_label = (
            first_angle_label.copy()
            .move_to(angle_arc2.point_from_proportion(0.5))
            .shift((angle_arc2.point_from_proportion(0.5) - zoomed_axes @ (1, 0)) * 1.1)
        )
        self.play(
            LaggedStart(
                GrowFromPoint(zoomed_axes.get_axes()[0], point=zoomed_axes @ (0, 0)),
                GrowFromPoint(zoomed_axes.get_axes()[1], point=zoomed_axes @ (0, 0)),
                *[
                    GrowFromPoint(it, it.get_start())
                    for it in zoomed_axes.background_lines
                ],
                lag_ratio=0.1,
            )
        )
        self.next_slide()
        # self.wait()
        self.play(Create(ants))
        self.play(
            ants.animate.arrange(RIGHT).next_to(zoomed_axes, RIGHT, 3 * LARGE_BUFF)
        )
        self.play(
            *[
                ant.animate.move_to(zoomed_axes @ pos).set_color(col)
                for ant, pos, col in zip(ants, [[1, 0], [0, 1]], [RED, GREEN])
            ]
        )
        self.next_slide()
        # self.wait()
        # self.play(Create(beginning_arc))
        self.play(
            LaggedStart(
                GrowFromPoint(connecting_line, connecting_line.get_start()),
                GrowArrow(velocity_vector),
                Write(velocity_vector_label),
                lag_ratio=0.5,
            )
        )
        self.next_slide()
        # self.wait()
        self.play(
            LaggedStart(
                ReplacementTransform(velocity_vector.copy(), transversal_velocity),
                ReplacementTransform(
                    velocity_vector_label[2:4].copy(), transversal_velocity_lablel
                ),
                ReplacementTransform(velocity_vector.copy(), radial_velocity),
                ReplacementTransform(
                    velocity_vector_label[4:].copy().copy(), radial_velocity_lablel
                ),
                lag_ratio=0.5,
            )
        )
        self.next_slide()
        # self.wait()
        self.play(velocity_vector.animate.set_opacity(0))
        self.play(
            LaggedStart(
                GrowFromPoint(
                    angle_arc,
                    angle_arc.get_start(),
                ),
                FadeIn(first_angle_label),
                GrowFromPoint(angle_arc2, angle_arc2.get_end()),
                FadeIn(second_angle_label),
                lag_ratio=0.25,
            ),
            run_time=2,
        )
        self.play(
            LaggedStart(
                *[
                    FadeOut(it, shift=DOWN)
                    for it in [
                        angle_arc,
                        first_angle_label,
                        angle_arc2,
                        second_angle_label,
                    ]
                ],
                lag_ratio=0.15,
            ),
            run_time=2,
        )

        self.next_slide()
        # self.wait()

        self.play(
            LaggedStart(
                *[FadeOut(it, shift=DOWN) for it in self.mobjects], lag_ratio=0.1
            ),
            run_time=2,
        )
        self.wait()


DEFAULT_FONT_SIZE = 30


class Derivation(Scene):  # Slide):
    def construct(self):
        r_vector = MathTex(r"\vec{r} = [r\vec{e}_r]").to_corner(UL)
        versors = (
            VGroup(
                MathTex(
                    r"\vec{e}_r = \cos(\varphi(t)) \hat i + \sin(\varphi(t))\hat j"
                ),
                MathTex(
                    r"\vec{e}_r = -\sin(\varphi(t)) \hat i + \cos(\varphi(t))\hat j"
                ),
            )
            .arrange(RIGHT, MED_SMALL_BUFF)
            .next_to(r_vector, RIGHT, MED_SMALL_BUFF)
        )
        first_step = MathTex(
            r"\vec{v} = \frac{d\vec{r}}{dt} = \frac{d}{dt}\left( r \cdot \vec{e_r}\right) = \dot{r}\vec{e}_r + r\frac{d \vec{e}_r}{dt}"
        ).next_to(r_vector, DOWN)
        self.add(r_vector, versors, first_step)


a = 2

class FullAnimation(ZoomedScene, Slide):

    def __init__(self, **kwargs):
        self.zoom_tracker = ValueTracker(1)
        self.zoom_display_width = ValueTracker(3)

        self.zoom_display_height = ValueTracker(6)
        ZoomedScene.__init__(
            self,
            zoom_factor=self.zoom_tracker.get_value(),
            zoomed_display_height=7.75,
            zoomed_display_width=7.75,
            image_frame_stroke_width=20,
            zoomed_camera_config={
                "default_frame_stroke_width": 4,
            },
            **kwargs,
        )

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

    def r_function(self, t: float, v0: float) -> float:
        return ((a * np.sqrt(2)) / 2) * (a - v0 * t)

    def phi_function(self, phi_i0: float, t: float, v0: float) -> float:
        return phi_i0 + np.log((a) / (a - v0 * t))

    def construct(self):
        display, _, frame = self.zoomed_objs
        some_mob = self.change_stroke_width_zoomed_display()
        self.next_slide()
        self.add(some_mob)
        self.time_tracker = ValueTracker(0)
        v0 = 0.1
        scale_tracker = ValueTracker(0.1)
        self.transversal_velocity_factor = ValueTracker(1)
        length = 7
        start, step = 3.01, 1.0
        axes = (
            NumberPlane(
                x_range=[-start, start, step],
                x_length=length,
                y_range=[-start, start, step],
                y_length=length,
                y_axis_config={"label_direction": LEFT},
            )
            .add_coordinates()
            .to_corner(DL, MED_LARGE_BUFF)
        )
        display.scale_to_fit_width(5).next_to(axes, RIGHT, MED_SMALL_BUFF)
        frame.add_updater(lambda mob: mob.move_to(axes @ (0, 0)))
        ants_colors = [RED, GREEN, BLUE_D, YELLOW]
        ants_starting_angles = [0, PI / 2, PI, 1.5 * PI, 2 * PI]
        # ants = VGroup(*[Dot(ORIGIN, radius=0.05) for _ in range(4)])
        ants = VGroup(*[SVGMobject(ant_path).scale_to_fit_height(0.75).rotate(i*PI/2 ) for i in range(4)])
        for ant, color, angle in zip(ants, ants_colors, ants_starting_angles):
            ant.set_color(color)
            ant.add_updater(
                self.ant_updater(axes, self.time_tracker, v0, scale_tracker, angle)
            )
        paths = VGroup(
            *[
                TracedPath(
                    lambda ant=ant: ant.get_center(),
                    stroke_width=4,
                    stroke_color=ant.get_color(),
                )
                for ant in ants
            ]
        )

        radial_arrows = VGroup(*[Arrow(color=ant.get_color(),tip_kwargs={"tip_shape": StealthTip}) for ant in ants])
        transversal_arrows = VGroup(*[Arrow(color=ant.get_color(),tip_kwargs={"tip_shape": StealthTip}) for ant in ants])

        self.play(
            LaggedStart(
                GrowFromPoint(axes.get_axes()[0], point=axes @ (0, 0)),
                GrowFromPoint(axes.get_axes()[1], point=axes @ (0, 0)),
                *[GrowFromPoint(it, it.get_start()) for it in axes.background_lines],
                Create(ants),
                lag_ratio=0.1,
            )
        )
        self.next_slide()
        self.play(
            self.time_tracker.animate.set_value(19.7),
            run_time=10,
            rate_func=rate_functions.ease_out_expo,
        )
        self.next_slide()
        self.play(self.time_tracker.animate.set_value(0))
        self.wait()
        self.next_slide()

        for arrow, ant, initial_angle in zip(radial_arrows, ants, ants_starting_angles):
            arrow.add_updater(
                self.radial_velocity_arrow_updater(ant, initial_angle, v0, axes)
            )
        for arrow, ant, initial_angle in zip(
            transversal_arrows, ants, ants_starting_angles
        ):
            arrow.add_updater(
                self.transversal_velocity_arrow_updater(ant, initial_angle, v0, axes)
            )
        self.play(Create(paths))
        self.play(
            LaggedStart(
                *[GrowArrow(it) for it in [*radial_arrows]],
                lag_ratio=0.1,
            )
        )
        self.play(
            LaggedStart(
                *[GrowArrow(it) for it in [*transversal_arrows]],
                lag_ratio=0.1,
            )
        )
        self.update_self(1/self.duration)
        v_r_text = VGroup(*[MathTex(r"\vec{v}_r").next_to(it,dir,MED_SMALL_BUFF).match_color(it) for it,dir in zip(radial_arrows,[UP,LEFT,DOWN,RIGHT])])
        self.next_slide()
        self.play(
            LaggedStart(
                *[
                    FadeIn(it, rate_func=there_and_back_with_pause, run_time=2)
                    for it in v_r_text
                ],
                lag_ratio=0.4,
            ),
        )
        self.play(self.transversal_velocity_factor.animate.set_value(5))

        v_phi_text = VGroup(*[MathTex(r"\vec{v}_{\varphi}").next_to(it,dir,MED_SMALL_BUFF).match_color(it) for it,dir in zip(transversal_arrows,[LEFT,DOWN,RIGHT,UP])])
        self.next_slide()
        self.play(
            LaggedStart(
                *[
                    FadeIn(it, rate_func=there_and_back_with_pause, run_time=2)
                    for it in v_phi_text
                ],
                lag_ratio=0.4,
            ),
        )

        self.wait()
        self.next_slide()
        self.play(self.time_tracker.animate.set_value(10.8812),run_time=2)
        self.wait()
        self.next_slide()
        self.activate_zooming(animate=True)
        self.play(
            self.time_tracker.animate.set_value(19.98),
            scale_tracker.animate.set_value(0.0005),
            self.zoom(500),
            run_time=10,
            rate_func=rate_functions.ease_out_expo,
        )

        self.next_slide()
        self.wait()
        self.play(
            LaggedStart(
                *[
                    FadeOut(it, shift=DOWN)
                    for it in [
                        *ants,
                        *paths,
                        *axes.get_axes(),
                        *axes.background_lines,
                        *transversal_arrows,
                        *radial_arrows,
                    ]
                ],
                lag_ratio=0.1,
            ),
            run_time=2,
        )
        self.next_slide()
        self.wait()

    def ant_updater(
        self,
        nbpl: NumberLine,
        time_tracker: ValueTracker,
        v0: float,
        scale_tracker: ValueTracker,
        starting_angle: float,
        next_ant: SVGMobject = None,
        sprite_heading_offset: float = 0.0,  # adjust if your SVG "nose" isn't along +x
    ):
        def updator(mob: SVGMobject, dt: float) -> None:
            t = time_tracker.get_value()

            # Position on the logarithmic spiral
            r  = self.r_function(t, v0)
            phi = self.phi_function(starting_angle, t, v0)
            x, y = r * np.cos(phi), r * np.sin(phi)
            new_pos = nbpl @ (x, y)

            # Analytic velocity (no numerical noise)
            # r(t) = (a*sqrt(2)/2)*(a - v0*t)  =>  dr/dt = - (a*sqrt(2)/2) * v0
            drdt   = - (a * np.sqrt(2) / 2.0) * v0
            dphidt =  v0 / (a - v0 * t)       # from phi(t) = phi0 + ln(a/(a - v0 t))

            vx = drdt * np.cos(phi) - r * np.sin(phi) * dphidt
            vy = drdt * np.sin(phi) + r * np.cos(phi) * dphidt

            # Absolute heading
            heading = np.arctan2(vy, vx) + sprite_heading_offset

            # Initialize cached angle once
            if not hasattr(mob, "_heading"):
                mob._heading = heading

            # Rotate by the delta to reach the desired absolute angle
            delta = heading - mob._heading
            if abs(delta) > 1e-8:
                mob.rotate(delta, about_point=mob.get_center())
                mob._heading = heading

            # Move & scale
            mob.move_to(new_pos).scale_to_fit_height(4 * scale_tracker.get_value())

        return updator

    def radial_velocity_arrow_updater(
        self,
        ant: Dot,
        initial_angle: float,
        v0: float,
        axes: NumberPlane,
    ) -> Callable[[Arrow, float], None]:
        def updater(mob: Arrow, dt: float) -> None:
            t = self.time_tracker.get_value()
            r, phi = self.r_function(t, v0), self.phi_function(initial_angle, t, v0)
            x, y = r * np.cos(phi), r * np.sin(phi)
            start = np.array([x, y, 0])
            # end = (start-ORIGIN)/2
            end = (
                r * v0 * a * (np.sqrt(2) / 2) * np.array([np.cos(phi), np.sin(phi), 0])
            )

            it = Arrow(
                axes @ start,
                axes @ end,
                color=ant.get_color(),
                buff=0,
                max_stroke_width_to_length_ratio=1e2,
                tip_kwargs={"tip_shape": StealthTip}
            )
            mob.become(it)

        return updater

    def transversal_velocity_arrow_updater(
        self,
        ant: Dot,
        initial_angle: float,
        v0: float,
        axes: NumberPlane,
    ) -> Callable[[Arrow, float], None]:
        def updater(mob: Arrow, dt: float) -> None:
            t = self.time_tracker.get_value()
            r, phi = self.r_function(t, v0), self.phi_function(initial_angle, t, v0)
            x, y = r * np.cos(phi), r * np.sin(phi)
            start = np.array([x, y, 0])
            # end = (start-ORIGIN)/2

            end = (
               self.transversal_velocity_factor.get_value()*r * v0 * a * (np.sqrt(2) / 2) * np.array([-np.sin(phi), np.cos(phi), 0])
            )
            it = Arrow(
                axes @ start,
                axes @ (start + end),
                color=ant.get_color(),
                buff=0,
                max_stroke_width_to_length_ratio=1e2,
                tip_kwargs={"tip_shape": StealthTip}
            )
            mob.become(it)

        return updater

    def radial_acceleration_arrow_updater(
        self, arrow: Arrow, ant: Mobject, initial_angle: float, v0: float
    ) -> Callable[[Arrow, float], None]:
        def updater(mob: Arrow, dt: float) -> None:
            pass

        return updater

    def transversal_acceleration_arrow_updater(
        self, arrow: Arrow, ant: Mobject, initial_angle: float, v0: float
    ) -> Callable[[Arrow, float], None]:
        def updater(mob: Arrow, dt: float) -> None:
            pass

        return updater


# with tempconfig({"quality": "low_quality", "preview": True}):
#     FullAnimation().render()


class FractalTree(Scene):
    def construct(self):
        tree_group = VGroup()
        seg = Line(np.array([0, 0, 0]), np.array([0, 1, 0]), stroke_width=4)
        tree_group.add(seg)
        self.add(tree_group)
        self.play(
            AnimationGroup(
                tree_group.animate(rate_func=rate_functions.linear).shift(
                    np.array([-3, 0, 0])
                ),
            ),
            run_time=2,
            rate_func=rate_functions.linear,
        )
