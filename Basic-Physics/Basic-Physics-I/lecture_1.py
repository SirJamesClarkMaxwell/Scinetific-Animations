from turtle import position
from typing import Callable, Dict, Any, List
from numpy import number
from pkg_resources import non_empty_lines
from typing_extensions import runtime
from manim import *
from manim_slides import Slide

from manim import Scene


class Clock(VGroup):
    def __init__(
        self,
        circ_radius: float = 0.4,
        clock_hand_length: float = 0.3,
        circ_tick_length: float = 0.2,
        tick_number: int = 4,
        start_time: float = 0.0,
        omega: float = PI / 4,  # rad per time-unit (overridden if complete_turn=True)
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.omega = omega
        self._bound_tracker: ValueTracker | None = None
        self._end_value: float | None = None
        self._complete_turn: bool = False

        self.circle = Circle(radius=circ_radius, color=YELLOW)
        inner = Circle(radius=circ_radius - circ_tick_length)

        self.circle_ticks = VGroup(
            *[
                Line(inner.point_at_angle(a), self.circle.point_at_angle(a))
                for a in np.arange(0, TAU, TAU / int(tick_number))
            ]
        )

        self.main_tick = Line(
            start=self.circle.get_center(),
            end=self.circle.get_center() + RIGHT * clock_hand_length,
            color=RED,
        )

        # numeric readout
        self.t = Variable(start_time, label="t").next_to(
            self.circle, RIGHT, MED_SMALL_BUFF
        )

        self.add(self.circle, *self.circle_ticks, self.main_tick, self.t)

        # default binding to internal readout
        self.bind_to_tracker(self.t.tracker)

    def bind_to_tracker(
        self,
        tracker: ValueTracker,
        end_value: float | None = None,
        *,
        complete_turn: bool = False,
    ):
        """
        Make the clock follow `tracker`. If `end_value` is given, the updater is removed
        when tracker reaches it. If `complete_turn=True` and `end_value` is given,
        the hand will make exactly one full revolution over [0, end_value].
        """
        self._bound_tracker = tracker
        self._end_value = end_value
        self._complete_turn = bool(complete_turn)

        if self._complete_turn and end_value is not None and end_value > 0:
            self.omega = TAU / end_value  # 2π over the whole duration

        if self._sync_with_tracker not in self.updaters:
            self.add_updater(self._sync_with_tracker)

    def _set_hand_for_time(self, t: float):
        c = self.circle.get_center()
        self.main_tick.put_start_and_end_on(c, c + RIGHT * self.main_tick.get_length())
        self.main_tick.rotate(-self.omega * t, about_point=c)

    def _sync_with_tracker(self, mob, dt):
        if self._bound_tracker is None:
            return
        current_t = float(self._bound_tracker.get_value())
        # keep numeric readout synced
        self.t.tracker.set_value(current_t)
        # orient the hand
        self._set_hand_for_time(current_t)

        # snap exactly at the end, then remove the updater
        if self._end_value is not None and current_t >= self._end_value - 1e-8:
            self._set_hand_for_time(self._end_value)  # <<< ensure exact final angle
            mob.remove_updater(self._sync_with_tracker)


FACTOR = 1
COLORS = [RED, GREEN, PURPLE]
START_TIME = 0
END_TIME = 1


class Animation1(Slide):

    def add_dot_updater(
        self,
        dot: Dot,
        func: Callable[float, float],
        numberline: NumberLine,
        time_tracker: ValueTracker,
    ) -> None:
        return dot.add_updater(
            lambda mob: mob.move_to(numberline.n2p(func(time_tracker.get_value())))
        )

    def construct(self):
        x_range = [START_TIME, END_TIME + 0.1, 0.1]
        numberline = NumberLine(
            x_range=[-0.1, 1.0, 0.1],
            length=8,
            # include_tip=True,
            # include_numbers=True,
            tip_shape=StealthTip,
            tick_size=0.15,
        )
        numberline.to_edge(UP, buff=MED_LARGE_BUFF).to_edge(LEFT)
        self.time_tracker = ValueTracker(START_TIME)
        clock = Clock(omega=5).next_to(numberline, RIGHT, MED_SMALL_BUFF)
        clock.bind_to_tracker(self.time_tracker, end_value=5.0)
        numberline.create_label("x", edge=RIGHT, direction=DOWN, buff=MED_SMALL_BUFF)

        balls = VGroup(*[Dot(color=c).move_to(numberline @ 0) for c in COLORS])
        position_functions: List[Callable] = [
            lambda time: FACTOR * time,
            lambda time: (FACTOR * time) ** 2,
            lambda time: np.sin(time * (PI / 2)),
        ]
        velocity_functions: List[Callable] = [
            lambda time: FACTOR,
            lambda time: 2 * FACTOR * time,
            lambda time: PI/2*np.cos(time * (PI / 2)),
        ]
        acceleration_functions: List[Callable] = [
            lambda time: 0,
            lambda time: 2 * FACTOR,
            lambda time: -(PI/2)**2 * np.sin(time * (PI / 2)),
        ]

        for ball, func in zip(balls, position_functions):
            self.add_dot_updater(ball, func, numberline, self.time_tracker)

        x_axis_conf: Dict[str, Any] = {"tip_shape": StealthTip}
        y_axis_conf: Dict[str, Any] = {"tip_shape": StealthTip}
        axis_config: Dict[str, Any] = {
            "x_length": 4,
            "y_length": 4,
            "x_axis_config": x_axis_conf,
            "y_axis_config": y_axis_conf,
        }
        position_axes = NumberPlane(
            x_range=x_range, y_range=[0, 1.1, 0.2], **axis_config
        )
        velocity_axes = NumberPlane(
            x_range=x_range, y_range=[0, 2.2, 0.3], **axis_config
        )
        acceleration_axes = NumberPlane(
            x_range=x_range, y_range=[-2.5, 2.5, 0.5], **axis_config
        )
        axes = (
            VGroup(position_axes, velocity_axes, acceleration_axes)
            .arrange(RIGHT, buff=MED_LARGE_BUFF)
            .to_edge(DOWN)
        )
        axes_labels = VGroup(
            *[
                ax.get_axis_labels("t", y_label)
                for ax, y_label in zip(axes, ["x", "v", "a"])
            ]
        )
        position_plots: VGroup = always_redraw(
            lambda: VGroup(
                *[
                    always_redraw(
                        lambda func=func, col=col: position_axes.plot(
                            func,
                            x_range=[START_TIME, self.time_tracker.get_value()],
                            color=col,
                        )
                    )
                    for func, col in zip(position_functions, COLORS)
                ]
            )
        )
        velocity_plots: VGroup = always_redraw(
            lambda: VGroup(
                *[
                    always_redraw(
                        lambda func=func, col=col: velocity_axes.plot(
                            func,
                            x_range=[START_TIME, self.time_tracker.get_value()],
                            color=col,
                        )
                    )
                    for func, col in zip(velocity_functions, COLORS)
                ]
            )
        )
        acceleration_plots: VGroup = always_redraw(
            lambda: VGroup(
                *[
                    always_redraw(
                        lambda func=func, col=col: acceleration_axes.plot(
                            func,
                            x_range=[START_TIME, self.time_tracker.get_value()],
                            color=col,
                        )
                    )
                    for func, col in zip(acceleration_functions, COLORS)
                ]
            )
        )
        dot_labels: VGroup = (
            VGroup(
                *[
                    MathTex(rf"x(t) =  {text}", color=col)
                    for (i, text), col in zip(
                        enumerate([r"t", r"t^2", r"\sin\left(\frac{\pi}{2}*t\right)"]),
                        COLORS,
                    )
                ]
            )
            .arrange(RIGHT, LARGE_BUFF)
            .next_to(numberline, DOWN, MED_LARGE_BUFF)
            .shift(RIGHT)
        )
        self.play(
            Succession(
                *[
                    FadeIn(it, shift=UP, run_time=1)
                    for it in [numberline, *axes, *axes_labels, clock]
                ],
                lag_ratio=0.25,
            )
        )
        # self.add(
        #     numberline, balls, axes, axes_labels, clock, position_plots, velocity_plots,acceleration_plots,dot_labels
        # )
        self.time_tracker.set_value(1)
        copy_obj = VGroup(
            *[
                it.copy().match_updaters(it)
                for it in [
                    *balls,
                    # *position_plots,
                    # *velocity_plots,
                    # *acceleration_plots,
                    *dot_labels,
                ]
            ]
        )
        self.time_tracker.set_value(0)

        for dot, pos_plot, vel_plot, acc_plot, label in zip(
            balls, position_plots, velocity_plots, acceleration_plots, dot_labels
        ):
            self.next_slide()
            self.play(
                Succession(
                    *[
                        FadeIn(it, run_time=1)
                        for it in [dot, pos_plot, vel_plot, acc_plot]
                    ],
                    Write(label, run_time=1),
                    lag_ratio=0.1,
                )
            )
            self.next_slide()
            self.play(
                self.time_tracker.animate.set_value(1), run_time=5, rate_func=linear
            )
            # copy_obj.add(pos_plot.copy(), vel_plot.copy(), acc_plot.copy())
            # for it in [pos_plot, vel_plot, acc_plot]:
            #     it.clear_updaters()
            #     it.add_updater(lambda mob: mob.set_opacity(0))
            #     it.set_opacity(0)
            self.next_slide()
            self.play(
                *[
                    FadeOut(it, shift=DOWN)
                    for it in [dot, pos_plot, vel_plot, acc_plot, label]
                ],
                self.time_tracker.animate.set_value(0),
                lag_ratio=0.1,
            )

        self.wait()
        # for it in [*position_plots,*velocity_plots,*acceleration_plots]: it.set_opacity(1)
        self.next_slide()
        self.add(*position_plots, *velocity_plots, *acceleration_plots)
        self.next_slide()
        self.play(
            Succession(
                *[FadeIn(it, shift=UP, run_time=1) for it in copy_obj], lag_ratio=0.3
            )
        )
        self.next_slide()
        self.play(
            self.time_tracker.animate.set_value(1), run_time=5.0, rate_func=linear
        )
        self.next_slide()
        # self.wait()
        self.play(FadeOut(Group(*self.mobjects), shift=DOWN, run_time=2))
        self.wait()


class Animation2(Scene):  # Slide):
    def construct(self):
        range_ = [-2, 2, 0.5]
        length = 7
        numberplane = (
            NumberPlane(
                x_range=range_, y_range=range_, x_length=length, y_length=length
            )
            .add_coordinates()
            .to_edge(LEFT)
        )
        numberplane_unit_length = (numberplane @ (1, 0) - numberplane @ (0, 0))[0]
        x_range = [0, 5, 1]
        y_range = [-1.25, 1.25, 0.5]
        x_length, y_length = 6, 2
        position_axes = NumberPlane(
            x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length
        )
        velocity_axes = NumberPlane(
            x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length
        )
        acceleration_axes = NumberPlane(
            x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length
        )
        axes = (
            VGroup(position_axes, velocity_axes, acceleration_axes)
            .arrange(DOWN, MED_LARGE_BUFF)
            .to_edge(RIGHT, 2 * SMALL_BUFF)
        )

        phi = ValueTracker(0)
        circle = Circle(
            radius=(numberplane @ (1, 0) - numberplane @ (0, 0))[0]
        ).move_to(numberplane @ (0, 0))
        dot = Dot().add_updater(self.dot_position_updater(phi, numberplane))
        start = numberplane @ (0, 0)
        position_vector = Arrow().add_updater(
            lambda mob: mob.become(
                Arrow(
                    start=start,
                    end=dot.get_center(),
                    buff=0,
                    color=YELLOW,
                )
            )
        )
        position_vector_label = MathTex(r"\vec{r}").add_updater(
            lambda mob: mob.move_to(position_vector.point_from_proportion(0.5)).shift(
                UP * MED_LARGE_BUFF
            )
        )

        x_pos_vector = Arrow().add_updater(
            lambda mob: mob.become(
                Arrow(
                    start=start,
                    end=numberplane @ (np.cos(phi.get_value()), 0),
                    color=GREEN,
                    buff=0,
                )
            )
        )
        x_pos_label = MathTex(r"x", color=GREEN).add_updater(
            lambda mob: mob.move_to(x_pos_vector.point_from_proportion(0.5)).shift(
                UP * MED_LARGE_BUFF
            )
        )
        y_pos_vector = Arrow().add_updater(
            lambda mob: mob.become(
                Arrow(
                    start=start,
                    end=numberplane @ (0, np.sin(phi.get_value())),
                    color=RED,
                    buff=0,
                )
            )
        )
        y_pos_label = MathTex(r"y", color=RED).add_updater(
            lambda mob: mob.move_to(y_pos_vector.point_from_proportion(0.5)).shift(
                LEFT * MED_LARGE_BUFF
            )
        )
        # position_vector.

        x_velocity_vector = Arrow().add_updater(
            lambda mob: mob.become(
                Arrow(
                    start=dot.get_center(), end=numberplane_unit_length * RIGHT, buff=0
                )
            )
        )
        y_velocity_vector = Arrow().add_updater(
            lambda mob: mob.become(
                Arrow(
                    start=dot.get_center(),
                    end=numberplane_unit_length * np.cos(phi.get_value()) * UP,
                    buff=0,
                )
            )
        )
        self.add(
            numberplane,
            circle,
            dot,
            # position_vector,
            # position_vector_label,
            # x_pos_vector,
            # x_pos_label,
            # y_pos_vector,
            # y_pos_label,
            axes,
            x_velocity_vector,
            y_velocity_vector,
        )
        self.wait()
        self.play(phi.animate.set_value(TAU), rate_func=linear, run_time=4)
        # self.wait()

    def dot_position_updater(
        self, phi_tracker: ValueTracker, numberplane: CoordinateSystem
    ) -> Callable[[Mobject, float], None]:
        def updater(mob: Mobject, dt: float) -> None:
            x_coord = np.cos(phi_tracker.get_value())
            y_coord = np.sin(phi_tracker.get_value())
            mob.move_to(numberplane @ (x_coord, y_coord))

        return updater


class Animation3(Slide):
    def construct(self):
        # ----- LEFT: plane, circle, moving dot -----
        rng = [-1.25, 1.25, 0.5]
        L = 7
        plane = (
            NumberPlane(x_range=rng, y_range=rng, x_length=L, y_length=L)
            .add_coordinates()
            .to_edge(LEFT)
        )
        unit = plane.x_axis.get_unit_size()
        center = plane.c2p(0, 0)
        circle = Circle(radius=unit, color=RED).move_to(center)

        t = ValueTracker(0.0)  # we take t = phi
        TREV = TAU

        dot = Dot().move_to(plane.c2p(1, 0))
        dot.add_updater(
            lambda m: m.move_to(plane.c2p(np.cos(t.get_value()), np.sin(t.get_value())))
        )

        # --- r vector stays visible always ---
        r_vec = always_redraw(
            lambda: Arrow(start=center, end=dot.get_center(), buff=0, color=YELLOW)
        )
        r_lbl = always_redraw(
            lambda: MathTex(r"\vec r").next_to(r_vec.point_from_proportion(0.6), UP)
        )

        # velocity & acceleration components (functions)
        def v_comp(tt):  # (vx, vy)
            return (-np.sin(tt), np.cos(tt))

        def a_comp(tt):  # (ax, ay)
            return (-np.cos(tt), -np.sin(tt))

        scale = 0.8 * unit
        x_vec = always_redraw(
            lambda: Arrow(
                center,
                plane@(np.cos(t.get_value()),0),
                buff=0,
                color=GREEN,
            )
        )
        y_vec = always_redraw(
            lambda: Arrow(
                center,
                plane @ (0,np.sin(t.get_value())),
                buff=0,
                color=RED,
            )
        )
        # --- Velocity vectors ---
        v_vec = always_redraw(
            lambda: Arrow(
                dot.get_center(),
                dot.get_center()
                + scale
                * (v_comp(t.get_value())[0] * RIGHT + v_comp(t.get_value())[1] * UP),
                buff=0,
                color=YELLOW,
            )
        )
        vx_vec = always_redraw(
            lambda: Arrow(
                dot.get_center(),
                dot.get_center() + scale * v_comp(t.get_value())[0] * RIGHT,
                buff=0,
                color=GREEN,
            )
        )
        vy_vec = always_redraw(
            lambda: Arrow(
                dot.get_center(),
                dot.get_center() + scale * v_comp(t.get_value())[1] * UP,
                buff=0,
                color=RED,
            )
        )

        # --- Acceleration vectors (including components you asked for) ---
        a_vec = always_redraw(
            lambda: Arrow(
                dot.get_center(),
                dot.get_center()
                + scale
                * (a_comp(t.get_value())[0] * RIGHT + a_comp(t.get_value())[1] * UP),
                buff=0,
                color=BLUE,
            )
        )
        ax_vec = always_redraw(
            lambda: Arrow(
                dot.get_center(),
                dot.get_center() + scale * a_comp(t.get_value())[0] * RIGHT,
                buff=0,
                color=GREEN,
            )
        )
        ay_vec = always_redraw(
            lambda: Arrow(
                dot.get_center(),
                dot.get_center() + scale * a_comp(t.get_value())[1] * UP,
                buff=0,
                color=RED,
            )
        )

        # ----- helper: progressive reveal curves (stable, no flicker) -----
        def reveal_curve(ax, f, x0, x1, tracker, color, sw=4, faint=0.15):
            full = ax.plot(f, x_range=[x0, x1], color=color, stroke_width=sw)
            live = full.copy()

            def _upd(m):
                alpha = np.clip((tracker.get_value() - x0) / (x1 - x0), 0, 1)
                m.pointwise_become_partial(full, 0, alpha)

            live.add_updater(_upd)
            return VGroup(full.set_opacity(faint), live)

        # ----- RIGHT: three rows of axes: position, velocity, acceleration -----
        ax_cfg = dict(
            x_range=[0, TREV, PI], y_range=[-1.2, 1.2, 0.5], x_length=6, y_length=2
        )
        ax_pos = NumberPlane(**ax_cfg)
        ax_vel = NumberPlane(**ax_cfg)
        ax_acc = NumberPlane(**ax_cfg)
        right = (
            VGroup(ax_pos, ax_vel, ax_acc)
            .arrange(DOWN, buff=MED_LARGE_BUFF)
            .to_edge(RIGHT, 2 * SMALL_BUFF)
        )

        labels = VGroup(
            ax_pos.get_axis_labels(MathTex("t"), MathTex("x,y")),
            ax_vel.get_axis_labels(MathTex("t"), MathTex("v_x,v_y")),
            ax_acc.get_axis_labels(MathTex("t"), MathTex("a_x,a_y")),
        )

        # functions of t
        fx, fy = (lambda tt: np.cos(tt), lambda tt: np.sin(tt))
        fvx, fvy = (lambda tt: -np.sin(tt), lambda tt: np.cos(tt))
        fax, fay = (lambda tt: -np.cos(tt), lambda tt: -np.sin(tt))

        # curves (GREEN = x, RED = y)
        pos_curves = VGroup(
            reveal_curve(ax_pos, fx, 0, TREV, t, color=GREEN),
            reveal_curve(ax_pos, fy, 0, TREV, t, color=RED),
        )
        vel_curves = VGroup(
            reveal_curve(ax_vel, fvx, 0, TREV, t, color=GREEN),
            reveal_curve(ax_vel, fvy, 0, TREV, t, color=RED),
        )
        acc_curves = VGroup(
            reveal_curve(ax_acc, fax, 0, TREV, t, color=GREEN),
            reveal_curve(ax_acc, fay, 0, TREV, t, color=RED),
        )

        # moving markers (optional but nice)
        def tracker_dot(ax, f, color):
            d = Dot(color=color, radius=0.05)
            d.add_updater(
                lambda m: m.move_to(ax.coords_to_point(t.get_value(), f(t.get_value())))
            )
            return d

        pos_marks = VGroup(tracker_dot(ax_pos, fx, GREEN), tracker_dot(ax_pos, fy, RED))
        vel_marks = VGroup(
            tracker_dot(ax_vel, fvx, GREEN), tracker_dot(ax_vel, fvy, RED)
        )
        acc_marks = VGroup(
            tracker_dot(ax_acc, fax, GREEN), tracker_dot(ax_acc, fay, RED)
        )

        # --- add static context (plane, circle, dot, right axes, labels, r vectors) ---
        self.add(plane, circle, dot, right, labels, r_vec, r_lbl,x_vec,y_vec)

        # ---------------- STEP 1: POSITION ONLY ----------------
        show_pos = VGroup(pos_curves, pos_marks)
        self.play(FadeIn(show_pos, shift=UP))
        self.next_slide()  # ready to start position step

        t.set_value(0)
        self.play(t.animate.set_value(PI/4), run_time=2, rate_func=linear)
        self.next_slide()  # discuss after full revolution
        self.play(t.animate.set_value(TREV), run_time=6, rate_func=linear)
        self.next_slide()  # discuss after full revolution

        # ---------------- STEP 2: VELOCITIES ONLY ----------------
        # keep r_vec visible; hide pos plots, show velocity vectors & plots
        self.play(
            FadeOut(show_pos, shift=DOWN),
            FadeIn(VGroup(v_vec, vx_vec, vy_vec, vel_curves, vel_marks), shift=UP),
        )
        t.set_value(0)
        self.next_slide()  # start velocities

        # quarter stops: 0 -> π/2 -> π -> 3π/2 -> 2π
        for target in [PI/4,PI / 2, PI, 3 * PI / 2, TREV]:
            self.play(t.animate.set_value(target), run_time=1.8, rate_func=linear)
            self.next_slide()  # pause at each quarter

        # ---------------- STEP 3: ACCELERATIONS ONLY ----------------
        self.play(
            FadeOut(VGroup(v_vec, vx_vec, vy_vec, vel_curves, vel_marks), shift=DOWN),
            FadeIn(VGroup(a_vec, ax_vec, ay_vec, acc_curves, acc_marks), shift=UP),
        )
        t.set_value(0)
        self.next_slide()  # start accelerations
        self.play(t.animate.set_value(PI/4), run_time=2, rate_func=linear)
        self.next_slide()  # start accelerations
        self.play(t.animate.set_value(TREV), run_time=6, rate_func=linear)
        self.next_slide()  # end of acceleration step

        # ---------------- FINAL: SHOW EVERYTHING ----------------
        # bring back position & velocity plots so ALL plots are visible together
        self.play(
            FadeIn(show_pos, shift=UP),
            FadeIn(VGroup(vel_curves, vel_marks), shift=UP),
        )
        self.next_slide()  # final discussion with all plots on screen
        self.wait()
