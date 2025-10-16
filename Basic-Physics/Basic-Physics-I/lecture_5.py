from turtle import update
from typing import Callable, Tuple, Dict, Any
from manim import *

# from manim_slides import SLide


class Point(VGroup):
    def __init__(self, omega: float, color: ManimColor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.omega = omega
        self.dot = Dot(color=color)
        self.add(self.dot)

    def dot_updater(
        self, ellipse_function: Callable, time_tracker: ValueTracker, axes: NumberPlane
    ) -> Callable[[Mobject, float], None]:
        def updater(mob: Mobject, dt: float) -> None:
            mob.move_to(axes @ (ellipse_function(self.omega * time_tracker.value)))

        return updater


class Vectors(VGroup):
    def __init__(
        self,
        axes: NumberPlane,
        point: Point,
        a: float,
        b: float,
        omega: float,
        time_tracker: ValueTracker,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.first_vector = Arrow(color=GREEN, buff=0,stroke_width = 4)
        self.second_vector = Arrow(color=RED, buff=0,stroke_width = 4)
        self.point = point
        self.t: ValueTracker = time_tracker
        self.axes = axes
        self.a, self.b, self.omega = a, b, omega
        self.active_updater: Callable | None = None
        self.active_updater_kwargs: Dict[str, Any] = {}

        self.add(self.first_vector, self.second_vector)

    def change_updaters(self, new_updater, **kwargs) -> None:
        if self.active_updater:
            self.remove_updater(self.active_updater)
        self.add_updater(new_updater(**kwargs))
        self.active_updater = new_updater
        self.active_updater_kwargs = kwargs

    def change_axes(self, new_axes: Axes):
        self.axes = new_axes
        self.remove_updater(self.active_updater)
        self.add_updater(self.active_updater, self.active_updater_kwargs)

    def normal_and_tangential_velocities_updater(self):
        def updater(mob, dt) -> None:
            v1: Arrow = mob.first_vector
            v2: Arrow = mob.second_vector
            ax = mob.axes
            om = mob.omega
            start = mob.point.get_center()
            t = mob.t.value
            vx_end = ax @ (
                ax.p2c(start)
                - 0.5* om * np.array([-mob.a * np.sin(om * t), mob.b * np.cos(om * t)])
            )
            vy_end = ax @ (
                ax.p2c(start)
                -0.5* om * -np.array([-mob.b * np.cos(om * t), -mob.a * np.sin(om * t)])
            )
            v1.put_start_and_end_on(start,vx_end)
            v2.put_start_and_end_on(start, vy_end)

        return updater

    def normal_and_tangential_accelerations_updater(self):
        def updater(mob, dt) -> None:
            v1: Arrow = mob.first_vector
            v2: Arrow = mob.second_vector
            ax = mob.axes
            om = mob.omega
            start = mob.point.get_center()
            t = mob.t.value
            ax_end = ax @ (
                ax.p2c(start)
                - 0.5
                * om**2
                * np.array([-mob.a * np.cos(om * t), -mob.b * np.sin(om * t)])
            )
            ay_end = ax @ (
                ax.p2c(start)
                - 0.5
                * om**2
                * -np.array([-mob.b * np.sin(om * t), -mob.a * np.cos(om * t)])
            )
            v1.put_start_and_end_on(start, ax_end)
            v2.put_start_and_end_on(start, ay_end)
        return updater

    def radial_and_transversal_velocities_updater(self):
        def updater(mob, dt) -> None:
            pass

        return updater

    def radial_and_transversal_accelerations_updater(self):
        def updater(mob, dt) -> None:
            pass

        return updater


class NormalAndTangentialVelocities(Scene):  # Slide):
    def construct(self):
        range_ = [-2, 2, 0.5]
        length = 7
        axes = NumberPlane(
            x_range=range_, y_range=range_, x_length=length, y_length=length
        )#.to_edge(LEFT, MED_LARGE_BUFF)
        zoomed_axes = NumberPlane(
            x_range=[-2, 0], y_range=[0, 2], x_length=length, y_length=length
        ).to_edge(LEFT, MED_LARGE_BUFF)

        ellipse = ParametricFunction(
            lambda t: axes @ (self.generate_ellipse_function(2, 1)(t)),
            t_range=[0, TAU, 0.01],
            color=YELLOW,
        )
        zoomed_ellipse = ParametricFunction(
            lambda t: zoomed_axes @ (self.generate_ellipse_function(2, 1)(t)),
            t_range=[PI / 2, PI, 0.01],
            color=YELLOW,
        )
        time_tracker = ValueTracker(TAU)
        point_c = Point(1, GREEN)
        point_c.add_updater(
            point_c.dot_updater(self.generate_ellipse_function(), time_tracker, axes)
        )
        vectors_c = Vectors(axes, point_c, 2, 1, point_c.omega, time_tracker)
        vectors_c.change_updaters(vectors_c.normal_and_tangential_velocities_updater)
        # vectors_c.change_updaters(vectors_c.normal_and_tangential_accelerations_updater)
        def tangential_unit_vector_updater(arrow:Arrow):
            def updater(mob:Arrow,dt:float):
                direction = arrow.get_end() - arrow.get_start()
                start = arrow.get_start()
                direction /=np.linalg.norm(direction)
                mob.put_start_and_end_on(start,start+direction)
            return updater

        tangential_unit_vector = Arrow(buff=0,color=PINK)
        tangential_unit_vector.add_updater(tangential_unit_vector_updater(vectors_c.first_vector))

        point_d = Point(2, RED)
        point_d.add_updater(
            point_d.dot_updater(self.generate_ellipse_function(), time_tracker, axes)
        )
        vectors_d = Vectors(axes, point_d, 2, 1, point_d.omega, time_tracker)
        vectors_d.change_updaters(vectors_d.normal_and_tangential_velocities_updater)

        self.add(axes, ellipse, point_c,point_d, vectors_c,vectors_d,tangential_unit_vector)
        self.play(time_tracker.animate.set_value(0), run_time=5, rate_func=linear)
        self.wait()
    def generate_ellipse_function(
        self, a: float = 2, b: float = 1
    ) -> Callable[float, [float, float]]:

        def ellipse_function(t: float) -> [float, float]:
            return a * np.cos(t), b * np.sin(t)

        return ellipse_function
