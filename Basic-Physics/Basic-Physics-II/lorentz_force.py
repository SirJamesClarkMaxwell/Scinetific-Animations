from typing import Callable
from itertools import cycle
from enum import Enum
from sympy import Point3D
import numpy as np
from manimlib import *


class AxesEnumer(Enum):
    X = (0, "X")
    Y = (1, "Y")
    Z = (2, "Z")

    def index(self):
        return self.value[0]

    def label(self):
        return self.value[1]


class PlotModifiers(Enum):
    POSITION = (0, "Position", GREEN, lambda coords: coords)
    VELOCITY = (1, "Velocity", RED, lambda coords: np.column_stack([
        np.gradient(coords[:, 0]),  # dx/dt
        np.gradient(coords[:, 1]),  # dy/dt
        np.gradient(coords[:, 2])   # dz/dt
    ]))
    ACCELERATION = (2, "Acceleration", BLUE, lambda coords: np.column_stack([
        np.gradient(np.gradient(coords[:, 0])),  # d^2x/dt^2
        np.gradient(np.gradient(coords[:, 1])),  # d^2x/dt^2
        np.gradient(np.gradient(coords[:, 2])),  # d^2x/dt^2
    ]))

    def index(self):
        return self.value[0]

    def label(self):
        return self.value[1]

    def color(self):
        return self.value[2]

    def __call__(self, coords):
        return self.value[3](coords)


class LorentzForce(InteractiveScene):
    """_summary_

    Args:
        InteractiveScene (_type_): _description_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_val, self.max_val = 0.0, 4.0
        self.axes = (
            ThreeDAxes(
                x_range=[self.min_val, self.max_val, 1.0],
                y_range=[self.min_val, self.max_val, 1.0],
                z_range=[0.0, 3.0, 1.0],
            )
            # .rotate(angle=0.5 * np.pi, axis=LEFT, about_point=[0, 0, 0])
            # .shift(LEFT + DOWN)
        )
        self.axes.add_axis_labels()

    def setup(self):
        super().setup()
        self.starting_position = np.array([1.5,1.5, 0], dtype=np.float32)
        self.add(self.axes)
        self.camera.frame.reorient(
            0, 0, 0, (np.float32(2.32), np.float32(-0.03), np.float32(0.0)), 8.00
        )

    def construct(self):

        ## Scene controls
        B_field_tracker = ValueTracker(2.0)
        initial_velocity = [ValueTracker(2.0), ValueTracker(2.0), ValueTracker(0.0)]
        q = ValueTracker(1.0)
        mass = ValueTracker(1.0)
        alpha = ValueTracker(5)
        time_tracker = ValueTracker(6)
        time_stamp = 0.1

        ## Scene elements
        charge = DotCloud(
            [Point3D(*self.starting_position)],
            color=BLUE,
            radius=0.1,
        )
        B_field = VectorField(
            func=lambda p: np.column_stack(
                (
                    np.zeros_like(p[:, 0]),  # x-component
                    np.zeros_like(p[:, 1]),  # y-component
                    B_field_tracker.get_value() * np.ones_like(p[:, 2]),  # z-component
                )
            ),
            x_range=[-self.min_val, self.max_val - 1, 1],
            y_range=[-self.min_val, self.max_val - 1, 1],
            z_range=[0, 0.5, 1],
            coordinate_system=self.axes,
            density=1.0,
            color=GREEN,
        )
        trajectory = VMobject()

        postition_plots = VGroup([VMobject().fix_in_frame() for _ in range(3)])
        velocity_plots = postition_plots.copy()
        acceleration_plots = postition_plots.copy()
        ### plots
        axis_config = {
            "x_range": [0, time_tracker.get_value(), 1],
            "y_range": [-3, 3, 1],
            "height": 2,
            "width": 5,
        }
        plots_axes = VGroup(*[Axes(**axis_config) for _ in range(3)])
        plots_axes[0].to_corner(UP + RIGHT)

        for (i, plots), y_lbl in zip(
            enumerate(plots_axes), ["position", "velocity", "acceleration"]
        ):
            labels_kwargs = {"font_size": 20, "buff": SMALL_BUFF}
            if i != 0:
                plots.next_to(plots_axes[i - 1], DOWN, buff=0.5)
            labels = plots.get_axis_labels(
                x_label_tex="time [s]", y_label_tex=y_lbl, **labels_kwargs
            )
            labels.fix_in_frame()
            plots.fix_in_frame()

            self.add(plots, labels)

        ## updaters
        def trajectory_updater() -> Callable[..., None]:
            def generate_trajectory(
                charge: DotCloud, q, m, B, v0_vec, alpha_deg, total_time, dt
            ):
                alpha_rad = np.radians(alpha_deg)

                # Initial velocity vector
                v = np.array([
                    v0_vec[0],
                    v0_vec[1] * np.cos(alpha_rad),
                    v0_vec[1] * np.sin(alpha_rad),
                ])
                r = np.array(self.starting_position)
                B_vec = np.array([0.0, 0.0, B])

                def acceleration(v):
                    return (q / m) * np.cross(v, B_vec)

                points = [r.copy()]
                steps = int(total_time / dt)

                for _ in range(steps):
                    # RK4 integration
                    k1v = dt * acceleration(v)
                    k1r = dt * v

                    k2v = dt * acceleration(v + 0.5 * k1v)
                    k2r = dt * (v + 0.5 * k1v)

                    k3v = dt * acceleration(v + 0.5 * k2v)
                    k3r = dt * (v + 0.5 * k2v)

                    k4v = dt * acceleration(v + k3v)
                    k4r = dt * (v + k3v)

                    v += (k1v + 2 * k2v + 2 * k3v + k4v) / 6
                    r += (k1r + 2 * k2r + 2 * k3r + k4r) / 6

                    points.append(r.copy())

                return points



            def updater(mob, dt):
                # Extract values from trackers
                q_val = q.get_value()
                m_val = mass.get_value()
                B_val = B_field_tracker.get_value()
                alpha_val = alpha.get_value()
                v0 = [v.get_value() for v in initial_velocity]

                points = generate_trajectory(
                    charge,
                    q_val,
                    m_val,
                    B_val,
                    v0,
                    alpha_val,
                    time_tracker.get_value(),
                    time_stamp,
                )

                # Convert points to path
                trajectory.set_points_smoothly([self.axes.c2p(*p) for p in points])

            return updater

        def charge_updater(trajectory: VMobject) -> Callable[..., None]:
            def updater(mob):
                if trajectory.get_num_points() > 0:
                    mob.move_to(trajectory.get_end())
                else:
                    mob.move_to(self.axes.c2p(0, 0, 0))

            return updater

        charge.add_updater(charge_updater(trajectory))
        trajectory.add_updater(trajectory_updater())
        # Split Axes

        def plot_group_updater(
            axes: Axes,
            modifier: PlotModifiers,
        ) -> Callable[..., None]:
            def updater(mob: VGroup, dt: float):
                points = trajectory.get_points()
                if len(points) < 3:
                    for line in mob:
                        line.clear_points()
                    return

                # Convert trajectory points to coordinates in 3D
                coords = np.array([self.axes.p2c(p) for p in points])  # shape (N, 3)
                coords = modifier(coords)  # Apply modifier (position, velocity, acceleration)

                t = np.linspace(0, time_tracker.get_value(), len(coords))

                # Apply plot line updates for x, y, z components
                for dim_index, line in enumerate(mob):
                    y_vals = coords[:, dim_index]
                    plot_coords = [axes.c2p(t_val, y_val, 0) for t_val, y_val in zip(t, y_vals)]

                    line.set_points_smoothly(plot_coords)
                    line.set_color([GREEN, RED, BLUE][dim_index])  # color by axis

            return updater


        postition_plots.add_updater(plot_group_updater(plots_axes[0], PlotModifiers.POSITION))
        velocity_plots.add_updater(plot_group_updater(plots_axes[1], PlotModifiers.VELOCITY))
        acceleration_plots.add_updater(plot_group_updater(plots_axes[2], PlotModifiers.ACCELERATION))

        self.add(
            B_field,
            charge,
            trajectory,
            plots_axes,
            # position_velocity_acceleration_group
            postition_plots,
            velocity_plots,
            acceleration_plots,
        )

        ## Animations
        def play_time_animations(scene, run_time=5):
            scene.play(
                time_tracker.animate.set_value(run_time),
                run_time=run_time,
                rate_func=linear,
            )
            scene.play(
                time_tracker.animate.set_value(0),
                run_time=int(run_time / 2),
                rate_func=linear,
            )
            scene.wait()

        # Call the function
        # play_time_animations(self)
        # B_field_tracker.set_value(2.0)
        # play_time_animations(self)
        # B_field_tracker.set_value(1.0)
        # alpha.set_value(70.0)
        # play_time_animations(self)
        # self.play(time_tracker.animate.set_value(5), run_time=2, rate_func=linear)
        self.play(self.camera.frame.animate.reorient(3, 79, 0, (np.float32(2.27), np.float32(0.28), np.float32(1.63)), 8.00))
        for v in initial_velocity[:2]:
            self.play(v.animate.set_value(5.0),run_time = 2)
        self.play(B_field_tracker.animate.set_value(5.0), run_time=5)
        self.play(alpha.animate.set_value(15.0), run_time=5)
