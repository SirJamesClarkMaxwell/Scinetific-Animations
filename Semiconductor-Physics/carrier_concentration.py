from manimlib import *


class CarrierConcentrationV1(InteractiveScene):
    def construct(self) -> None:
        
        self.frame.set_euler_angles(1.37580519e+00, 1.06666667e+00, 9.99200722e-16)

        k_B = 0.025
        E_c = 1.0
        E_min = E_c
        E_max = E_c + 0.5
        band_gap = ValueTracker(3)
        ef_tracker = ValueTracker(1.5)
        # mu = ValueTracker(2.9)
        T = ValueTracker(10)
        axes = Axes()
        # labels = axes.add_coordinate_labels()

        conduction_band = always_redraw(
            lambda: axes.get_parametric_curve(
                lambda t: [t, band_gap.get_value(), 0.0],
                t_range=[0, 3, 0.01],
                color=BLUE_D,
            )
        )
        valence_band = always_redraw(
            lambda: axes.get_parametric_curve(
                lambda t: [t, 0, 0.0], t_range=[0, 3, 0.01], color=RED_D
            )
        )
        fermi_level = always_redraw(
            lambda: DashedVMobject)(
                axes.get_parametric_curve(
                    lambda t: [t, ef_tracker.get_value(), 0.0],
                    t_range=[0, 3, 0.01],
                    color=YELLOW,
                )
            )

        def calculate_middle_band_gap(conduction_band, valence_band) -> np.ndarray:
            x, y, z = valence_band.get_end()
            # y -= (conduction_band.get_end()[1]-valence_band.get_end()[1])
            return np.array([x, y, z])

        # second_axis = always_redraw(lambda: Axes(x_range=[-3,5],y_range=[-3,3]))
        second_axis = Axes(x_range=[-3, 5], y_range=[-3, 3])

        def show_animations_second_axis(axis):
            self.play(axis.animate.move_to(valence_band.get_end()))
            point = axis.get_origin()
            self.play(Rotate(axis, PI / 2, RIGHT,about_point = point))
            self.play(Rotate(axis, -PI / 2, IN,about_point = point))
            # second_axis.rotate(PI / 2, RIGHT, about_point=point)

        # second_axis.rotate(-PI/2,IN,about_point=point)
        # axes_labels = always_redraw(lambda: second_axis.add_coordinate_labels())
        # axes_labels = second_axis.add_coordinate_labels()

        def fermi_dirac_function(E):
            return 1 / (
                np.exp((E - ef_tracker.get_value()) / (k_B * T.get_value())) + 1
            )

        def density_of_states(E):
            return np.sqrt(E - band_gap.get_value())
        def concentration(E):
            return fermi_dirac_function(E)*density_of_states(E)

        fermi_points = always_redraw(
            lambda: second_axis.get_graph(
                lambda x: fermi_dirac_function(x), x_range=[-2, 4]
            ).set_stroke(GREEN, width=1)
        )

        fermi_area = always_redraw(
            lambda: second_axis.get_area_under_graph(
                fermi_points,
                x_range=[conduction_band.get_end()[0], 4],
                fill_color=GREEN,
                fill_opacity=0.95,
            )
        )

        dos_points = always_redraw(
            lambda: second_axis.get_graph(
                lambda x:
                    density_of_states(x),
                    x_range=[conduction_band.get_end()[0], 4],
                    color = BLUE
                    
            )
        )

        dos_area = always_redraw(
            lambda: second_axis.get_area_under_graph(
                dos_points,
                x_range=[conduction_band.get_end()[0], 4],
                fill_color=BLUE,
                fill_opacity=0.5,
            )
        )
        
        concentration_plot = always_redraw(lambda:
            second_axis.get_graph(
                lambda E: concentration(E),
                    x_range=[conduction_band.get_end()[0], 4],
                    color = RED
                    
            ))
        concentration_area = always_redraw(
            lambda: second_axis.get_area_under_graph(
                concentration_plot,
                x_range=[conduction_band.get_end()[0], 4],
                fill_color=RED,
                fill_opacity=0.5,
            )
        )
        
        self.play(ShowCreation(axes))  # ,axes_labels)
        show_animations_second_axis(second_axis)
        # self.add(conduction_band, valence_band, fermi_level)

        # self.play(FadeIn(VGroup(axes)))
        self.wait()
        self.play(LaggedStart(*[Write(item)for item in [conduction_band, valence_band, fermi_level]],lag_ratio=0.75,run_time = 2))
        # self.wait()
        self.play(band_gap.animate.set_value(3),ef_tracker.animate.set_value(1.5))
        # self.wait()
        self.play(FadeIn(VGroup(fermi_points,fermi_area)))
        self.play(FadeIn(VGroup(dos_points,dos_area)))
        self.play(FadeIn(VGroup(concentration_plot,concentration_area)))
        # self.add(dos_points,dos_area,fermi_points, fermi_area,concentration_plot,concentration_area)
        self.play(ef_tracker.animate.set_value(2.9))
        # self.play(band_gap.animate.set_value(1.1),ef_tracker.animate.set_value(1.1/2))
        # self.play(band_gap.animate.set_value(3),ef_tracker.animate.set_value(3/2))
