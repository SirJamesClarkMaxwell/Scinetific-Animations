from unicodedata import numeric
from manim import *

# from manim_slides import Slide


class VectorsDotProduct(Scene):  # Slide):
    def construct(self):
        _range = [0, 5.01, 0.5]
        length = 6
        axes = (
            NumberPlane(
                x_range=_range, y_range=_range, x_length=length, y_length=length
            )
            .add_coordinates()
            .to_edge(LEFT)
        )
        phi_tracker = ValueTracker(PI / 6)  # type: ignore
        a_x, a_y = ValueTracker(3), ValueTracker(0)
        b_x, b_y = ValueTracker(0), ValueTracker(3)
        vector_a = always_redraw(
            lambda: Arrow(start=axes @ ORIGIN, end=axes @ ([a_x.get_value(), a_y.get_value()]), buff=0, color=GREEN)  # type: ignore
        )
        vector_a_lable = always_redraw(
            lambda: MathTex(r"\vec{a}", color=GREEN)
            .move_to(vector_a.point_from_proportion(0.5))
            .shift(DOWN * MED_LARGE_BUFF)
        )
        vector_b = always_redraw(
            lambda: Arrow(
                start=axes @ ORIGIN,
                end=axes @ [b_x.get_value(), b_y.get_value()],
                buff=0,
                color=RED,
            )
        )

        vector_b_lable = always_redraw(
            lambda: MathTex(r"\vec{b}", color=RED)
            .move_to(vector_b.point_from_proportion(0.5))
            .shift(UP * MED_LARGE_BUFF)
        )
        angle = always_redraw(lambda: self.create_angle(vector_a, vector_b))
        label = always_redraw(
            lambda: self.create_angle_lable(vector_a, vector_b, angle, axes)
        )
        first_equation = MathTex(
            r"\vec{a}\circ\vec{b} = \vec{b}\circ\vec{a}",
            tex_to_color_map={r"\vec{a}": GREEN, r"\vec{b}": RED},
        ).next_to(axes,RIGHT,buff=MED_SMALL_BUFF,aligned_edge=UP)
        self.add(axes, vector_a, vector_b, vector_a_lable, vector_b_lable, angle, label,first_equation)
        self.play(Indicate(label))
        self.play(b_x.animate.set_value(5), b_y.animate.set_value(1))
        self.wait()

    def get_vectors_angle(self, vector_a: Arrow, vector_b: Arrow) -> float:
        a = vector_a.get_end() - vector_a.get_start()
        b = vector_b.get_end() - vector_b.get_start()
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cos_angle = np.clip(dot / (norm_a * norm_b), -1.0, 1.0)
        return np.arccos(cos_angle)

    def create_angle(self, vector_a: Arrow, vector_b: Arrow) -> VMobject:
        # Pusty kąt tylko gdy wektory są współliniowe i w tym samym kierunku
        a_dir = vector_a.get_end() - vector_a.get_start()
        b_dir = vector_b.get_end() - vector_b.get_start()
        if np.allclose(a_dir / np.linalg.norm(a_dir), b_dir / np.linalg.norm(b_dir)):
            return VMobject()
        return Angle(vector_a, vector_b, radius=2)

    def create_angle_lable(
        self, vector_a: Arrow, vector_b: Arrow, angle: Mobject, axes: NumberPlane
    ) -> MathTex:
        # Punkt w połowie łuku (współrzędne sceny)
        mid = angle.point_from_proportion(0.5)
        O = axes.c2p(0, 0)

        # Kierunek od środka do środka łuku
        dir_vec = mid - O
        r = np.linalg.norm(dir_vec) + 1e-8
        u = dir_vec / r  # wersor promieniowy
        theta = np.arctan2(u[1], u[0])  # kąt orientacji etykiety

        # Ile wysunąć na zewnątrz (dostosuj w razie potrzeby)
        # Spróbuj użyć promienia kąta, jeśli jest dostępny, w przeciwnym razie 2.0
        radius = getattr(angle, "radius", 2.0)
        extra_out = radius * 0.45 + MED_SMALL_BUFF  # bardziej na zewnątrz

        # Pozycja etykiety: środek łuku + promieniowo na zewnątrz
        pos = mid + u * extra_out

        # Wartość kąta (referencja do osi x)
        i_hat = Arrow(O, axes.c2p(1, 0), buff=0)
        radial = Arrow(O, mid, buff=0)
        angle_value = self.get_vectors_angle(i_hat, radial)  # [rad]
        deg = np.degrees(angle_value)

        label = MathTex(rf"\varphi = {deg:.1f}^\circ").set_color(YELLOW)
        # Ustaw pozycję i obróć tak, aby zachować orientację wektora O->mid
        label.move_to(pos).rotate(theta)

        return label


class WorkEnergyTheorem(Scene):  # Slide):
    def construct(self):
        pass


# with tempconfig({"quality": "low_quality", "preview": True}):
#     VectorsDotProduct().render()
