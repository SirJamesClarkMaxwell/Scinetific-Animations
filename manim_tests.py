import timeit
import os

from manim import *
from manim_slides import Slide
from manim_physics import *


def run_manim_example() -> None:
    class CheckHealthDemo(Scene):
        def _inner_construct(self) -> None:
            banner = ManimBanner().shift(UP * 0.5)
            self.play(banner.create())
            self.wait(0.5)
            self.play(banner.expand())
            self.wait(0.5)
            text_left = Text("All systems operational!")
            formula_right = MathTex(r"\oint_{\gamma} f(z)~dz = 0")
            text_tex_group = VGroup(text_left, formula_right)
            text_tex_group.arrange(RIGHT, buff=1).next_to(banner, DOWN)
            self.play(Write(text_tex_group))
            self.wait(0.5)
            self.play(
                FadeOut(banner, shift=UP),
                FadeOut(text_tex_group, shift=DOWN),
            )

        def construct(self) -> None:
            self.execution_time = timeit.timeit(self._inner_construct, number=1)

    with tempconfig({"preview": True, "disable_caching": True}):
        scene = CheckHealthDemo()
        scene.render()


def run_manim_physics_example() -> None:

    class ElectricFieldExampleScene(Scene):
        def construct(self):
            charge1 = Charge(-1, LEFT + DOWN)
            charge2 = Charge(2, RIGHT + DOWN)
            charge3 = Charge(-1, UP)
            field = ElectricField(charge1, charge2, charge3)
            self.add(charge1, charge2, charge3)
            self.add(field)

    with tempconfig({"preview": True, "disable_caching": True}):
        scene = ElectricFieldExampleScene()
        scene.render()


def run_manim_slides_example() -> None:

    class BasicExample(Slide):
        def construct(self):
            circle = Circle(radius=3, color=BLUE)
            dot = Dot()

            self.play(GrowFromCenter(circle))
            self.next_slide()  # Waits user to press continue to go to the next slide

            self.next_slide(loop=True)  # Start loop
            self.play(MoveAlongPath(dot, circle), run_time=2, rate_func=linear)
            self.next_slide()  # This will start a new non-looping slide

            self.play(dot.animate.move_to(ORIGIN))


    with tempconfig({"preview": True, "disable_caching": True}):
        scene = BasicExample()
        os.system("manim-slides manim_tests.py BasicExample")
        os.system(f"manim-slides present BasicExample")

    # ...existing code...


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Manim examples.")
    parser.add_argument(
        "--example",
        choices=["basic", "physics", "slides", "all"],
        default="all",
        help="Which example to run",
    )
    args = parser.parse_args()

    if args.example == "basic":
        run_manim_example()
    elif args.example == "physics":
        run_manim_physics_example()
    elif args.example == "slides":
        run_manim_slides_example()

    else:
        run_manim_example()
        run_manim_physics_example()
        run_manim_slides_example()

if __name__ == "__main__":
    main()
