
from manim import *  

from manim_physics import *
from manim_slides.slide import Slide


class BasicExample(Slide):
    def construct(self):
        circle = Circle(radius=3, color=BLUE)
        dot = Dot()

        self.play(GrowFromCenter(circle))
        self.next_slide()  # pylint: disable=...

        self.next_slide(loop=True)  

        self.play(MoveAlongPath(dot, circle), run_time=2, rate_func=linear)
        self.next_slide()  

        self.play(dot.animate.move_to(ORIGIN))

class SecondExample(Slide):    
    def construct(self):

        circle = Circle(radius=3, color=RED )
        dot = Dot()

        self.play(GrowFromCenter(circle))
        self.next_slide()  

        self.next_slide(loop=True, base_slide_config={})  # Start loop
        self.play(MoveAlongPath(dot, circle), run_time=2, rate_func=linear)
        self.next_slide()  

        self.play(dot.animate.move_to(ORIGIN))


class ElectricFieldExampleScene(Scene):
    def construct(self):
        charge1 = Charge(-1, LEFT + DOWN)
        charge2 = Charge(2, RIGHT + DOWN)
        charge3 = Charge(-1, UP)
        field = ElectricField(charge1, charge2, charge3)
        self.add(charge1, charge2, charge3)
        self.add(field)
