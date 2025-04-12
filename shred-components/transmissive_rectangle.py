from manimlib import *

class TransmissiveRectangle(VGroup):
    def __init__(self, width:float, height:float, n_steps:int,
                start_color, end_color=BLACK, 
                opacity:float=1, direction=UP,
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        strips = []
        for i in range(n_steps):
            alpha = i / n_steps
            step_opacity = max(opacity * (1 - alpha), 0)
            strip = Rectangle(
                height=height / n_steps,
                width=width,
                fill_color=interpolate_color(start_color, end_color, alpha),
                fill_opacity=step_opacity,
                stroke_width=0
            )
            strips.append(strip)
        self.add(*strips)
        self.arrange(direction, buff=0)