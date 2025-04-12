from manimlib import *

class Pointer(VGroup):
    def __init__(self,
                mobject:Mobject,
                label:str,
                line_direction=(UP+RIGHT),line_length:float=1,
                dot_size:float = 0.5,color=YELLOW,
                stroke_width:float=1,track:bool=False,*args,**kwargs):
        super().__init__(*args,**kwargs)        
        start_point = mobject.get_center()
        middle_point = start_point+line_direction*line_length
        
        text = TexText(fr"{label}")
        dot = Dot(dot_size if dot_size>0 else 0.5).move_to(mobject.get_center())
        first_line = Line(start=start_point,end=middle_point,stroke_width = stroke_width)
        
        text_width = text.get_width()
        second_line = Line(start=middle_point,end = middle_point+text_width*RIGHT,stroke_width = stroke_width)
        text.next_to(second_line,UP,SMALL_BUFF)
        super().add(dot,first_line,second_line,text).set_color(color)
        if track:
            self.add_updater(lambda mob: dot.move_to(mobject.get_center()))
            self.add_updater(lambda mob: first_line.become(Line(start=start_point,end=middle_point,stroke_width = stroke_width)))
