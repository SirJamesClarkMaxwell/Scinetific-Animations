from manim import *



class PNJunction(Scene):
    def setup(self):
        self.p_length = ValueTracker(2)
        self.n_length = ValueTracker(2)

        self.p_Eg = ValueTracker(1.2)
        self.n_Eg = ValueTracker(1.2)

        self.acceptor_tracker = ValueTracker(1e17)
        self.donor_tracker = ValueTracker(1e17)

        self.external_voltage = ValueTracker(0)

        return super().setup()

    def construct(self):
        # do x odjąć Wp
        pass

    def create_junction_animation(self):
        pass

    def from_homojunction_to_meatal_animation(self):
        pass

    def applied_voltage_animation(self):
        pass

    @property
    def Wn(self) -> float:
        pass

    @property
    def Wp(self) -> float:
        pass

    @property
    def Vbi(self) -> float:
        pass

    @property
    def potential(self):
        pass

    @property
    def E_field(self):
        pass

    @property
    def n_concentration(self):
        pass

    @property
    def p_concentration(self):
        pass

