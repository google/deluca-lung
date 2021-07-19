from deluca.lung.core import Controller


class Impulse(Controller):
    def __init__(self, impulse=50, start=0.5, end=0.65):
        self.impulse = impulse
        self.start = start
        self.end = end

    def action(self, pressure, target, t):
        impulse = 0
        if self.time >= self.start and self.time <= self.end:
            impulse = self.impulse

        return impulse
