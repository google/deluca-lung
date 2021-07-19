import numpy as np
from deluca.lung.core import Controller


class PeriodicImpulse(Controller):
    def __init__(self, period=2, duration=0.1, hold=0.5, amplitude=20, hold_amplitude=0):
        self.period = period
        self.duration = duration
        self.hold = hold
        self.amplitude = amplitude
        self.hold_amplitude = hold_amplitude

    def action(self, pressure, target, t):
        phase = self.time % self.period

        if phase < self.duration:
            u_in = self.amplitude
        elif phase < self.hold:
            u_in = self.hold_amplitude
        else:
            u_in = 0

        return u_in
