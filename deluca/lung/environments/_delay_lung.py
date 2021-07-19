import numpy as np

from deluca.lung.core import LungEnv


# 2nd attempt at vent simulation
# observed pressure comes from delayed control signal, with back-pressure from deluca.lung
class DelayLung(LungEnv):
    def __init__(
        self, R=10, C=6, delay=10, min_volume=1.5, inertia=0.995, control_gain=0.02, **kwargs
    ):
        # dynamics hyperparameters
        self.R = R
        self.C = C
        self.min_volume = min_volume
        self.inertia = inertia
        self.control_gain = control_gain
        self.delay = delay

        # reset states
        self.r0 = (3 * self.min_volume / (4 * np.pi)) ** (1 / 3)
        self.reset()

    def reset(self):
        super().reset()
        self.volume = self.min_volume
        self.pipe_pressure = 0.

        self.controls_in, self.controls_out = [], []
        self.compute_aux_states()

    def compute_aux_states(self):
        # compute all other state vars, which are just functions of volume
        r = (3 * self.volume / (4 * np.pi)) ** (1 / 3)
        self.vent_pressure = self.C * (1 - (self.r0 / r) ** 6) / (self.r0 ** 2 * r)

        if len(self.controls_in) < self.delay:
            self.pipe_impulse = 0
            self.peep = 0
        else:
            self.pipe_impulse = self.control_gain * self.controls_in[-self.delay]
            self.peep = self.controls_out[-self.delay]

        self.pipe_pressure = self.inertia * self.pipe_pressure + self.pipe_impulse
        self.pressure = np.maximum(0., self.pipe_pressure - self.vent_pressure)

        if self.peep:
            self.pipe_pressure *= 0.995

    def step(self, u_in, u_out):
        u_in = np.maximum(0., u_in)

        self.controls_in.append(u_in)
        self.controls_out.append(u_out)

        # 2-dimensional action per timestep
        flow = self.pressure / self.R

        # update by flow rate
        self.volume = np.maximum(self.volume + flow * self.dt, self.min_volume)

        # compute and record state
        self.compute_aux_states()

        return self.pressure