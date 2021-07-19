import numpy as np

from deluca.lung.core import LungEnv


def PropValve(x):  # copied from Controller.__SimulatedPropValve
    y = 3 * x
    flow_new = 1.0 * (np.tanh(0.03 * (y - 130)) + 1)
    return np.clip(flow_new, 0.0, 1.72)


def Solenoid(x):  # copied from Controller.__SimulatedSolenoid
    if x > 0:
        return x / x
    else:
        return x * 0.0


# balloon physics vent ported from Cohen lab's repository
# Sources:
# https://github.com/CohenLabPrinceton/ventilator-Dev/blob/master/sandbox/HOWTO_RunController.ipynb
# https://github.com/CohenLabPrinceton/ventilator-Dev/blob/master/vent/controller/control_module.py
class BalloonLung(LungEnv):
    def __init__(self, min_volume=6, PEEP=5, PC=40, RP=1, leak=False, **kwargs):
        # dynamics hyperparameters
        self.min_volume = min_volume
        self.PC = PC
        self.RP = RP
        self.P0 = 0.0
        self.leak = leak
        self.PEEP = PEEP

        self.r0 = (3 * self.min_volume / (4 * np.pi)) ** (1 / 3)
        # reset states
        self.reset()

    def reset(self):
        super().reset()
        # keep volume as the only free parameter
        self.volume = self.min_volume
        self.compute_aux_states()

    def compute_aux_states(self):
        # compute all other state vars, which are just functions of volume
        r = (3 * self.volume / (4 * np.pi)) ** (1 / 3)
        self.pressure = self.P0 + self.PC * (1 - (self.r0 / r) ** 6) / (self.r0 ** 2 * r)

    def step(self, u_in, u_out):
        # 2-dimensional action per timestep: PIP/PEEP voltages
        flow = np.clip(PropValve(u_in), 0, 2) * self.RP
        if self.pressure > self.PEEP:
            flow = flow - np.clip(Solenoid(u_out), 0, 2) * 0.05 * self.pressure

        # update by flow rate
        self.volume = self.volume + flow * self.dt

        # simulate leakage
        if self.leak:
            RC = 5
            s = self.dt / (RC + self.dt)
            self.volume = self.volume + s * (self.min_volume - self.volume)

        # compute and record state
        self.compute_aux_states()

        return self.pressure
