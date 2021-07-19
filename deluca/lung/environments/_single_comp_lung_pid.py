import torch
import numpy as np

from deluca.lung.core import TorchLungEnv


def PropValve(x):  # copied from Controller.__SimulatedPropValve
    y = 3 * x
    flow_new = 1.0 * (torch.tanh(0.03 * (y - 130)) + 1)
    return torch.clamp(flow_new, 0.0, 1.72)


def Solenoid(x):  # copied from Controller.__SimulatedSolenoid
    if x > 0:
        return x / x
    else:
        return x * 0.0


# balloon physics vent ported from Cohen lab's repository
# Sources:
# https://github.com/CohenLabPrinceton/Ventilator-Dev/blob/master/sandbox/HOWTO_RunController.ipynb
# https://github.com/CohenLabPrinceton/Ventilator-Dev/blob/master/vent/controller/control_module.py
class SingleCompLung(TorchLungEnv):
    def __init__(
        self,
        resistance=6.4,
        compliance=0.1,
        min_volume=0.2,
        peep_valve=5.0,
        PC=40.0,
        RP=1.0,
        leak=False,
        **kwargs
    ):
        # dynamics hyperparameters
        self.min_volume = self.tensor(min_volume)
        self.PC = self.tensor(PC)
        self.RP = self.tensor(RP)
        self.P0 = self.tensor(0.0)
        self.leak = leak
        self.peep_valve = self.tensor(peep_valve)
        self.R = self.tensor(resistance)
        self.C = self.tensor(compliance)
        self.flow = self.tensor(0.0, requires_grad=False)

        # reset states
        self.reset()

    def reset(self):
        self.flow = self.tensor(0.0, requires_grad=False)
        self.volume = self.min_volume
        self.p = (self.flow * self.R) + (self.volume / self.C) + self.P0
        self.pressure = self.tensor(self.p, requires_grad=False)

    def compute_aux_states(self):
        # compute all other state vars, which are just functions of volume
        self.p = (self.flow * self.R) + (self.volume / self.C) + self.P0

        self.pressure = self.tensor(self.p, requires_grad=False)

    def step(self, u_in, u_out):
        if not isinstance(u_in, torch.Tensor):
            u_in = self.tensor(u_in)
        dt = self.dt

        # 2-dimensional action per timestep: PIP/PEEP voltages
        self.flow = torch.clamp(PropValve(u_in), 0, 2) * self.RP

        # update by flow rate
        self.volume = self.volume + self.flow * dt

        # compute and record state
        self.compute_aux_states()

        return self.pressure
