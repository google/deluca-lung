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
class DoubleCompLung(TorchLungEnv):
    def __init__(
        self,
        resistances=(0.1, 0.1),
        compliances=(0.1, 0.1),
        min_volume=5,
        peep_value=5,
        PC=40,
        RP=1,
        leak=False,
        **kwargs
    ):
        # dynamics hyperparameters
        self.min_volume = min_volume
        self.PC = PC
        self.RP = RP
        self.P0 = peep_value
        self.leak = leak
        self.peep_value = peep_value
        self.R1 = resistances[0]
        self.R2 = resistances[1]
        self.C1 = compliances[0]
        self.C2 = compliances[1]
        self.flow_pvs = 0.0
        self.flow = 0.0
        self.pressure_pvs = peep_value
        self.pressure = peep_value

        # reset states
        self.reset()

    def reset(self):
        # keep volume as the only free parameter
        self.volume = self.min_volume
        self.flow_pvs = 0.0
        self.flow = 0.0
        self.pressure_pvs = self.peep_value
        self.pressure = self.peep_value

    def compute_aux_states(self):
        # compute dp and df
        p_dot = (self.pressure - self.pressure_pvs) / self.dt
        f_dot = (self.flow - self.flow_pvs) / self.dt
        r1 = self.R1
        r2 = self.R2
        e1 = 1 / self.C1
        e2 = 1 / self.C2

        # compute pressure
        self.pressure = (
            (
                (f_dot * r1 * r2)
                + (self.flow * ((e2 * r1) + (e1 * (r1 + r2))))
                + (self.volume * e1 * e2)
                - (p_dot * r2)
            )
        ) / e2

    def step(self, u_in, u_out):
        if not isinstance(u_in, torch.Tensor):
            u_in = self.tensor(u_in)

        dt = self.dt

        # 2-dimensional action per timestep: PIP/PEEP voltages
        self.flow_pvs = self.flow

        self.flow = torch.clamp(PropValve(u_in), 0, 2) * self.RP

        # update by flow rate
        self.volume = self.volume + self.flow * dt

        # simulate leakage
        if self.leak:
            RC = 5
            s = dt / (RC + dt)
            self.volume = self.volume + s * (self.min_volume - self.volume)

        # compute and record state
        self.compute_aux_states()

        return self.pressure
