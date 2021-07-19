##########################################################
###################
################### DoubleComp Lung Environment
###################
##########################################################

import torch
import numpy as np

from deluca.lung.core import LungEnv

# Sources:
# https://github.com/CohenLabPrinceton/Ventilator-Dev/blob/master/sandbox/HOWTO_RunController.ipynb
# https://github.com/CohenLabPrinceton/Ventilator-Dev/blob/master/vent/controller/control_module.py
# https://github.com/MinRegret/venti
def PropValve(x):  # copied from Controller.__SimulatedPropValve
    y = 3 * x
    y_torch = torch.from_numpy(np.array([0.03 * (y - 130)])).float()
    flow_new = 1.0 * (torch.tanh(y_torch) + 1)

    return torch.clamp(flow_new, 0.0, 1.72)


def Solenoid(x):  # copied from Controller.__SimulatedSolenoid
    if x > 0:
        return x / x
    else:
        return x * 0.0


class DoubleCompLung(LungEnv):
    def __init__(
        self, resistances=(0.1, 0.1), compliances=(0.1, 0.1), min_volume=5, peep_value=5, **kwargs
    ):
        # dynamics hyperparameters
        self.min_volume = min_volume
        self.P0 = peep_value
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
        dt = self.dt

        # calculate flow from u_in
        self.flow_pvs = self.flow
        self.flow = torch.clamp(PropValve(u_in), 0, 2)

        # update by flow rate
        self.volume = self.volume + self.flow * dt

        # compute and record state
        self.compute_aux_states()

        return self.pressure
