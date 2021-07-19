##########################################################
###################
################### SingleComp Lung Environment
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


class SingleCompLung(LungEnv):
    def __init__(self, resistance=15, compliance=50, min_volume=0.2, peep_value=5.0, **kwargs):
        # dynamics hyperparameters
        self.min_volume = min_volume
        self.P0 = 10.0
        self.peep_value = peep_value
        self.R = resistance
        self.C = compliance
        self.flow = 0.0

        # reset states
        self.reset()

    def reset(self):
        # volume is the only free parameter
        self.volume = self.min_volume
        self.pressure = (self.flow * self.R) + (self.volume / self.C) + self.P0

    def compute_aux_states(self):
        self.pressure = (self.flow * self.R) + (self.volume / self.C) + self.P0

    def step(self, u_in, u_out):
        dt = self.dt

        # calculate flow from u_in
        self.flow = torch.clamp(PropValve(u_in), 0, 2)
        # update volume by flow rate
        self.volume = self.volume + self.flow * dt

        # compute and record state
        self.compute_aux_states()

        return self.pressure
