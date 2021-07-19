import torch

from deluca.lung.core import Controller, LungEnv


class PIDCorrection(Controller):
    def __init__(self, base_controller: Controller, sim: LungEnv, pid_K=[0.0, 0.0], decay=0.1, **kwargs):
        self.base_controller = base_controller
        self.sim = sim
        self.I = 0.0
        self.K = pid_K
        self.decay = decay

        self.reset()

    def reset(self):
        self.base_controller.reset()
        self.sim.reset()
        self.I = 0.0

    def compute_action(self, state, t):
        u_in_base, u_out = self.base_controller(state, t)

        err = self.sim.pressure - state
        self.I = self.I * (1 - self.decay) + err * self.decay

        pid_correction = self.K[0] * err + self.K[1] * self.I

        u_in = torch.clamp(u_in_base + pid_correction, min=0.0, max=100.0)
        self.sim(u_in, u_out, t)

        return u_in, u_out
