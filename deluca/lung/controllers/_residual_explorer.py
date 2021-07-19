import torch
import numpy as np
from deluca.lung.core import Controller
from deluca.lung.utils.core import BreathWaveform
from deluca.lung.utils.core import Phase


class ResidualExplorer(Controller):
    def __init__(self, base_controller, waveform, **kwargs):
        self.base_controller = base_controller

        self.delta_range = (-30, 30)
        self.duration_range = (0.1, 0.5)

        self.boundary_prob = 0.15
        self.boundary_delta_range = (0, 100)
        self.boundary_duration_range = (0.05, 0.3)

        self.needs_refresh = False

        self.waveform = waveform
        self.refresh_exploration_perturbation()

    def refresh_exploration_perturbation(self):
        if np.random.random() < self.boundary_prob:
            self.resample_boundary_delta()
        else:
            self.resample_delta()

    def resample_delta(self):
        # random triangular bump
        height = np.random.uniform(*self.delta_range)
        duration = np.random.uniform(*self.duration_range)

        self.set_delta(height, duration)

    def set_delta(self, height, duration):
        t_min, t_max = 0, self.waveform._keypoints[2]

        t_begin = np.random.uniform(t_min, t_max - duration)
        t_mid = t_begin + 0.5 * duration
        t_end = t_begin + duration

        self.xp = np.array([t_min, t_begin, t_mid, t_end, t_max])
        self.fp = np.array([0.0, 0.0, height, 0.0, 0.0])

    def resample_boundary_delta(self):
        # random lean-to triangular bump at inspiratory boundary
        height = np.random.uniform(*self.boundary_delta_range)
        duration = np.random.uniform(*self.boundary_duration_range)

        self.set_boundary_delta(height, duration)

    def set_boundary_delta(self, height, duration):
        t_min, t_max = 0, self.waveform._keypoints[2]

        self.xp = np.array([t_min, duration, t_max])
        self.fp = np.array([height, 0, 0])

    def action(self, pressure, target, t):
        base_u_in = self.base_controller(pressure, target, t)

        if isinstance(base_u_in, torch.Tensor):
            base_u_in = base_u_in.detach().numpy()

        if (
            self.waveform.phase(t) == Phase.PEEP or self.waveform.phase(t) == Phase.RAMP_DOWN
        ):  # if base exhales, just follow base
            self.u_in = base_u_in
            if self.needs_refresh:
                self.refresh_exploration_perturbation()
                self.needs_refresh = False
        else:  # if base inhales, add residual to cycle phase
            self.needs_refresh = True
            self.u_in = base_u_in + np.interp(t % self.waveform.period, self.xp, self.fp)

        return np.clip(self.u_in, self.min, self.max)
