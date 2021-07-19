from deluca.lung.controllers.core import Controller
from deluca.lung.utils import BreathWaveform

import numpy as np
import torch

class ClippedAdvDeep(Controller):
    def __init__(
        self,
        clip=35.0,
        H=100,
        waveform=None,
        bptt=1,
        input_dim=1,
        activation=torch.nn.ReLU,
        history_len=15,
        kernel_size=5,
        **kwargs
    ):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, H, kernel_size),
            activation(),
            torch.nn.Flatten(),
            torch.nn.Linear(H * (history_len - kernel_size + 1), 1),
        )

        # linear feature transform:
        # errs -> [average of last h errs, ..., average of last 2 errs, last err]
        # emulates low-pass filter bank

        self.featurizer = torch.ones((2*history_len, history_len), requires_grad=False).tril_()
        self.featurizer /= torch.arange(history_len, 0, -1).unsqueeze(0)

        self.history_len = history_len
        self.input_dim = input_dim
        self.waveform = waveform or BreathWaveform()

        self.clip = clip

        self.reset()

    def reset(self):
        self.errs = [self.tensor(0.0)] * self.history_len
        self.pressures = [self.tensor(0.0)] * self.history_len
        self.targets = [self.tensor(0.0)] * self.history_len

    def update(self, key, state):
        getattr(self, key).append(state)

    def compute_action(self, state, t):
        target = self.tensor(self.waveform.at(t))
        self.errs.append(target - state)
        self.pressures.append(state)
        self.targets.append(target)

        decay = self.waveform.decay(t)

        if decay is None:
            trajectory = torch.stack(self.errs[-self.history_len :] + self.pressures[-self.history_len :]).unsqueeze(0).unsqueeze(0)
            u_in = self.model(trajectory @ self.featurizer)
        else:
            u_in = self.tensor(decay)
            self.errs = [self.tensor(0.0)] * self.history_len
            self.pressures = [self.tensor(0.0)] * self.history_len
            self.targets = [self.tensor(0.0)] * self.history_len

        u_in = torch.clamp(u_in, min=0.0, max=self.clip).squeeze()

        return (u_in, self.u_out(t))
