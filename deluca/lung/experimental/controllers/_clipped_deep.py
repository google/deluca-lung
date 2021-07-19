from deluca.lung.core import Controller
from deluca.lung import BreathWaveform

import numpy as np
import torch

class ClippedDeep(Controller):
    def __init__(
        self,
        clip=35.,
        H=100,
        waveform=None,
        bptt=1,
        input_dim=1,
        activation=torch.nn.ReLU,
        history_len=10,
        kernel_size=5,
        **kwargs
    ):
        super().__init__()

        self.clip = clip
        self.internal_time = 0
        self.last_t = 0

        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, H, kernel_size),
            activation(),
            torch.nn.Flatten(),
            torch.nn.Linear(H * (history_len - kernel_size + 1), 1),
        )

        # linear feature transform:
        # errs -> [average of last h errs, ..., average of last 2 errs, last err]
        # emulates low-pass filter bank

        self.featurizer = torch.ones((history_len, history_len), requires_grad=False).tril_()
        self.featurizer /= torch.arange(history_len, 0, -1).unsqueeze(0)

        self.history_len = history_len
        self.input_dim = input_dim
        self.waveform = waveform or BreathWaveform()

        self.breath_start = True

        self.reset()

    def reset(self):
        self.errs = [self.tensor(0.0)] * self.history_len

    def update(self, key, state):
        getattr(self, key).append(state)

    def compute_action(self, state, t):
        self.internal_time += t - self.last_t
        self.last_t = t

        decay = self.waveform.decay(t)

        if decay is None:

            if self.breath_start:
                self.errs = [self.tensor(0.0)] * self.history_len
                self.breath_start = False

            target = self.tensor(self.waveform.at(t))
            self.errs.append(target - state)
            self.errs.pop(0)

            trajectory = torch.stack(self.errs).unsqueeze(0).unsqueeze(0)
            u_in = self.model(trajectory @ self.featurizer)
        else:
            u_in = self.tensor(decay)
            self.breath_start = True
            self.internal_time = 0
            self.last_t = 0

        timed_clip = max(80-100*self.internal_time, self.clip)

        u_in = torch.clamp(u_in, min=0.0, max=timed_clip).squeeze()

        return (u_in, self.u_out(t))

    def train(
        self,
        sim,
        pip_feed="parallel",
        duration=3,
        dt=0.03,
        epochs=100,
        use_noise = False,
        optimizer=torch.optim.Adam,
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-4},
        loss_fn=torch.nn.L1Loss,
        loss_fn_params={},
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params={"factor":0.9, "patience":15},
        use_tqdm=True,
        print_loss=1,
        shuffle=False,
        device="cpu",
    ):
        optimizer = optimizer(self.parameters(), **optimizer_params)
        scheduler = scheduler(optimizer, **scheduler_params)
        loss_fn = loss_fn(**loss_fn_params)

        tt = torch.linspace(0, duration, int(duration / dt))
        losses = []

        torch.autograd.set_detect_anomaly(True)

        PIPs = [10, 15, 20, 25, 30, 35]
        PEEP = 5

        # TODO: handle device-awareness
        for epoch in range(epochs):

            if pip_feed == "parallel":
                self.zero_grad()

            for PIP in PIPs:

                if pip_feed == "sequential":
                    self.zero_grad()

                self.waveform = BreathWaveform((PEEP, PIP))

                self.reset()
                sim.reset()

                loss = torch.tensor(0.0, device=device, requires_grad=True)

                for t in tt:
                    sim.pressure += use_noise * torch.normal(mean=torch.tensor(2.), std=1.)
                    pressure = sim.pressure
                    u_in, u_out = self(pressure, t)
                    sim(u_in, u_out, t)

                    if u_out == 0:
                        loss = loss + loss_fn(torch.tensor(self.waveform.at(t)), pressure)

                if pip_feed == "sequential":
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    scheduler.step(loss)
                    per_step_loss = loss / len(tt)
                    losses.append(per_step_loss)

                    if epoch % print_loss == 0:
                        print(
                            f"Epoch: {epoch}, PIP: {PIP}\tLoss: {per_step_loss:.2f}\tLR: {optimizer.param_groups[0]['lr']}"
                        )

            if pip_feed == "parallel":
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step(loss)
                per_step_loss = loss / len(tt)
                losses.append(per_step_loss)

                if epoch % print_loss == 0:
                    print(
                        f"Epoch: {epoch}\tLoss: {per_step_loss:.2f}\tLR: {optimizer.param_groups[0]['lr']}"
                    )

        return losses
