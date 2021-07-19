import torch

from deluca.lung.controllers import PID
from deluca.lung.core import TorchController
from deluca.lung.utils import BreathWaveform
import itertools
import numpy as np
import random
from scipy.special import softmax
from deluca.lung.controllers import Expiratory


class DeepPIDResidual(TorchController):
    def __init__(
            self,
            H=100,
            waveform=None,
            bptt=1,
            input_dim=1,
            activation=torch.nn.ReLU,
            history_len=10,
            kernel_size=5,
            normalize=False,
            time_as_feature=False,
            u_scaler=None,
            p_scaler=None,
            pid_K=[3.0, 4.0, 0.0],
            **kwargs
    ):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, H, kernel_size),
            activation(),
            torch.nn.Flatten(),
            torch.nn.Linear(H * (history_len - kernel_size + 1), 1),
        )

        self.time_as_feature = time_as_feature
        if self.time_as_feature:
            self.time_model = torch.nn.Sequential(torch.nn.Linear(1, 10), activation(), torch.nn.Linear(10, 1))

        multiplier = self.tensor(1.0)
        self.residual_multiplier = torch.nn.Parameter(multiplier)

        # linear feature transform:
        # errs -> [average of last h errs, ..., average of last 2 errs, last err]
        # emulates low-pass filter bank

        self.featurizer = torch.ones((history_len, history_len), requires_grad=False).tril_()
        self.featurizer /= torch.arange(history_len, 0, -1).unsqueeze(0)

        self.history_len = history_len
        self.input_dim = input_dim
        self.waveform = waveform or BreathWaveform()
        self.pid_base = PID(K=pid_K)
        self.normalize = normalize
        if normalize:
            self.u_scaler = u_scaler
            self.p_scaler = p_scaler

        self.reset()

    def reset(self):
        self.errs = [self.tensor(0.0)] * self.history_len
        self.pid_base.waveform = self.waveform
        self.pid_base.reset()

    def update(self, key, state):
        getattr(self, key).append(state)

    def compute_action(self, state, t):
        self.pid_base.waveform = self.waveform
        target = self.tensor(self.waveform.at(t))
        if self.normalize:
            target_scaled = self.p_scaler.transform(target).squeeze()
            state_scaled = self.p_scaler.transform(state).squeeze()
            self.errs.append(target_scaled - state_scaled)
        else:
            self.errs.append(target - state)

        decay = self.waveform.decay(t)

        if decay is None:
            trajectory = torch.stack(self.errs[-self.history_len:]).unsqueeze(0).unsqueeze(0)
            u_in_base= self.pid_base.action(state, self.waveform.at(t),t)
            u_in_residual = self.model(trajectory @ self.featurizer)
            if self.time_as_feature:
                u_in_residual += self.time_model(self.tensor(self.cycle_phase(t)).unsqueeze(0))
            # u_in = u_in_base + u_in_residual * 0.1 * torch.relu(torch.tanh(5e-2 * self.residual_multiplier))
            u_in = u_in_base + u_in_residual * 1e-3
        else:
            u_in = self.tensor(decay)

        u_in = torch.clamp(u_in, min=0.0, max=100.0).squeeze()

        return u_in

    def train(
            self,
            sim,
            pip_feed="parallel",
            duration=3,
            dt=0.03,
            epochs=100,
            use_noise=False,
            optimizer=torch.optim.Adam,
            optimizer_params={"lr": 1e-3, "weight_decay": 1e-4},
            loss_fn=torch.nn.L1Loss,
            loss_fn_params={},
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params={"factor": 0.9, "patience": 10},
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
            self.reset()
            sim.reset()

            if pip_feed == "parallel":
                self.zero_grad()
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            for PIP in random.sample(PIPs, 6):

                if pip_feed == "sequential":
                    self.zero_grad()
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

                self.waveform = BreathWaveform((PEEP, PIP))
                expiratory = Expiratory(waveform=self.waveform)

                for t in tt:
                    sim.pressure += use_noise * torch.normal(mean=torch.tensor(1.5), std=1.)
                    pressure = sim.pressure
                    u_in = self.compute_action(pressure, t)
                    u_out = expiratory.action(pressure, self.waveform.at(t), t)
                    sim.step(u_in, u_out) # potentially add multiplicative noise by * torch.normal(mean=torch.tensor(1.5), std=0.5)

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

    def train_global(
            self,
            sims,
            pip_feed="parallel",
            duration=3,
            dt=0.03,
            epochs=100,
            use_noise=False,
            optimizer=torch.optim.Adam,
            optimizer_params={"lr": 1e-3, "weight_decay": 1e-4},
            loss_fn=torch.nn.L1Loss,
            loss_fn_params={},
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params={"factor": 0.9, "patience": 10},
            use_tqdm=True,
            print_loss=1,
            shuffle=False,
            device="cpu", 
        PIPs = [10, 15, 20, 25, 30, 35],
    ):
        optimizer = optimizer(self.parameters(), **optimizer_params)
        scheduler = scheduler(optimizer, **scheduler_params)
        loss_fn = loss_fn(**loss_fn_params)

        tt = torch.linspace(0, duration, int(duration / dt))
        losses = []

        torch.autograd.set_detect_anomaly(True)

#         PIPs = [10, 15, 20, 25, 30, 35]
        PEEP = 5

        # TODO: handle device-awareness
        for epoch in range(epochs):

            if pip_feed == "parallel":
                self.zero_grad()
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            for PIP, sim in itertools.product(PIPs, sims):

                if pip_feed == "sequential":
                    self.zero_grad()
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

                self.waveform = BreathWaveform((PEEP, PIP))

                self.reset()
                sim.reset()

                for t in tt:
                    sim.pressure += use_noise * torch.normal(mean=torch.tensor(1.5), std=1.)
                    pressure = sim.pressure
                    u_in, u_out = self(pressure, t)
                    sim(u_in, u_out,
                        t)  # potentially add multiplicative noise by * torch.normal(mean=torch.tensor(1.5), std=0.5)

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

    def train_global_boosted(
            self,
            sim_datas, #Make this a list of (R, C, sim)s
            duration=3,
            dt=0.03,
            epochs=100,
            use_noise=False,
            optimizer=torch.optim.Adam,
            optimizer_params={"lr": 1e-3, "weight_decay": 1e-4},
            loss_fn=torch.nn.L1Loss,
            loss_fn_params={},
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params={"factor": 0.9, "patience": 1000},
            alpha=0.5,
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
        PIPSims = list(itertools.product(PIPs, sim_datas))
        loss_by_simpip = np.zeros(len(PIPSims))
        self.zero_grad()

        idxs =range(len(PIPSims))
        for epoch in range(epochs):
            p = softmax(alpha * loss_by_simpip)
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            for idx in idxs:
                PIP, (R, C, sim) = PIPSims[idx]
                self.waveform = BreathWaveform((PEEP, PIP))

                self.reset()
                sim.reset()
                loss_idx = 0.0
                for t in tt:
                    sim.pressure += use_noise * torch.normal(mean=torch.tensor(1.5), std=1.)
                    pressure = sim.pressure
                    u_in, u_out = self(pressure, t)
                    sim(u_in, u_out, t)

                    if u_out == 0:
                        curr_loss = loss_fn(torch.tensor(self.waveform.at(t)), pressure)
                        loss = loss + curr_loss * p[idx]
                        loss_idx += curr_loss

                loss.backward(retain_graph=True)
                per_step_loss = loss_idx / len(tt)
                losses.append(per_step_loss)
                loss_by_simpip[idx] = per_step_loss
            optimizer.step()
            self.zero_grad()

            print(f"Epoch: {epoch}, PIP: {PIP} R: {R} C: {C}\tLoss: {per_step_loss:.2f}\tLR: {optimizer.param_groups[0]['lr']}")
            for i in range(len(PIPSims)):
                PIP,(R,C, _) = PIPSims[i]
                print(f"PIP, {PIP}, R {R}, C {C}, loss {loss_by_simpip[i] :.2f}")
            print(f"loss mean {np.mean(loss_by_simpip):.2f}+-{np.std(loss_by_simpip):.2f}")
        return losses
