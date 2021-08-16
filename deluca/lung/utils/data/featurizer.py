# NOTE: API needs work (e.g., what should __call__'s signature be)

import torch

from deluca.lung.utils.core import TorchStandardScaler


class Featurizer:
    def reset(self):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


# Simulator featurizers


class ScalingHistoryFeaturizer(Featurizer):
    def __init__(self, u_window, p_window, u_scaler, p_scaler):
        self.u_window = u_window
        self.p_window = p_window
        self.u_scaler = TorchStandardScaler(u_scaler)
        self.p_scaler = TorchStandardScaler(p_scaler)

        self.reset()

    def reset(self):
        self.u_history = [self.u_scaler.transform(0).detach()] * self.u_window
        self.p_history = [self.p_scaler.transform(0).detach()] * self.p_window

    def __call__(self, u_in, u_out, pressure, *args, **kwargs):
        if not isinstance(u_in, torch.Tensor):
            u_in = torch.tensor(u_in)
        u_in = self.u_scaler.transform(u_in)
        self.u_history.append(u_in)
        self.u_history.pop(0)

        if not isinstance(pressure, torch.Tensor):
            pressure = torch.tensor(pressure)
        pressure = self.p_scaler.transform(pressure)
        self.p_history.append(pressure)
        self.p_history.pop(0)

        return torch.cat([torch.cat(self.u_history), torch.cat(self.p_history)])

# Controller featurizers


class TriangleErrorFeaturizer(Featurizer):
    def __init__(self, history_len):
        self.history_len = history_len

        self.coef = torch.ones((history_len, history_len), requires_grad=False).tril_()
        self.coef /= torch.arange(history_len, 0, -1).unsqueeze(0)

        self.reset()

    def reset(self):
        self.errs = [torch.tensor(0., dtype=torch.get_default_dtype())] * self.history_len

    def __call__(self, pressure, target, t, *args, **kwargs):
        self.errs.append(target - pressure)
        self.errs.pop(0)

        trajectory = torch.stack(self.errs).unsqueeze(0).unsqueeze(0)

        return trajectory @ self.coef
