import torch

from deluca.lung.core import TorchLungEnv


class LSTM(TorchLungEnv):
    def __init__(self, featurizer, device, hidden_size=10, num_layers=1):
        self.featurizer = featurizer
        self.in_dim = featurizer.u_window + featurizer.p_window

        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = torch.nn.LSTM(self.in_dim, hidden_size, num_layers)
        self.fc = torch.nn.Linear(hidden_size, 1)

        self.reset()

    def reset(self):
        self.featurizer.reset()
        self.pressure = 0

    def forward(self, x):
        return self.train_step(x)

    def step(self, u_in, u_out):
        """
        For inference and returns unscaled pressure
        """
        features = self.featurizer(u_in, u_out, self.pressure)
        self.pressure = self.featurizer.p_scaler.inverse_transform(
            self.train_step(features)
        ).squeeze()
        return self.pressure

    def train_step(self, features):
        """
        For training and expects/returns scaled features
        """
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        out, _ = self.lstm(features.unsqueeze(0))
        out = self.fc(out.squeeze(0))
        return out
