import torch

from deluca.lung.core import TorchLungEnv


class MLP(TorchLungEnv):
    def __init__(self, featurizer, hid_1=10, hid_2=20, droprate=0.3):
        self.featurizer = featurizer
        self.in_dim = featurizer.u_window + featurizer.p_window
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.in_dim, hid_1),
            torch.nn.Dropout(droprate),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_1, hid_2),
            torch.nn.Dropout(droprate),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_2, 1),
        )

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
        return self.network(features)
