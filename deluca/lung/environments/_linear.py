import torch

from deluca.lung.core import TorchLungEnv


class Linear(TorchLungEnv):
    def __init__(self, featurizer):
        self.featurizer = featurizer
        self.network = torch.nn.Linear(featurizer.u_window + featurizer.p_window, 1, bias=True)

        self.reset()

    def reset(self):
        self.featurizer.reset()
        self.pressure = 0

    def step(self, u_in, u_out):
        """
        For inference and returns unscaled pressure
        """
        features = self.featurizer(u_in, u_out, self.pressure)
        self.pressure = self.featurizer.p_scaler.inverse_transform(self.network(features)).squeeze()
        return self.pressure

    def forward(self, features):
        """
        For training and returns scaled pressure
        """
        return self.network(features)
