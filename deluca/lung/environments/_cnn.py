import torch

from deluca.lung.core import TorchLungEnv


class CNN(TorchLungEnv):
    def __init__(
        self,
        featurizer,
        out_channels=10,
        kernel=(1, 2),
        stride=1,
        pool_kernel=(1, 2),
        pool_stride=2,
    ):
        self.featurizer = featurizer
        self.in_dim = featurizer.u_window + featurizer.p_window

        out_dim_conv = int((self.in_dim - kernel[1]) / stride + 1)
        out_dim_pool = int((out_dim_conv - pool_kernel[1]) / pool_stride + 1)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=kernel, stride=stride)
        )
        self.fc1 = torch.nn.Linear(out_dim_conv * out_channels, 1)

    def reset(self):
        self.featurizer.reset()
        self.pressure = 0

    def train_step(self, x):
        if len(x.shape) == 1:
            x = torch.reshape(x, (1, x.shape[0]))
        x = torch.reshape(x, (x.shape[0], 1, 1, x.shape[1]))
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out

    def forward(self, x):
        return self.train_step(x)

    def step(self, u_in, u_out):
        features = self.featurizer(u_in, u_out, self.pressure)
        self.pressure = self.featurizer.p_scaler.inverse_transform(
            self.train_step(features)
        ).squeeze()
        return self.pressure

