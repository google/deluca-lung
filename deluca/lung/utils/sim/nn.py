import flax.linen as fnn
from deluca.lung.utils.data.munger import Munger
from deluca.lung.utils.data.alpha_dropout import Alpha_Dropout

# TODO: learning rate scheduler for Stitched_sim_trainer
class SNN(fnn.Module):
    out_dim: int = 1
    hidden_dim: int = 100
    n_layers: int = 4
    dropout_prob: float = 0.0
    scale: float = 1.0507009873554804934193349852946
    alpha: float = 1.6732632423543772848170429916717

    @fnn.compact
    def __call__(self, x):
        for i in range(self.n_layers - 1):
            x = fnn.Dense(features=self.hidden_dim, use_bias=False, name=f"SNN_fc{i}")(x)
            x = self.scale * fnn.elu(x, alpha=self.alpha)
            x = Alpha_Dropout(rate=self.dropout_prob, deterministic=False)(x)
        x = fnn.Dense(features=self.out_dim, use_bias=True, name=f"SNN_fc{i + 1}")(x)
        return x


class ShallowBoundaryModel(fnn.Module):
    out_dim: int = 1
    hidden_dim: int = 100
    model_num: int = 0

    @fnn.compact
    def __call__(self, x):
        x = fnn.Dense(
            features=self.hidden_dim,
            use_bias=False,
            name=f"shallow_fc{1}_model" + str(self.model_num),
        )(x)
        x = fnn.tanh(x)
        x = fnn.Dense(
            features=self.out_dim, use_bias=True, name=f"shallow_fc{2}_model" + str(self.model_num)
        )(x)
        return x


class ConstantModel(fnn.Module):
    const_out: float = 0.0

    @fnn.compact
    def __call__(self, x):
        return self.const_out

    def update_constant(self, const_out):
        self.const_out = const_out
