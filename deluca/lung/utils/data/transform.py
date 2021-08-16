import jax.numpy as jnp


class ShiftScaleTransform:
    # vectors is an array of vectors
    def __init__(self, vectors):
        vectors_concat = jnp.concatenate(vectors)
        self.mean = jnp.mean(vectors_concat)
        self.std = jnp.std(vectors_concat)
        print(self.mean, self.std)

    def _transform(self, x, mean, std):
        return (x - mean) / std

    def _inverse_transform(self, x, mean, std):
        return (x * std) + mean

    def __call__(self, vector):
        return self._transform(vector, self.mean, self.std)

    def inverse(self, vector):
        return self._inverse_transform(vector, self.mean, self.std)
