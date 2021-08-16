import tqdm
import jax.numpy as jnp
from deluca.lung.utils.data.analyzer import Analyzer


class Dataset:
    def __init__(self, paths, inspiratory_only=True, clip=(1, -1), breath_length=29):
        self.data = []

        if isinstance(paths, str):
            paths = [paths]

        for path in tqdm.tqdm(paths):
            analyzer = Analyzer(path)
            if inspiratory_only:
                clips = analyzer.infer_inspiratory_phases()
            else:
                clips = analyzer.infer_breaths()

            for start, end in clips[clip[0] : clip[1]]:
                if end - start < breath_length:
                    continue

                u_in = analyzer.u_in[start : start + breath_length]
                pressure = analyzer.pressure[start : start + breath_length]

                self.data.append((jnp.array(u_in), jnp.array(pressure)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __add__(self, other):
        self.data.extend(other.data)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            item = self[self.n]
            self.n += 1
            return item
        else:
            raise StopIteration
