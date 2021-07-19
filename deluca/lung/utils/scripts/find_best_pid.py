from deluca.lung.utils import Analyzer

import os
import tqdm
import torch
import numpy as np
import itertools

from deluca.lung.core import ROOT

DEFAULT_DATA_DIR = f"{ROOT}/data"


def where(arr, idxs):
    return [arr[idx] for idx in idxs]


def find_best_pid(
    Rs=[50],
    Cs=[10],
    PEEPs=[5],
    directory=DEFAULT_DATA_DIR,
    loss_fn=np.abs,
    k=10,
    p_mask=lambda x: True,
    i_mask=lambda x: True,
    d_mask=lambda x: x == 0,
):
    losses = {}
    for R in Rs:
        for C in Cs:
            for PEEP in PEEPs:
                pid_dir = os.path.join(directory, f"R{R}_C{C}_PEEP{PEEP}", "pid")
                for path in tqdm.tqdm(os.listdir(pid_dir)):
                    if path[-4:] == ".pkl":
                        try:
                            path = os.path.join(pid_dir, path)
                            analyzer = Analyzer(path)

                            K = analyzer.controller.K
                            if isinstance(K, torch.Tensor):
                                K = K.detach().numpy()
                            K = tuple(K.tolist())
                            if not (p_mask(K[0]) and i_mask(K[1]) and d_mask(K[2])):
                                continue
                            if K not in losses:
                                losses[K] = []
                            loss = analyzer.default_metric(loss_fn=loss_fn)
                            losses[K].append(loss)
                        except Exception as e:
                            print(path, e)

    keys = list(losses.keys())
    losses = [np.mean(losses[key]) for key in keys]
    k = min(k, len(keys))
    idxs = np.argsort(losses)

    return where(keys, idxs[:k]), where(losses, idxs[:k])


def find_global_best_pid(directory=DEFAULT_DATA_DIR, loss_fn=np.abs, k=10):
    pkls = []
    losses = {}
    for (R, C) in itertools.product([5, 20, 50], [10, 20, 50]):
        pid_dir = os.path.join(directory, f"R{R}_C{C}_PEEP{5}", "pid")
        for path in tqdm.tqdm(os.listdir(pid_dir)):
            if path[-4:] == ".pkl":

                try:

                    path = os.path.join(pid_dir, path)
                    analyzer = Analyzer(path)

                    pkls.append(path)
                    K = analyzer.controller.K
                    if isinstance(K, torch.Tensor):
                        K = K.detach().numpy()
                    K = tuple(K.tolist())
                    if K not in losses:
                        losses[K] = []
                    loss = analyzer.default_metric(loss_fn=loss_fn)
                    losses[K].append(loss)
                except Exception as e:
                    print(path, e)
    keys = list(losses.keys())
    losses = [np.mean(losses[key]) for key in keys]
    k = min(k, len(keys))
    idxs = np.argsort(losses)

    return where(keys, idxs[:k]), where(losses, idxs[:k])


def plot_pid(R=50, C=10, PEEP=5, K=(2.0, 10.0, 0.0), directory=DEFAULT_DATA_DIR):
    pid_dir = os.path.join(directory, f"R{R}_C{C}_PEEP{PEEP}", "pid")
    K = np.array(K)

    for path in os.listdir(pid_dir):
        if path[-4:] == ".pkl":
            try:
                path = os.path.join(pid_dir, path)
                analyzer = Analyzer(path)

                CK = analyzer.controller.K
                if isinstance(CK, torch.Tensor):
                    CK = CK.detach().numpy()

                if np.all(np.array(K) == CK):
                    analyzer.plot(
                        title=f"Range: ({analyzer.controller.waveform.PEEP}, {analyzer.controller.waveform.PIP}), {path}"
                    )
            except Exception as e:
                print(path, e)
