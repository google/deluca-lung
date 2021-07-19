import os
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt

from deluca.lung.controllers import ResidualExplorer, PID
from deluca.lung.environments import PhysicalLung
from deluca.lung.utils.data.analyzer import Analyzer
from deluca.lung.utils.core import BreathWaveform
from deluca.lung.utils.scripts.run_calibration import run_calibration
from deluca.lung.utils.scripts.run_controller import run_controller


plt.rc("figure", figsize=(10, 3))


def collect_runs(controller, path, n_runs=1, append_to=None, env=None, **kwargs):
    if append_to is None:
        results = []
    else:
        results = append_to

    for i in range(n_runs):
        env.reset()
        result = run_controller(controller, env=env, **kwargs)
        results.append(result)
        pickle.dump(result, open(path % i, "wb"))
    return results


def run_explorer(
    controller,
    directory,
    R=50,
    C=10,
    PEEP=5,
    T=10000,
    n_runs=5,
    PIPs=[10, 15, 20, 25, 30, 35],
    abort=70,
    dt=0.03,
    env=None,
):
    if not isinstance(controller, ResidualExplorer):
        print("ERROR: expecting ResidualExplorer as controller")
        return

    env = env or PhysicalLung()
    print("Running calibration")
    run_calibration(R, C, PEEP, directory)

    all_results = {}
    for PIP in PIPs:
        path = os.path.join(directory, f"R{R}C{C}PEEP{PEEP}_PIP{PIP}_pid_triangle_residual_%i.pkl")

        controller.waveform = BreathWaveform((PEEP, PIP))

        results = collect_runs(
            controller,
            path,
            R=R,
            C=C,
            PEEP=PEEP,
            T=T,
            dt=dt,
            n_runs=n_runs,
            abort=abort,
            env=env,
        )
        all_results[PIP] = results

    print("Running calibration")
    run_calibration(R, C, PEEP, directory)

    return all_results
