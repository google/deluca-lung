import numpy as np
import tqdm
import datetime

from deluca.lung.controllers import PID
from deluca.lung.environments import PhysicalLung
from deluca.lung.utils import Analyzer, BreathWaveform

from deluca.lung.utils import run_calibration
from deluca.lung.utils import run_controller

DEFAULT_VALUES = np.concatenate((np.linspace(0, 0.9, 10), np.linspace(1, 10, 10)))


def run_pid_grid(
    Ps=DEFAULT_VALUES,
    Is=DEFAULT_VALUES,
    Ds=[0.0],
    PIPs=[10, 15, 20, 25, 30, 35],
    R=50,
    C=10,
    PEEP=5,
    abort=60,
    T=300,
    directory=None,
    env=None,
    **kwargs,
):
    analyzers = []
    env = env or PhysicalLung()

    print("Running calibration")
    run_calibration(R, C, PEEP, directory, env=env)

    print("Running grid")
    for PIP in tqdm.tqdm(PIPs):
        for P in tqdm.tqdm(Ps, position=0, leave=False):
            for I in Is:
                for D in Ds:
                    env.reset()

                    waveform = BreathWaveform((PEEP, PIP))
                    pid = PID(K=[P, I, D], waveform=waveform)
                    result = run_controller(
                        pid, R, C, T, abort, env=env, waveform=waveform, directory=directory
                    )
                    analyzers.append(Analyzer(result))

    print("Running calibration")
    run_calibration(R, C, PEEP, directory, env=env)

    return analyzers
