from deluca.lung.core import Controller
from deluca.lung.controllers._expiratory import Expiratory
from deluca.lung.environments._physical_lung import PhysicalLung
from deluca.lung.environments._balloon_lung import BalloonLung
from deluca.lung.utils.core import BreathWaveform
from deluca.lung.utils.scripts.save_data_and_plot import save_data_and_plot

import os
import click
import dill as pickle
import datetime
import time
import tqdm
import jax
import jax.numpy as jnp

def run_controller(
    controller,
    R,
    C,
    T=1000,
    dt=0.03,
    abort=60,
    env=None,
    waveform=None,
    use_tqdm=False,
    directory=None,
):
    env = env or BalloonLung()
    waveform = waveform or BreathWaveform.create()
    expiratory = Expiratory.create(waveform=waveform)

    result = locals()

    controller_state = controller.init()
    expiratory_state = expiratory.init()

    tt = range(T)
    if use_tqdm:
        tt = tqdm.tqdm(tt, leave=False)

    timestamps = jnp.zeros(T)
    pressures = jnp.zeros(T)
    flows = jnp.zeros(T)
    u_ins = jnp.zeros(T)
    u_outs = jnp.zeros(T)

    state, obs = env.reset()

    try:
        for i, _ in enumerate(tt):
            pressure = obs.predicted_pressure
            if env.should_abort():
                break

            controller_state, u_in = controller.__call__(controller_state, obs)
            expiratory_state, u_out = expiratory.__call__(expiratory_state, obs)
            state, obs = env(state, (u_in, u_out))

            timestamps = jax.ops.index_update(timestamps, i, env.time(state) - dt)
            u_ins = jax.ops.index_update(u_ins, i, u_in)
            u_outs = jax.ops.index_update(u_outs, i, u_out)
            pressures = jax.ops.index_update(pressures, i, pressure)
            flows = jax.ops.index_update(flows, i, env.flow)

            env.wait(max(dt - env.dt, 0))

    finally:
        env.cleanup()

    timeseries = {
        "timestamp": jnp.array(timestamps),
        "pressure": jnp.array(pressures),
        "flow": jnp.array(flows),
        "target": waveform.at(timestamps),
        "u_in": jnp.array(u_ins),
        "u_out": jnp.array(u_outs),
    }

    for key, val in timeseries.items():
        timeseries[key] = val[: T + 1]

    result["timeseries"] = timeseries

    if directory is not None:
        if not os.path.exists(directory):
            os.makedirs(directory)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        pickle.dump(result, open(f"{directory}/{timestamp}.pkl", "wb"))

    return result

def run_controller_scan(
    controller,
    R,
    C,
    T=1000,
    dt=0.03,
    abort=60,
    env=None,
    waveform=None,
    use_tqdm=False,
    directory=None,
):
    env = env or BalloonLung()
    waveform = waveform or BreathWaveform.create()
    expiratory = Expiratory.create(waveform=waveform)

    result = locals()

    controller_state = controller.init()
    expiratory_state = expiratory.init()

    tt = range(T)
    if use_tqdm:
        tt = tqdm.tqdm(tt, leave=False)

    timestamps = jnp.zeros(T)
    pressures = jnp.zeros(T)
    flows = jnp.zeros(T)
    u_ins = jnp.zeros(T)
    u_outs = jnp.zeros(T)

    state, obs = env.reset()
    xp = jnp.array(waveform.xp)
    fp = jnp.array(waveform.fp)
    period = waveform.period
    dtype = waveform.dtype

    try:
        def loop_over_tt(envState_obs_ctrlState_ExpState_i, dummy_data):
            state, obs, controller_state, expiratory_state, i = envState_obs_ctrlState_ExpState_i
            pressure = obs.predicted_pressure
            # if env.should_abort(): # TODO: how to handle break in scan
            #     break

            controller_state, u_in = controller.__call__(controller_state, obs)
            expiratory_state, u_out = expiratory.__call__(expiratory_state, obs)

            state, obs = env(state, (u_in, u_out))

            timestamps_i = env.time(state) - dt
            u_ins_i = u_in
            u_outs_i = u_out
            pressures_i = pressure
            flows_i = env.flow

            env.wait(max(dt - env.dt, 0))
            return (state, obs, controller_state, expiratory_state, i + 1), (timestamps_i, u_ins_i, u_outs_i, pressures_i, flows_i)

        _, (timestamps, u_ins, u_outs, pressures, flows) = jax.lax.scan(loop_over_tt, (state, obs, controller_state, expiratory_state, 0), jnp.arange(T))
        
    finally:
        env.cleanup()

    timeseries = {
        "timestamp": jnp.array(timestamps),
        "pressure": jnp.array(pressures),
        "flow": jnp.array(flows),
        "target": waveform.at(timestamps),
        "u_in": jnp.array(u_ins),
        "u_out": jnp.array(u_outs),
    }

    for key, val in timeseries.items():
        timeseries[key] = val[: T + 1]

    result["timeseries"] = timeseries

    if directory is not None:
        if not os.path.exists(directory):
            os.makedirs(directory)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        pickle.dump(result, open(f"{directory}/{timestamp}.pkl", "wb"))

    return result


def run_controller_command_line(path, r, c, peep, pip, bpm, t, dt, abort, name):
    directory = os.path.dirname(os.path.realpath(path))
    controller = Controller.load(path)
    waveform = BreathWaveform(range=(peep, pip), bpm=bpm)
    pickle.dump(locals(), open(os.path.join(directory, "meta.pkl"), "wb"))

    result = run_controller(
        controller,
        R=r,
        C=c,
        T=t,
        dt=dt,
        abort=abort,
        env=PhysicalLung(),
        waveform=waveform,
        directory=directory,
    )

    save_data_and_plot(result, directory, name)


@click.command()
@click.argument("path", type=click.Path(exists=True), required=1)
@click.option("-R", type=int, default=50, help="R value for phsyical lung")
@click.option("-C", type=int, default=10, help="C value for phsyical lung")
@click.option("--PEEP", type=int, default=5, help="PEEP")
@click.option("--PIP", type=int, default=35, help="PIP")
@click.option("--bpm", type=int, default=20, help="bpm")
@click.option("-T", type=int, default=1000, help="Default timesteps")
@click.option("--dt", type=float, default=0.03, help="Time to wait")
@click.option("--abort", type=float, default=60, help="Abort pressure")
@click.option("--name", type=str, default="run", help="Name of experiment")
def _run_controller_command_line(path, r, c, peep, pip, bpm, t, dt, abort, name):
    return run_controller_command_line(path, r, c, peep, pip, bpm, t, dt, abort, name)


# For running on the physical lung from command line with controller path
if __name__ == "__main__":
    _run_controller_command_line()
