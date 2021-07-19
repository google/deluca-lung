import torch
import random
from deluca.lung.controllers import Expiratory
from deluca.lung.utils import BreathWaveform
import jax
import jax.numpy as jnp
import optax

def rollout(controller, sim, waveform, tt, use_noise, loss_fn, loss):
    expiratory = Expiratory.create(waveform=waveform)

    controller_state = controller.init()
    expiratory_state = expiratory.init()
    sim_state, obs = sim.reset() 
    def loop_over_tt(ctrlState_expState_simState_obs_loss, t):
        controller_state, expiratory_state, sim_state, obs, loss = ctrlState_expState_simState_obs_loss
        mean = 1.0
        std = 1.0
        noise = mean + std * jax.random.normal(jax.random.PRNGKey(0), shape=())
        pressure = sim_state.predicted_pressure + use_noise * noise
        sim_state = sim_state.replace(predicted_pressure=pressure) # Need to update p_history as well or no?
        obs = obs.replace(predicted_pressure=pressure, time=t)

        controller_state, u_in = controller(controller_state, obs)
        expiratory_state, u_out = expiratory(expiratory_state, obs)

        sim_state, obs = sim(sim_state, (u_in, u_out))
        
        loss = jax.lax.cond(u_out == 0,
                            lambda x: x + loss_fn(jnp.array(waveform.at(t)), pressure),
                            lambda x: x,
                            loss)
        return (controller_state, expiratory_state, sim_state, obs, loss), None
    (_, _, _, _, loss), _ = jax.lax.scan(loop_over_tt, (controller_state, expiratory_state, sim_state, obs, loss), tt)
    return loss

# TODO: add scheduler and scheduler_params
# Question: Jax analogue of torch.autograd.set_detect_anomaly(True)?
def train_controller_jax(
    controller,
    sim,
    waveform,
    duration=3,
    dt=0.03,
    epochs=100,
    use_noise=False,
    optimizer=optax.adamw,
    optimizer_params={"learning_rate": 1e-3, "weight_decay": 1e-4},
    loss_fn=lambda x, y: (jnp.abs(x - y)).mean(),
    # scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    # scheduler_params={},
    use_tqdm=True,
    print_loss=1,
    shuffle=False,
):
    optim = optimizer(**optimizer_params)
    optim_state = optim.init(controller)

    tt = jnp.linspace(0, duration, int(duration / dt))
    losses = []

    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        value, grad = jax.value_and_grad(rollout)(controller, sim, waveform, tt, use_noise, loss_fn, jnp.array(0.))
        updates, optim_state = optim.update(grad, optim_state, controller)
        controller = optax.apply_updates(controller, updates)
        per_step_loss = value / len(tt)
        losses.append(per_step_loss)

        if epoch % print_loss == 0:
            print(
                f"Epoch: {epoch}\tLoss: {per_step_loss:.2f}"
            )
    return controller
    

def train_controller(
    controller,
    sim,
    waveform,
    duration=3,
    dt=0.03,
    epochs=100,
    use_noise=False,
    optimizer=torch.optim.Adam,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-4},
    loss_fn=torch.nn.L1Loss,
    loss_fn_params={},
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_params={},
    use_tqdm=True,
    print_loss=1,
    shuffle=False,
    device="cpu",
):
    optimizer = optimizer(controller.parameters(), **optimizer_params)
    scheduler = scheduler(optimizer, **scheduler_params)
    loss_fn = loss_fn(**loss_fn_params)

    tt = torch.linspace(0, duration, int(duration / dt))
    losses = []

    torch.autograd.set_detect_anomaly(True)

    # TODO: handle device-awareness
    for epoch in range(epochs):
        controller.reset()
        sim.reset()

        loss = torch.tensor(0.0, device=device, requires_grad=True)

        controller.zero_grad()

        expiratory = Expiratory(waveform=waveform)

        for t in tt:
            sim.pressure += use_noise * torch.normal(mean=torch.tensor(1.0), std=1.0)
            pressure = sim.pressure
            u_in = controller.action(pressure, waveform.at(t), t)
            u_out = expiratory.action(pressure, waveform.at(t), t)
            sim.step(u_in, u_out)

            if u_out == 0:
                loss = loss + loss_fn(torch.tensor(waveform.at(t)), pressure)

        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step(loss)
        per_step_loss = loss / len(tt)
        losses.append(per_step_loss)
        if epoch % print_loss == 0:
            print(
                f"Epoch: {epoch}\tLoss: {per_step_loss:.2f}\tLR: {optimizer.param_groups[0]['lr']}"
            )

    return losses


def train_controller_multipip(
    controller,
    sim,
    pip_feed="sequential",
    duration=3,
    dt=0.03,
    epochs=100,
    use_noise=False,
    optimizer=torch.optim.Adam,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-4},
    loss_fn=torch.nn.L1Loss,
    loss_fn_params={},
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_params={"factor": 0.9, "patience": 10},
    use_tqdm=True,
    print_loss=1,
    shuffle=False,
    device="cpu",
):
    optimizer = optimizer(controller.parameters(), **optimizer_params)
    scheduler = scheduler(optimizer, **scheduler_params)
    loss_fn = loss_fn(**loss_fn_params)

    tt = torch.linspace(0, duration, int(duration / dt))
    losses = []

    torch.autograd.set_detect_anomaly(True)

    PIPs = [10, 15, 20, 25, 30, 35]
    PEEP = 5

    # TODO: handle device-awareness
    for epoch in range(epochs):
        controller.reset()
        sim.reset()

        if pip_feed == "parallel":
            controller.zero_grad()
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        for PIP in random.sample(PIPs, 6):

            if pip_feed == "sequential":
                controller.zero_grad()
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            waveform = BreathWaveform((PEEP, PIP))
            expiratory = Expiratory(waveform=waveform)

            for t in tt:
                sim.pressure += use_noise * torch.normal(
                    mean=torch.tensor(1.5), std=1.0
                )
                pressure = sim.pressure
                u_in = controller.action(pressure, waveform.at(t), t)
                u_out = expiratory.action(pressure, waveform.at(t), t)
                sim.step(u_in, u_out)

                if u_out == 0:
                    loss = loss + loss_fn(torch.tensor(waveform.at(t)), pressure)

            if pip_feed == "sequential":
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step(loss)
                per_step_loss = loss / len(tt)
                losses.append(per_step_loss)

                if epoch % print_loss == 0:
                    print(
                        f"Epoch: {epoch}, PIP: {PIP}\tLoss: {per_step_loss:.2f}\tLR: {optimizer.param_groups[0]['lr']}"
                    )

        if pip_feed == "parallel":
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step(loss)
            per_step_loss = loss / len(tt)
            losses.append(per_step_loss)

            if epoch % print_loss == 0:
                print(
                    f"Epoch: {epoch}\tLoss: {per_step_loss:.2f}\tLR: {optimizer.param_groups[0]['lr']}"
                )

    return losses
