import jax
import jax.numpy as jnp
import deluca.core
from deluca.lung.core import Controller
from deluca.lung.core import DEFAULT_DT
from deluca.lung.core import proper_time
from deluca.lung.utils import BreathWaveform

class BangBang(Controller):
    waveform: deluca.Obj = deluca.field(BreathWaveform.create(), jaxed=False)
    min_action: float = deluca.field(0.0, jaxed=False)
    max_action: float = deluca.field(100.0, jaxed=False)
    def __call__(self, controller_state, obs):
        pressure, t = obs.predicted_pressure, obs.time
        target = self.waveform.at(t)
        action = jax.lax.cond(pressure < target,
                              lambda x : self.max_action,
                              lambda x : self.min_action,
                              None)
        # update controller_state
        new_dt = jnp.max(jnp.array([DEFAULT_DT, t - proper_time(controller_state.time)]))
        new_time = t
        new_steps = controller_state.steps + 1
        controller_state = controller_state.replace(time=new_time, steps=new_steps, dt=new_dt)
        return controller_state, action
        
