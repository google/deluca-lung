import numpy as np
import jax
import jax.numpy as jnp
import deluca
from deluca.lung.core import Controller
from deluca.lung.utils.core import BreathWaveform
from deluca.lung.utils.core import Phase
from deluca.lung.core import DEFAULT_DT
from deluca.lung.core import proper_time

class Expiratory(Controller):
    waveform: BreathWaveform = deluca.field(BreathWaveform, jaxed=False)
    def __call__(self, state, obs, *args, **kwargs):
        pressure, time = obs.predicted_pressure, obs.time
        phase = self.waveform.phase(time)
        u_out = jnp.zeros_like(phase)
        u_out = jax.lax.cond(jax.numpy.logical_or(jnp.equal(phase, Phase.RAMP_DOWN.value), 
                                                  jnp.equal(phase, Phase.PEEP.value)),
                             lambda x : 1,
                             lambda x: x,
                             u_out)

        new_dt = jnp.max(jnp.array([DEFAULT_DT, time - proper_time(state.time)]))
        new_time = time
        new_steps = state.steps + 1
        state = state.replace(time=new_time, steps=new_steps, dt=new_dt)
        return state, u_out
