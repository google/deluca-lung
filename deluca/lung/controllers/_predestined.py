import jax
import jax.numpy as jnp
import deluca
from functools import partial
from typing import List, Callable
from deluca.lung.core import Controller
from deluca.lung.core import DEFAULT_DT
from deluca.lung.core import proper_time


class Predestined(Controller):
    time: float = deluca.field(jaxed=False)
    steps: int = deluca.field(jaxed=False)
    dt: float = deluca.field(jaxed=False)
    u_ins: jnp.array = deluca.field(jaxed=False)
    switch_funcs: List[Callable] = deluca.field(jaxed=False)
    def setup(self):
        self.switch_funcs = [partial(lambda x, y: self.u_ins[y], y=i) for i in range(len(self.u_ins))]

    def __call__(self, state, obs, *args, **kwargs):
        action = jax.lax.switch(state.steps, self.switch_funcs, None)
        time = obs.time
        new_dt = jnp.max(jnp.array([DEFAULT_DT, time - proper_time(state.time)]))
        new_time = time
        new_steps = state.steps + 1
        state = state.replace(time=new_time, steps=new_steps, dt=new_dt)
        return state, action
