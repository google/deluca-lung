import os

import deluca.core
import jax
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.realpath(os.path.join(__file__, "../..")))

DEFAULT_DT = 0.03


class LungEnv(deluca.Env):
    # @property
    def time(self, state):
        return self.dt * state.steps

    @property
    def dt(self):
        return DEFAULT_DT

    @property
    def physical(self):
        return False

    def should_abort(self):
        return False

    def wait(self, duration):
        pass

    def step(self, u_in, u_out):
        pass

    def cleanup(self):
        pass

class ControllerState(deluca.Obj):
    time: float = float("inf")
    steps: int = 0
    dt: float = DEFAULT_DT

def proper_time(t):
    return jax.lax.cond(t == float('inf'), lambda x : 0., lambda x: x, t)

class Controller(deluca.Agent):
    def __new__(cls, *args, **kwargs):
        obj = deluca.Agent.__new__(cls)

        super(cls, obj).init()

        return obj

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if name == "init":
            super(self.__class__, self).init()
        return attr

    def init(self):
        return ControllerState()

    @property
    def max(self):
        return 100

    @property
    def min(self):
        return 0

    def action(self, pressure, target, time):
        return 0
