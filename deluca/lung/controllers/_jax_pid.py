import jax

from deluca.lung.core import Controller


class PIDJax(Controller):
    def __init__(self, K=[3., 4., 0.], RC=0.5, jaxed=False, **kwargs):
        self.init_K = K
        self.RC = RC
        self.reset()

    def reset(self):
        self.coef = [0., 0., 0.]

    def action(self, pressure, target, t):
        err = target - pressure
        decay = self.dt / (self.dt + self.RC)

        # self.coef += 
