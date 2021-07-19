import torch

from deluca.lung import Controller


class IBonusCubic(Controller):
    '''
    Trainable I controller with cubic bonus function of time for jaxed parameters A,B
    '''

    def __init__(self, K_I=2.0, K_P = 1.0, A=-1.0, B= 0.0, C=0.0, D = 0.0,  RC=0.5, **kwargs):
        self.init_K_I = self.tensor(K_I)
        self.init_K_P = self.tensor(K_P)
        self.init_A = self.tensor(A)
        self.init_B = self.tensor(B)
        self.init_C = self.tensor(C)
        self.init_D = self.tensor(D)

        self.K_I = torch.nn.Parameter(self.init_K_I)
        self.K_P = torch.nn.Parameter(self.init_K_P)
        self.A = torch.nn.Parameter(self.init_A)
        self.B = torch.nn.Parameter(self.init_B)
        self.C = torch.nn.Parameter(self.init_C)
        self.D = torch.nn.Parameter(self.init_D)
        self.RC = RC

        self.reset()

    def reset(self):
        self.I = self.tensor(0.0)


    def compute_action(self, state, t):
        err = torch.tensor(self.waveform.at(t)) - state
        dt = self.dt(t)
        decay = dt / (dt + self.RC)

        t_cycle = self.cycle_phase(t) + 1

        self.I = self.I + self.tensor(decay * (err - self.I))


        poly = self.A * t_cycle **3 + self.B *t_cycle**2 + self.C * t_cycle + self.D

        u_in = torch.clamp(self.K_I * self.I + self.K_P * err + poly, min=0.0, max=100.0)

        return (u_in, self.u_out(t))

# generic PID controller
class LookaheadPID(Controller):
    def __init__(self, K=[3, 4, 0], RC=0.5, pressure_history = 4, lookahead = 1,**kwargs):
        self.init_K = self.tensor(K)

        self.K = torch.nn.Parameter(self.init_K)
        self.RC = RC
        self.pressure_history, self.lookahead = pressure_history, lookahead

        self.reset()

    def reset(self):
        self.coef = self.tensor([0.0, 0.0])
        self.last_states = [torch.tensor(0.) for _ in range(self.pressure_history)]

    @property
    def P(self):
        return self.coef[0]

    @property
    def I(self):
        return self.coef[1]

    def compute_action(self, state, t):
        self.last_states.pop(0)
        self.last_states.append(state)
        delta_state = sum(list(map(lambda x: x[0]-x[1], zip(self.last_states[1:], self.last_states[:-1]))))/(self.pressure_history-1)
        err = torch.tensor(self.waveform.at(t)) - state
        dt = self.dt(t)
        decay = dt / (dt + self.RC)

        self.coef = self.tensor(
            [
                err,
                (1 - decay) * self.I + decay * err,
                sum([
                    self.waveform.at(t + dt * i) - (state + delta_state * i) for i in range(self.lookahead)
                ]) / self.lookahead
            ]
        )

        u_in = torch.clamp(torch.dot(self.K, self.coef), min=0.0, max=100.0)

        return (u_in, self.u_out(t))


class IBonus(Controller):
    '''
    Trainable I controller with bonus of the form A t^{-B} for jaxed parameters A,B
    '''

    def __init__(self, K_I=2.0, A=1.0, B= 1.0, RC=0.5, **kwargs):
        self.init_K_I = self.tensor(K_I)
        self.init_A = self.tensor(A)
        self.init_B = self.tensor(B)

        self.K_I = torch.nn.Parameter(self.init_K_I)
        self.A = torch.nn.Parameter(self.init_A)
        self.B = torch.nn.Parameter(self.init_B)
        self.RC = RC

        self.reset()

    def reset(self):
        self.I = self.tensor(0.0)


    def compute_action(self, state, t):
        err = torch.tensor(self.waveform.at(t)) - state
        dt = self.dt(t)
        decay = dt / (dt + self.RC)

        t_cycle = self.cycle_phase(t) + 1

        self.I = self.I + self.tensor(decay * (err - self.I))

        u_in = torch.clamp(self.K_I * self.I + self.A * torch.pow(t_cycle, -self.B), min=0.0, max=100.0)

        return (u_in, self.u_out(t))
