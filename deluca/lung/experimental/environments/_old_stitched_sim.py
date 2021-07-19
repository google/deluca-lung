import numpy as np
import torch
from copy import deepcopy

from deluca.lung.environments.core import Environment


class OldStitchedSim(Environment):
    def __init__(self, inspiratory_model, u_window, p_window, u_scaler, p_scaler):
        super().__init__()

        self.inspiratory_model = inspiratory_model

        self.u_window = u_window
        self.p_window = p_window

        self.u_scaler = u_scaler
        self.p_scaler = p_scaler

        # reset states
        self.reset()

    def reset(self):
        self.scaled_peep = self.inspiratory_model.boundary_dict["0"]().item()
        self.state = {'t_in':0, 'u_history':[], 'p_history':[self.scaled_peep]}
        self.sync_unscaled_pressure()

    def sync_unscaled_pressure(self):
        self.pressure = self.p_scaler.inverse_transform([[self.state['p_history'][-1]]])[0, 0]

    def cache_state(self, return_state=False):
        # set up a snapshot for rewind_state
        state_copy = deepcopy(self.state)
        if return_state:
            return state_copy
        else:
            self.cached_state = state_copy

    def rewind_state(self, state=None):
        # load a history state cached by cache_state
        if state:
            self.state = deepcopy(state)
        else:
            self.state = deepcopy(self.cached_state)
        self.sync_unscaled_pressure()

    def step(self, u_in, u_out, t):

        u_in_scaled = self.u_scaler.transform([[u_in]])[0, 0]
        self.state['u_history'].append(u_in_scaled)

        if u_out == 1:
            if self.state['t_in'] > 0: # reset once per u_out=1
                self.state = {'t_in':0, 'u_history':[], 'p_history':[self.scaled_peep]}
                self.sync_unscaled_pressure()
        else:
            self.state['t_in'] += 1
            t_key = str(self.state['t_in'])

            if t_key in self.inspiratory_model.boundary_dict:  # predict from boundary model
                features = np.concatenate([self.state['u_history'], self.state['p_history']])
                features = torch.tensor(features, dtype=torch.float)
                scaled_pressure = self.inspiratory_model.boundary_dict[t_key](features).item()
            else:  # predict from default model
                features = np.concatenate(
                    [self.state['u_history'][-self.u_window:], self.state['p_history'][-self.p_window:]]
                )
                features = torch.tensor(features, dtype=torch.float)
                scaled_pressure = self.inspiratory_model.default_model(features).item()

            self.state['p_history'].append(scaled_pressure)
            self.sync_unscaled_pressure()
