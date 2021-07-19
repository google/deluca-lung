from deluca.lung.utils.data.munger import Munger
import math
import torch
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import pickle
import flax
import flax.linen as fnn
import jax
import jax.numpy as jnp
from jax import jit, partial
import functools
import optax
import deluca
from deluca.lung.utils.data.alpha_dropout import Alpha_Dropout

class SNN_jax(fnn.Module):
    u_history: int # = 5 # max(u_window, num_boundary_models)
    u_window: int # = 5 # number of features for default model
    p_window: int # = 3 # number of features for default model
    out_dim: int # = 1
    hidden_dim: int # = 100
    n_layers: int # = 4
    dropout_prob: float # = 0.0
    scale: float = 1.0507009873554804934193349852946
    alpha: float= 1.6732632423543772848170429916717

    @fnn.compact
    def __call__(self, x):
        for i in range(self.n_layers - 1):
            x = jnp.hstack([x[self.u_history - self.u_window: self.u_history], x[-self.p_window:]])
            x = fnn.Dense(features=self.hidden_dim, use_bias=False, name=f"fc{i}")(x)
            x = self.scale*fnn.elu(x, alpha=self.alpha)
            x = Alpha_Dropout(rate=self.dropout_prob, deterministic=False)(x)
        x = fnn.Dense(features=self.out_dim, use_bias=True, name=f"fc{i + 1}")(x)
        return x

class ShallowBoundaryModel_jax(fnn.Module):
    u_history: int # = 5
    input_dim: int # = 2
    out_dim: int # = 1
    hidden_dim: int # = 100
    model_num: int
        
    @fnn.compact
    def __call__(self, x):
        x = jnp.hstack([x[self.u_history - self.input_dim: self.u_history], x[-self.input_dim:]])
        x = fnn.Dense(features=self.hidden_dim, use_bias=False, name="shallow_fc1_model"+str(self.model_num))(x)
        x = fnn.tanh(x)
        x = fnn.Dense(features=self.out_dim, use_bias=True, name="shallow_fc2_model"+str(self.model_num))(x)
        return x

class InspiratoryModel_jax(fnn.Module):
    # boundary_mods: dict = deluca.field(default_factory=dict, jaxed=False)
    # default_mod: fnn.module = deluca.field(SNN_jax, jaxed=False)
    u_history: int = 5 # max(u_window, num_boundary_models)
    p_history: int = 5 # max(p_window, num_boundary_models)
    u_window: int = 5 # number of features for default model
    p_window: int = 3 # number of features for default model
    
    default_out_dim: int = 1
    default_hidden_dim: int = 100
    default_n_layers: int = 4
    default_dropout_prob: float = 0.0

    num_boundary_models: int = 5
    boundary_out_dim: int = 1
    boundary_hidden_dim: int = 100
    seed: int = 0

    def setup(self):
        
        self.default_mod = SNN_jax(u_history=self.u_history,
                                   u_window=self.u_window,
                                   p_window=self.p_window,
                                   out_dim=self.default_out_dim,
                                   hidden_dim=self.default_hidden_dim,
                                   n_layers=self.default_n_layers,
                                   dropout_prob=self.default_dropout_prob)

        self.boundary_mod = ShallowBoundaryModel_jax(u_history=self.u_history,
                                                     input_dim=self.u_history,
                                                     out_dim=self.boundary_out_dim,
                                                     hidden_dim=self.boundary_hidden_dim,
                                                     model_num=0)
        
        self.boundary_mods = {str(i): ShallowBoundaryModel_jax(u_history=self.u_history,
                                                                 input_dim=self.u_history,
                                                                 out_dim=self.boundary_out_dim,
                                                                 hidden_dim=self.boundary_hidden_dim,
                                                                 model_num=i)
                              for i in range(self.num_boundary_models)}
        '''
        self.all_models_dict = self.boundary_mods.copy({str(self.num_boundary_models+1): 
            SNN_jax(u_history=self.u_history,
                    u_window=self.u_window,
                    p_window=self.p_window,
                    out_dim=self.default_out_dim,
                    hidden_dim=self.default_hidden_dim,
                    n_layers=self.default_n_layers,
                    dropout_prob=self.default_dropout_prob)})
        self.all_models = [self.all_models_dict[str(i+1)]
                           for i in range(self.num_boundary_models+1)]
        print(type(self.all_models))
        print(self.all_models)'''

    def call_boundary_model(self, t, features):
        return self.boundary_mods[str(t)](features)

    def call_default_model(self, features):
        return self.default_mod(features)

    # couldn't get this methods to work
    def all_models_call(self, t_features):
        t, features = t_features
        # return jax.lax.switch(t, self.all_models, features)
        return self.all_models[t](features)
        # model = next(flax.traverse_util.TraverseItem(t).iterate(self.all_models))
        # return model(features)

    # couldn't get this methods to work
    def boundary_model(self, features):
        model = partial(ShallowBoundaryModel_jax, u_history=self.u_history,
                                                  input_dim=self.u_history, # wasted parameters
                                                  out_dim=self.boundary_out_dim,
                                                  hidden_dim=self.boundary_hidden_dim)
        return model(features)
        # x = jnp.hstack([x[self.u_history - t: self.u_history], x[-t:]])
        # x = jnp.hstack([jax.lax.dynamic_slice(x, (self.u_history-t,), (t,)),
        #                 jax.lax.dynamic_slice(x, (-t), (t,))])
        '''x = fnn.Dense(features=self.boundary_hidden_dim, use_bias=False, name="shallow_fc1_indim"+str(2*t))(x)
        x = fnn.tanh(x)
        x = fnn.Dense(features=self.boundary_out_dim, use_bias=True, name="shallow_fc2_indim"+str(2*t))(x)
        return x'''

    # couldn't get this methods to work
    def default_model(self, features):
        
        model = partial(SNN_jax, u_history=self.u_history,
                                 u_window=self.u_window,
                                 p_window=self.p_window,
                                 out_dim=self.default_out_dim,
                                 hidden_dim=self.default_hidden_dim,
                                 n_layers=self.default_n_layers,
                                 dropout_prob=self.default_dropout_prob)
        return model(features)
        '''
        for i in range(self.default_n_layers - 1):
            x = jnp.hstack([x[self.u_history - self.u_window: self.u_history], x[-self.p_window:]])
            x = fnn.Dense(features=self.default_hidden_dim, use_bias=False, name=f"fc{i}")(x)
            x = self.scale*fnn.elu(x, alpha=self.alpha)
            x = Alpha_Dropout(rate=self.default_dropout_prob, deterministic=False)(x)
        x = fnn.Dense(features=self.default_out_dim, use_bias=True, name=f"fc{i + 1}")(x)
        return x'''
        
    @fnn.compact
    def __call__(self, t, features):
        if t <= self.num_boundary_models:
            return self.boundary_mod(features)
        else:
            return self.default_mod(features)
        '''self.default = functools.partial(SNN_jax, 
                                    self.u_history,
                                    self.u_window,
                                    self.p_window,
                                    self.default_out_dim,
                                    self.default_hidden_dim,
                                    self.default_n_layers,
                                    self.default_dropout_prob)

        self.boundary = functools.partial(ShallowBoundaryModel_jax, 
                                     self.u_history,
                                     self.u_history, # wasted parameters
                                     self.boundary_out_dim,
                                     self.boundary_hidden_dim)'''
        

        ''' output = jax.lax.cond(t <= self.num_boundary_models,
                              boundary,
                              default,
                              features)
            return output'''

        
        # t, features = t_features
        # model_idx = jnp.min(jnp.array([self.num_boundary_models, t]))
        # print('model_idx:' + str(model_idx))
        # return self.all_models[str(model_idx)](features)
        # return jax.lax.switch(model_idx, self.all_models, features)
        # model = jax.lax.dynamic_slice(np.array((self.all_models)), (t,), (1,))
        # return model(features)
        # return self.all_models[t](features)
        '''
        def true_func(x):
            return self.boundary_models[str(t)](x)
        def false_func(x):
            return self.default_model(x)
        
        output = jax.lax.cond(t <= self.num_boundary_models,
                              true_func,
                              false_func,
                              features)
        return output'''
        '''
        if t <= len(self.boundary_models):
            # return jax.lax.switch(t, self.boundary_models, features)
            return self.boundary_models[str(t)](features)
        else:
            return self.default_model(features)'''
        