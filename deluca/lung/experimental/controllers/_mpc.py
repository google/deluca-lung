import numpy as np
import torch
from deluca.lung.controllers.core import Controller
from deluca.lung.controllers import PID
import time

class MPC(Controller):
    def __init__(self, sim, MPC_lookahead=10, output_length=1, optimization_params=None, verbose=False,
                 initializer='ZEROS', init_params=None, **kwargs):
        super().__init__()

        self.sim = sim
        self.MPC_lookahead = MPC_lookahead
        self.output_length = output_length
        self.optimization_params = optimization_params
        self.verbose = verbose
        self.initializer = initializer
        self.dthard = 0.03
        self.init_params = init_params
        self.reset()

    def reset(self):
        self.plan = []
        self.immediate_counter = 0
        self.counter = 0

    def compute_action(self, state, t):
        #print(t)
        a = time.time()
        decay = self.waveform.decay(t)
        self.sim.set_current_pressure(state)
        # returns a copy of the state as it will be needed to reset sim
        if decay is None:
            if self.counter >= self.immediate_counter:
                #a = time.time()
                self.update_plan(t)
                #print(time.time()-a)
                self.immediate_counter = self.counter+self.output_length
            u_in = self.plan[self.counter]
            # is this necessary?
            u_in = torch.clamp(u_in, min=0., max=100.)
        else:
            self.immediate_counter = self.counter + 1
            u_in = decay
            if self.counter < len(self.plan):
                self.plan[self.counter] = torch.tensor(u_in)
            else:
                self.plan.append(torch.tensor(u_in))
        u_out = self.u_out(t)
        # Update the sim according to new syntax
        self.sim(u_in, u_out, t)
        self.counter += 1
        # print(time.time()-a)
        return u_in, u_out

    def initialization(self, sub_plan, extra_plan_length, trange):
        #print("In initialization")
        #print(sub_plan)
        #print(extra_plan_length)
        #print(trange)
        if self.initializer == 'ZEROS':
            return [torch.tensor(0.0, requires_grad=True) for _ in range(extra_plan_length)]
        elif self.initializer == 'PID':
            PIDvals = self.init_params['PID']
            PIDcont = PID(K=PIDvals, waveform=self.waveform)
            self.sim.cache_state()
            #print(self.sim.state['t_in'])
            #print(self.sim.state['u_history'])
            #print(self.sim.state['p_history'])
            # rollout the subplan
            for i in range(len(sub_plan)):
                self.sim(sub_plan[i], self.u_out(trange[i]), trange[i])
            # now use the controller
            extra_plan = []
            for i in range(len(sub_plan), len(trange)):
                #print(self.sim.pressure.item(), trange[i])
                u_in, u_out = PIDcont(self.sim.pressure.item(), trange[i])
                #print(u_in, u_out)
                extra_plan.append(torch.tensor(u_in.item(), requires_grad=True))
                self.sim(u_in, self.u_out(trange[i]), trange[i])
            #print("aaa", self.sim.state['t_in'])
            #print("aaa", self.sim.state['u_history'])
            #print("aaa", self.sim.state['p_history'])
            self.sim.rewind_state()
            #print(extra_plan)
            return extra_plan
        else:
            print("Initializer not implemented")
            exit()

    def update_plan(self, start_t):
        if self.verbose:
            print("Updating plan")
        # This code finds a good lookahead open loop policy
        extra_plan_length = self.counter + self.MPC_lookahead - len(self.plan)
        sub_plan = self.plan[self.counter:]
        trange = [start_t + i*self.dthard for i in range(self.MPC_lookahead)]
        extra_plan = self.initialization(sub_plan, extra_plan_length, trange)
        self.plan.extend(extra_plan)
        if self.optimization_params:
            epochs = self.optimization_params['epochs']
            schedule = self.optimization_params['schedule']
            optimizer = self.optimization_params['optimizer']
        else:
            epochs = 30
            schedule = {
                0: .5,
            }
            optimizer = 'Adam'
        losses = []
        end_t = start_t + self.MPC_lookahead*self.dthard
        tt = np.linspace(start_t, end_t, self.MPC_lookahead)
        local_plan = self.plan[self.counter:]
        self.sim.cache_state()
        for epoch in range(epochs):
            if epoch in schedule:
                if optimizer == 'SGD':
                    optim = torch.optim.SGD(local_plan, lr=schedule[epoch])
                elif optimizer == 'Adam':
                    optim = torch.optim.Adam(local_plan, lr=schedule[epoch])
                else:
                    print('error')
            loss = torch.tensor(0.0, requires_grad=True)
            counter = 0
            for j, t in enumerate(tt):
                if self.waveform.decay(t) is None:
                    pressure = self.sim.pressure
                    #u_in, u_out = torch.clamp(local_plan[j], 0., 100.), self.u_out(t)
                    u_in, u_out = local_plan[j], self.u_out(t)
                    # new syntax
                    self.sim(u_in, u_out, t)
                    decay = self.waveform.decay(t)
                    if decay is None:
                        loss = loss + torch.square(pressure - self.waveform.at(t))
                        counter += 1
            # loss.backward(retain_graph=True)
            loss.backward()
            optim.step()
            optim.zero_grad()
            #print(epoch, loss)
            normalized_loss = (loss / counter).sqrt()
            losses.append(normalized_loss / len(tt))
            self.sim.rewind_state()
        if self.verbose:
            print(f"Normalized loss at time step {start_t:0.2f} - { normalized_loss:0.4f}")
