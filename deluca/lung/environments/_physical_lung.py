import time
import torch
import numpy as np
import datetime

from deluca.lung.core import LungEnv


class PhysicalLung(LungEnv):
    def __init__(
        self,
        hal=None,
        sleep=3.0,
        abort=50,
        PEEP=5,
        dt_threshold=0.04,
        dt_patience=10,
        peep_threshold=0.5,
        peep_patience=10,
        peep_breaths=2,
    ):
        if hal is None:
            from deluca.lung.devices.hal import Hal

            hal = Hal()
        self.hal = hal
        self.sleep = sleep

        self.abort = abort
        self.PEEP = PEEP
        self.dt_threshold = dt_threshold
        self.dt_patience = dt_patience
        self.peep_threshold = peep_threshold
        self.peep_patience = peep_patience

        self.breaths = 0
        self.__time = 0
        self.prev_time = 0
        self.peep_breaths = peep_breaths

        self.reset()

    def reset(self):
        super().reset()
        self.dt_window = np.ones(self.dt_patience) * self.dt
        self.pressure_window = np.ones(self.peep_patience) * self.PEEP
        self.breaths = 0
        self.__time = 0
        self.prev_time = 0
        self.start = datetime.datetime.now().timestamp()

    @property
    def pressure(self):
        return self.hal.pressure

    @pressure.setter
    def pressure(self, p):
        pass

    @property
    def flow(self):
        return self.hal.flow_ex

    @flow.setter
    def flow(self, f):
        pass

    @property
    def time(self):
        return self.__time

    @property
    def dt(self):
        return self.__time - self.prev_time

    @property
    def physical(self):
        return True

    def should_abort(self):
        timestamp = self.__time
        if self.pressure > self.abort:
            print(f"Pressure of {self.pressure} > {self.abort}; quitting")
            return True

        self.dt_window = np.roll(self.dt_window, 1)
        self.dt_window[0] = self.dt

        if np.mean(self.dt_window) > self.dt_threshold:
            # print(
            # f"dt averaged {100 * self.dt_threshold:.1f}% higher over the last {self.dt_patience} timesteps; quitting"
            # )
            return False

        if self.breaths > self.peep_breaths:
            self.pressure_window = np.roll(self.pressure_window, 1)
            self.pressure_window[0] = self.pressure

        #if np.mean(self.pressure_window) < self.PEEP * self.peep_threshold:
        #    print("Pressure drop, did you blow up?")
        #    return True

        return False

    def wait(self, duration):
        time.sleep(duration)

    def step(self, u_in, u_out):
        if isinstance(u_in, torch.Tensor):
            u_in = u_in.detach().numpy()
        if u_out == 1:
            self.breaths += 1
        self.hal.setpoint_in = u_in
        self.hal.setpoint_ex = u_out

        self.prev_time = self.__time
        self.__time = time.time() - self.start

        return self.pressure

    def cleanup(self):
        self.hal.setpoint_in = 0
        self.hal.setpoint_ex = 1
        time.sleep(self.sleep)
        self.hal.setpoint_ex = 0
