import os
from deluca.lung.controllers._pid import PID
from deluca.lung.environments._physical_lung import PhysicalLung
from deluca.lung.utils.scripts.run_controller import run_controller
from deluca.lung.utils.scripts.save_data_and_plot import save_data_and_plot



def run_calibration(R, C, PEEP, directory, name="run", env=None, T=300):
    env = env or PhysicalLung()
    pid = PID(K=[1.0, 0.0, 0.0])
    result = run_controller(pid, R, C, T=T, env=env)
    
    if directory is not None:
        save_data_and_plot(result, f"{directory}/calibration", name)

    return result
