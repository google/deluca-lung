import os
import pickle
import matplotlib.pyplot as plt

from deluca.lung.utils import Analyzer


def save_data_and_plot(result, directory, name):
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(result, open(f"{directory}/{name}.pkl", "wb"))
    Analyzer(result).plot()
    plt.savefig(f"{directory}/{name}.png")