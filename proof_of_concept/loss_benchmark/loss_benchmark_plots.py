import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys


def plot_errors(max_t: float, delta_t: float, errors: np.ndarray, ax=None, **kwargs):

    if ax is None:
        
        _, ax = plt.subplots(figsize=(6,6))
        ax.spines[['top', 'right']].set_visible(False)
        
        # change title
        ax.set(xlabel="time", ylabel="mean error", title="Error over time")
        #ax.set(xlabel="time", ylabel="mean error", title="Error over time (data driven)")

    times = np.linspace(0, max_t, int(max_t / delta_t) + 1)
    ax.plot(times, np.mean(errors, axis=0), **kwargs)

    return ax


# Important: These parameters were used when evaluating the tcPINN for the circle ODE
n_runs = 50
max_t = 10
delta_t = 0.1
times = np.linspace(0, max_t, int(max_t / delta_t) + 1)


# T <-> use a certain loss summand. F <-> don't use a certain loss function.
# The order is: standard pinn loss, semigroup loss, smoothness loss, data loss.
# E.g: FFFT <-> don't use pinn loss, don't use semigroup loss, don't use smoothness loss, use data loss
loss_combinations = [
    "TFFF", "TFTF", "TTFF", "TTTF"
    #"FFFT", "FFTT", "FTFT", "FTTT",
]


labels = ["w/o sg and smooth", "w/o sg", "w/o smooth", "tcPINN"]
ax = None

for loss_combination, label in zip(loss_combinations, labels):

    mean_errors_runs = np.empty((n_runs, len(times)), dtype=float)

    for n_run in range(n_runs):

        with open(f"runs{loss_combination}/errors_tc{n_run}.pkl", "rb") as f:
            errors_run = pickle.load(f)

        # mean over evaluation points
        mean_errors_runs[n_run] = np.mean(errors_run, axis=0)

    ax = plot_errors(max_t, delta_t, mean_errors_runs, ax=ax, label=label)


mean_errors_standard = np.empty((n_runs, len(times)), dtype=float)

for n_run in range(n_runs):

    with open(f"runsTFFF/errors_standard{n_run}.pkl", "rb") as f:
    #with open(f"runsFFFT/errors_standard{n_run}.pkl", "rb") as f:
        errors_run = pickle.load(f)

    # mean over evaluation points
    mean_errors_standard[n_run] = np.mean(errors_run, axis=0)

ax = plot_errors(1.5, 0.1, mean_errors_standard[:, :16], ax=ax, label="standard")


plt.legend()
plt.savefig("loss_benchmark.pdf", bbox_inches="tight")
#plt.savefig("loss_benchmark_data_driven.pdf", bbox_inches="tight")