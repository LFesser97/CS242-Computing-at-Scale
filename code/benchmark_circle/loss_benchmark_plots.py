import matplotlib.pyplot as plt
import numpy as np
import pickle


# Important: These parameters were used when evaluating the tcPINN for the circle ODE
N_RUNS = 50
MAX_T = 10
DELTA_T = 0.1
TIMES = np.linspace(0, MAX_T, int(MAX_T / DELTA_T) + 1)


def plot_errors(max_t: float, delta_t: float, errors: np.ndarray, ax=None, **kwargs):

    if ax is None:
        
        _, ax = plt.subplots(figsize=(6,6))
        ax.spines[['top', 'right']].set_visible(False)
        ax.set(xlabel="time", ylabel="mean error", title="Error over time")
        #ax.set(xlabel="time", ylabel="mean error", title="Error over time (data driven)")

    times = np.linspace(0, max_t, int(max_t / delta_t) + 1)
    ax.plot(times, np.mean(errors, axis=0), **kwargs)

    return ax


loss_combinations = ["TFFF", "TFTF", "TTFF", "TTTF"]
# loss_comibnations = ["FFFT", "FFTT", "FTFT", "FTTT"]
labels = ["w/o sg and smooth", "w/o sg", "w/o smooth", "tcPINN"]
ax = None

for loss_combination, label in zip(loss_combinations, labels):

    mean_errors_runs = np.empty((N_RUNS, len(TIMES)), dtype=float)

    for n_run in range(N_RUNS):

        with open(f"runs{loss_combination}/errors_tc{n_run}.pkl", "rb") as f:
            errors_run = pickle.load(f)

        # mean over evaluation points
        mean_errors_runs[n_run] = np.mean(errors_run, axis=0)

    ax = plot_errors(MAX_T, DELTA_T, mean_errors_runs, ax=ax, label=label)


mean_errors_standard = np.empty((N_RUNS, len(TIMES)), dtype=float)

for n_run in range(N_RUNS):

    with open(f"runsTFFF/errors_standard{n_run}.pkl", "rb") as f:
    #with open(f"runsFFFT/errors_standard{n_run}.pkl", "rb") as f:
        errors_run = pickle.load(f)

    # mean over evaluation points
    mean_errors_standard[n_run] = np.mean(errors_run, axis=0)

ax = plot_errors(1.5, 0.1, mean_errors_standard[:, :16], ax=ax, label="standard")


plt.legend()
plt.savefig("loss_benchmark.pdf", bbox_inches="tight")
#plt.savefig("loss_benchmark_data_driven.pdf", bbox_inches="tight")