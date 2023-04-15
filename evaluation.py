import numpy as np
import matplotlib.pyplot as plt


def evaluate(model, n_evaluate: int, max_t: float, delta_t: float, sampler: callable, ode_solver: callable, sampler_kwargs=dict()):
    """
    Compute the Euclidean distance between the true solution and the prediction of the tcPINN model.
    For each i= 1, ..., 'n_evaluate', we sample a random initial value and compute the error at
    each time point.
    """
    times = np.linspace(0, max_t, int(max_t / delta_t) + 1)
    n_timepoints = len(times)
    
    errors_standard = np.empty((n_evaluate, n_timepoints), dtype=float)
    errors_tc = np.empty((n_evaluate, n_timepoints), dtype=float)
    
    y0s = sampler(n_evaluate, **sampler_kwargs)
    
    for i, y0 in enumerate(y0s):
        
        solution_true = np.array([
            ode_solver(t, y0)
            for t in times
        ])
        
        solution_standard = model.predict_standard(max_t, delta_t, y0)
        errors_standard[i] = np.sqrt(np.sum((solution_standard - solution_true) ** 2, axis=1))
        
        solution_tc = model.predict_tc(max_t, delta_t, y0)
        errors_tc[i] = np.sqrt(np.sum((solution_tc - solution_true) ** 2, axis=1))
    
    return errors_standard, errors_tc


def plot_errors_over_time(max_t: float, delta_t: float, errors_standard: np.ndarray, errors_tc: np.ndarray, outfile=None):

    fig, ax = plt.subplots(figsize=(6,6))
    ax.spines[['top', 'right']].set_visible(False)
    ax.set(xlabel="time", ylabel="mean error", title="Error over time")

    times = np.linspace(0, max_t, int(max_t / delta_t) + 1)

    ax.errorbar(
        times, np.mean(errors_standard, axis=0), np.std(errors_standard, axis=0),
        marker=".", label="standard", color="#03468F"
    )
    ax.errorbar(
        times, np.mean(errors_tc, axis=0), np.std(errors_tc, axis=0),
        marker=".", label="tcPINN", color="#A51C30"
    )
    plt.legend()

    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight")

    return fig, ax