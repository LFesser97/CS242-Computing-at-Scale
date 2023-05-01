import numpy as np
import os
import pickle
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
sys.path.append(PROJECT_ROOT)

import circle_ode
import sample_points


def get_training_data(X_pinn, X_semigroup, X_smooth, X_data, data, loss_combination="TTTT"):
    """
    T <-> use a certain loss summand. F <-> don't use a certain loss function.
    The order is: standard pinn loss, semigroup loss, smoothness loss, data loss.
    E.g: FFFT <-> don't use pinn loss, don't use semigroup loss, don't use smoothness loss, use data loss
    """
    if loss_combination[0] == "F": X_pinn = None
    if loss_combination[1] == "F": X_semigroup = None
    if loss_combination[2] == "F": X_smooth = None
    if loss_combination[3] == "F": X_data, data = None, None

    training_data = {
        'X_pinn': X_pinn, 'X_semigroup': X_semigroup, 
        'X_smooth': X_smooth, 'X_data': X_data, 'data': data
    }
    return training_data


def run(layers, T, n_run, training_data, outdir):

    model = circle_ode.CircleODE(
        layers=layers, T=T, 
        X_pinn=training_data['X_pinn'],
        X_semigroup=training_data['X_semigroup'], 
        X_smooth=training_data['X_smooth'], 
        X_data=training_data['X_data'], data=training_data['data']
    )
    model.train()

    with open(f"{outdir}/model{n_run}.pkl", "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return model


def evaluate(model, n_evaluate: int, max_t: float, delta_t: float, radius: float):
    """
    Compute the Euclidean distance between the true solution and the prediction of the tcPINN model.
    For each i= 1, ..., 'n_evaluate', we sample a random initial value and compute the error at
    each time point.
    """
    times = np.linspace(0, max_t, int(max_t / delta_t) + 1)
    errors_standard = np.empty((n_evaluate, len(times)), dtype=float)
    errors_tc = np.empty((n_evaluate, len(times)), dtype=float)
    y0s = sample_points.uniform_circle_2d(n_evaluate, radius)
    
    for i, y0 in enumerate(y0s):
        
        true_solution = circle_ode.get_solution(max_t, delta_t, y0)
        standard_solution = model.predict_standard(max_t, delta_t, y0)
        tc_solution = model.predict_tc(max_t, delta_t, y0)
        errors_standard[i] = np.sqrt(np.sum((standard_solution - true_solution) ** 2, axis=1))
        errors_tc[i] = np.sqrt(np.sum((tc_solution - true_solution) ** 2, axis=1))
    
    return errors_standard, errors_tc


def run_and_evaluate(layers, T, n_run, training_data, outdir, radius):

    model = run(layers, T, n_run, training_data, outdir)
    errors_standard, errors_tc = evaluate(model, n_evaluate=1000, max_t=10.0, delta_t=0.1, radius=radius)

    with open(f"{outdir}/errors_standard{n_run}.pkl", "wb") as handle:
        pickle.dump(errors_standard, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"{outdir}/errors_tc{n_run}.pkl", "wb") as handle:
        pickle.dump(errors_tc, handle, protocol=pickle.HIGHEST_PROTOCOL)


layers = [3] + 5 * [32] + [2]
T = 1
radius = 2
n_points = 100
n_runs = 50

loss_combinations = ["FFFT", "FFTT", "FTFT", "FTTT", "TFFF", "TFTF", "TTFF", "TTTF"]

for loss_combination in loss_combinations:

    os.mkdir(f"runs{loss_combination}")

    for n_run in range(n_runs):

        print(f"\n Run {n_run} started")
        np.random.seed(n_run)
        X_pinn, X_semigroup, X_smooth, X_data, data = circle_ode.sample_training_data(
            T, radius, n_points, n_points, n_points, n_points
        )
        training_data = get_training_data(
            X_pinn, X_semigroup, X_smooth, X_data, data, loss_combination
        )
        run_and_evaluate(
            layers, T, n_run=n_run, training_data=training_data,
            outdir=f"runs{loss_combination}", radius=radius 
        )