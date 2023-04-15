import numpy as np
import matplotlib
import os
import pickle
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
sys.path.append(PROJECT_ROOT)

from proof_of_concept.circle_ode import TcPINN
from evaluation import evaluate
from sample_points import sample_circle_2d, sample_triangle_2d


def solution_circle_ode(t, y0):
    
    y1t = y0[0] * np.cos(t) - y0[1] * np.sin(t)
    y2t = y0[1] * np.cos(t) + y0[0] * np.sin(t)

    return np.array([y1t, y2t])


def sample_training_data_circle_ode(T, radius, n_pinn, n_semigroup, n_smooth, n_data):

    t_pinn = np.random.uniform(0, T, size=(n_pinn, 1))
    y_pinn = sample_circle_2d(n_pinn, radius)
    X_pinn = np.hstack([t_pinn, y_pinn])

    st_semigroup = sample_triangle_2d(n_semigroup, T)
    y_semigroup = sample_circle_2d(n_semigroup, radius)
    X_semigroup = np.hstack([st_semigroup, y_semigroup])

    t_smooth = np.random.uniform(0, T, size=(n_smooth, 1))
    y_smooth = sample_circle_2d(n_smooth, radius)
    X_smooth = np.hstack([t_smooth, y_smooth])

    t_data = np.random.uniform(0, T, size=(n_data, 1))
    y_data = sample_circle_2d(n_data, radius)
    X_data = np.hstack([t_data, y_data])
    data = np.array([
        solution_circle_ode(t, y0) for t, y0 in zip(t_data.flatten(), y_data)
    ])

    return X_pinn, X_semigroup, X_smooth, X_data, data


def run_and_evaluate(layers, T, training_data, directory, n_run):

    X_pinn = training_data['X_pinn']
    X_semigroup = training_data['X_semigroup']
    X_smooth = training_data['X_smooth']
    X_data = training_data['X_data']
    data = training_data['data']

    model = TcPINN(
        layers=layers, T=T, 
        X_pinn=X_pinn, X_semigroup=X_semigroup, 
        X_smooth=X_smooth, X_data=X_data, data=data
    )
    model.train()

    with open(f"{directory}/model{n_run}.pkl", "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    max_t_evaluate = 10.
    delta_t_evaluate = 0.1
    n_evaluate = 1000

    errors_standard, errors_tc = evaluate(
        model, n_evaluate=n_evaluate, max_t=max_t_evaluate, delta_t=delta_t_evaluate, 
        sampler=sample_circle_2d, ode_solver=solution_circle_ode, sampler_kwargs={'radius': radius}
    )

    with open(f"{directory}/errors_standard{n_run}.pkl", "wb") as handle:
        pickle.dump(errors_standard, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"{directory}/errors_tc{n_run}.pkl", "wb") as handle:
        pickle.dump(errors_tc, handle, protocol=pickle.HIGHEST_PROTOCOL)


layers = [3, 32, 32, 32, 32, 32, 2]
T = 1
n_points = 10
radius = 2

n_runs = 2

# T <-> use a certain loss summand. F <-> don't use a certain loss function.
# The order is: standard pinn loss, semigroup loss, smoothness loss, data loss.
# E.g: FFFT <-> don't use pinn loss, don't use semigroup loss, don't use smoothness loss, use data loss
loss_combinations = [
    "FFFT", "FFTT", "FTFT", "FTTT",
    "TFFF", "TFTF", "TTFF", "TTTF"
]

for loss_combination in loss_combinations:

    os.mkdir(f"runs{loss_combination}")

    use_standard = (loss_combination[0] == "T")
    use_semigroup = (loss_combination[1] == "T")
    use_smooth = (loss_combination[2] == "T")
    use_data = (loss_combination[3] == "T")

    for n_run in range(n_runs):

        print(f"\n Run {n_run} started")
        np.random.seed(n_run)

        X_pinn, X_semigroup, X_smooth, X_data, data = sample_training_data_circle_ode(
            T, radius, n_points, n_points, n_points, n_points
        )

        if not use_standard:
            X_pinn = None

        if not use_semigroup:
            X_semigroup = None

        if not use_smooth:
            X_smooth = None

        if not use_data:
            X_data, data = None, None

        training_data = {
            'X_pinn': X_pinn, 'X_semigroup': X_semigroup, 'X_smooth': X_smooth, 'X_data': X_data, 'data': data
        }

        run_and_evaluate(
            layers, T, training_data, directory=f"runs{loss_combination}", n_run=n_run
        )