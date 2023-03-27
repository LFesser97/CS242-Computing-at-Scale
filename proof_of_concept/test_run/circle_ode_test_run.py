import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
sys.path.append(PROJECT_ROOT)

from proof_of_concept.circle_ode import TcPINN
from evaluation import evaluate, plot_errors_over_time
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


def plot_trajectories_circle_ode(solution_true, solution_standard, solution_tc, outfile=None):

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.spines[['top', 'right']].set_visible(False)

    radius = np.max(np.concatenate([solution_true])) + .5
    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])    

    ax.plot(solution_true[:, 0], solution_true[:, 1], '.-', label="true solution", color="orange")
    ax.plot(solution_standard[:, 0], solution_standard[:, 1], '.-', label="standard", color="#03468F")
    ax.plot(solution_tc[:, 0], solution_tc[:, 1], '.-', label="tcPINN", color="#A51C30")

    plt.legend()

    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight")

    return fig, ax


def plot_example_trajectory(model, max_t, delta_t, y0):

	solution_true = np.array([
	    solution_circle_ode(t, y0) 
	    for t in np.linspace(0, max_t, int(max_t / delta_t) + 1)
	])
	solution_standard = model.predict_standard(max_t, delta_t, y0)
	solution_tc = model.predict_tc(max_t, delta_t, y0)

	plot_trajectories_circle_ode(solution_true, solution_standard, solution_tc, outfile="example_trajectory.pdf")


T = 1
radius = 3
n_points = 10
layers = [3, 32, 32, 32, 2]

X_pinn, X_semigroup, X_smooth, X_data, data = sample_training_data_circle_ode(T, radius, n_points, n_points, n_points, n_points)

model = TcPINN(
	layers, T, 
	X_pinn=X_pinn, X_semigroup=X_semigroup, X_smooth=X_smooth, X_data=X_data, data=data
)
model.train()

with open("test_model_circle_ode.pkl", "wb") as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("test_model_circle_ode.pkl", "rb") as f:
	model = pickle.load(f)

plot_example_trajectory(model, max_t=10., delta_t=.1, y0=np.array([1., 0.]))

n_evaluate = 100
max_t_evaluate = 2.
delta_t_evaluate = .1

errors_standard, errors_tc = evaluate(
	model, n_evaluate=n_evaluate, max_t=max_t_evaluate, delta_t=delta_t_evaluate, 
	sampler=sample_circle_2d, ode_solver=solution_circle_ode, sampler_kwargs={'radius': radius}
)

plot_errors_over_time(
    max_t=max_t_evaluate, delta_t=delta_t_evaluate,
    errors_standard=errors_standard, errors_tc=errors_tc, outfile="test_errors_over_time.pdf"
)