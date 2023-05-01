"""
This is a proof of concept implementation of time-consistent physics informed neural networks.

We consider the linear ODE

\begin{align*}
    \frac{d}{dt} \begin{pmatrix} y_1 \\ y_2 \end{pmatrix}(t) = \begin{pmatrix} -y_2 \\ y_1 \end{pmatrix} (t).
\end{align*}
"""
import numpy as np
import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
sys.path.append(PROJECT_ROOT)

import sample_points
from tcpinn import TcPINN



class CircleODE(TcPINN):
    """
    A tcPINN implementation of the linear circle ODE.
    """
    def __init__(
        self, layers, T, X_pinn=None, X_semigroup=None, X_smooth=None, X_data=None, data=None,
        w_pinn=1., w_semigroup=1., w_smooth=1., w_data=1.
    ):
        super().__init__(
            layers, T, X_pinn, X_semigroup, X_smooth, X_data, data,
            w_pinn, w_semigroup, w_smooth, w_data
        )
    
    
    def _loss_pinn(self):
        """
        The ODE-specific standard PINN loss.
        """
        y = self.net_y(self.t_pinn, self.y_pinn)
        deriv = self.net_derivative(self.t_pinn, self.y_pinn)
        
        loss1 = torch.mean((deriv[0] + y[:, 1:2]) ** 2)
        loss2 = torch.mean((deriv[1] - y[:, 0:1]) ** 2)
        loss = self.w_pinn * (loss1 + loss2)
        
        return loss


def solution_circle_ode(t, y0):
    
    y1t = y0[0] * np.cos(t) - y0[1] * np.sin(t)
    y2t = y0[1] * np.cos(t) + y0[0] * np.sin(t)

    return np.array([y1t, y2t])


def get_solution(max_t, delta_t, y0):

    times = np.linspace(0, max_t, int(max_t / delta_t) + 1)
    y = np.array([
        solution_circle_ode(t, y0) for t in times
    ])
    
    return y


def sample_training_data(T, radius, n_pinn, n_semigroup, n_smooth, n_data):

    t_pinn = np.random.uniform(0, T, size=(n_pinn, 1))
    y_pinn = sample_points.uniform_circle_2d(n_pinn, radius)
    X_pinn = np.hstack([t_pinn, y_pinn])

    st_semigroup = sample_points.uniform_triangle_2d(n_semigroup, T)
    y_semigroup = sample_points.uniform_circle_2d(n_semigroup, radius)
    X_semigroup = np.hstack([st_semigroup, y_semigroup])

    t_smooth = np.random.uniform(0, T, size=(n_smooth, 1))
    y_smooth = sample_points.uniform_circle_2d(n_smooth, radius)
    X_smooth = np.hstack([t_smooth, y_smooth])

    t_data = np.random.uniform(0, T, size=(n_data, 1))
    y_data = sample_points.uniform_circle_2d(n_data, radius)
    X_data = np.hstack([t_data, y_data])
    data = np.array([
        solution_circle_ode(t, y0) for t, y0 in zip(t_data.flatten(), y_data)
    ])

    return X_pinn, X_semigroup, X_smooth, X_data, data