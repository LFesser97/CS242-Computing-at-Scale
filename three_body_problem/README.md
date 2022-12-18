# tcPINNs: The three-body problem

------------------------


We provide implementations of time-consistent physics-informed neural networks (tcPINNs) for the three-body problem. We have observed that during training, the tcPINN is unable to completely learn the dynamics. That is, the PINN loss does not converge to zero.

We hypothesize that this phenomenon is not caused by the ODE system being chaotic, but rather by the velocities of bodies converging to infinity during close encounters. This causes the loss function to have singularities, which makes training significantly more difficult.

When the barycenter of the three bodies is constant in time, the size of the ODE system can be reduced from twelve to eight. Without loss of generality, the domain of intial values can be further reduced by exploiting the roation and scale invariance of the ODE system. The provided jupyter notebooks iteratively simplify the training task and reduce the loss function after training by two orders of magnitude, relative to the first naive implementation.

* `three_body_problem_naive.ipynb`: A naive implementation of tcPINNs to solve the planar three body problem.

* `three_body_problem_center_barycenter.ipynb`: A tcPINN implementation for the planar three-body problem with constant barycenter.

* `three_body_problem_center_barycenter_restricted_ivp.ipynb`: A tcPINN implementation for the planar three-body problem with constant barycenter. Without loss of generality, the domain of the initial position was significantly reduced.

-------------------------------------------
