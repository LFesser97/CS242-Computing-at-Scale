# tcPINNs: Proof of concept

------------------------


We provide a baseline implemenation of time-consistent PINNs for a simple system of ODEs of size two. The code is structured as follows:

1) An MLP class to define an MLP of arbitrary layer size with Tanh activation. In the tcPINN implementation, the inputs of the MLP will correspond to the time and the initial state.

2) A TcPINN class implementing the time-consistent physics-informed neural network. The input of the network are the time and the initial state of the ODE solution. The output is approximates the solution of the ODE for a given time and intial state. This class includes the definition of the standard PINN loss, the semgigroup loss, smoothness loss, and the data loss.

3) An example of how to generate the training data points and train a tcPINN.

4) A visualization of the predicted time-consistent ODE solution beyond the maximum time in the training dataset.

5) An evaluation of the tcPINN measuring the average Euclidean distance of points on the predicted solution to the (a priori known) true solution.


-------------------------------------------
