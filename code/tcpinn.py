import numpy as np
import torch

from abc import ABC, abstractmethod
from collections import OrderedDict


if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")


class MLP(torch.nn.Module):
    """
    Define an multilayer perceptron (MLP) with arbitrary layers and Tanh activation. 
    In the time-consistent physics-informed neural network (tcPINN) implementation,
    the inputs of the MLP will correspond to the time and the initial state (t, 𝑦0).
    The output will correspond to the solution of the ODE y(t; y0) at time t for
    the given initial value y0.
    """
    def __init__(self, layers):
        """
        Build the MLP.
        
        Input:
        ------
        layers: list
            A list that specifies the number of neurons for each layer.
            Entry i of 'layers' specifies the number of neurons in layer i.
        """
        super().__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh
        layer_list = list()
        
        for i in range(self.depth - 1):
            
            linear_layer = torch.nn.Linear(layers[i], layers[i+1])
            torch.nn.init.xavier_normal_(linear_layer.weight, gain=5/3)
            torch.nn.init.zeros_(linear_layer.bias.data)
            
            layer_list.append(
                (f"layer_{i}", linear_layer)
            )
            layer_list.append((f"activation_{i}", self.activation()))
        
        last_layer = torch.nn.Linear(layers[-2], layers[-1])
        torch.nn.init.xavier_normal_(last_layer.weight)
        torch.nn.init.zeros_(last_layer.bias.data)
        
        layer_list.append(
            (f"layer_{self.depth - 1}", last_layer)
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
    
    
    def forward(self, x):
        return self.layers(x)


class TcPINN(ABC):
    """
    A general time-consistent physics-informed neural network (tcPINN) implementation.
    """
    def __init__(
        self, layers, T, X_pinn=None, X_semigroup=None, X_smooth=None, 
        X_data=None, data=None, w_pinn=1., w_semigroup=1., w_smooth=1., w_data=1.
    ):
        """
        Initialize the MLP and the training data. It is possible to use a subset of the 
        four loss functions {standard PINN loss, semigroup loss, smoothness loss, data loss} by
        only providing some of the training data.
        
        Input:
        ------
        layers: list
            A list that specifies the number of neurons for each layer of the MLP.
            Entry i of 'list' specifies the number of neurons in layer i.
        
        T: float
            The supremum of all times in the training data. When predicting the solution of
            the ODE after time T, the trajectory has to be stitched together.
        
        X_pinn: np.ndarray
            The training data for the standard PINN loss. The first column of 'X_pinn'
            corresponds to the times, the other columns correspond to the intial values.
        
        X_semigroup: np.ndarray
            The training data for the semigroup loss. The first column of 'X_semigroup'
            corresponds to the length of the first time step, the second to the length of the
            second time step, and the other columns correspond to the intial values.
        
        X_smooth: np.ndarray
            The training data for the smoothness loss. The first column of 'X_smooth'
            corresponds to the times, the other columns correspond to the intial values.
        
        X_data: np.ndarray
            The training data for the data loss. The first column of 'X_data'
            corresponds to the times, the other columns correspond to the intial values.
        
        data: np.ndarray
            The measured/true values of the ODE for the times and initial values given by
            'X_data'.
        
        w_pinn, w_semigroup, w_smooth, w_data: floats
            Weights for the respective loss functions. For example, if w_semigroup=2, the
            semigroup loss is scaled by a factor of two.
        """
        self.history = {"loss": []}
        self.ode_dimension = layers[-1]
        self.is_inverse = False
        self.mlp = MLP(layers).to(device)
        self.optimizer = torch.optim.LBFGS(
            self.mlp.parameters(), lr=1., max_iter=50000, max_eval=50000, 
            history_size=10, tolerance_grad=1e-5, tolerance_change=np.finfo(float).eps, 
            line_search_fn="strong_wolfe"
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", min_lr=1e-4, verbose=True
        )
        self.iter = 0
        self.T = torch.tensor(T).float().to(device)
        self._init_training_data(X_pinn, X_semigroup, X_smooth, X_data, data)
        self.w_pinn, self.w_semigroup, self.w_smooth, self.w_data = w_pinn, w_semigroup, w_smooth, w_data        
    

    def _init_training_data(self, X_pinn, X_semigroup, X_smooth, X_data, data):
        
        if X_pinn is not None:
            self.t_pinn = torch.tensor(X_pinn[:, :1], requires_grad=True).float().to(device)
            self.y_pinn = torch.tensor(X_pinn[:, 1:], requires_grad=True).float().to(device)
            self.history["loss_pinn"] = []
            self.use_standard = True
        
        else:
            self.use_standard = False
        
        if X_semigroup is not None:
            self.s_semigroup = torch.tensor(X_semigroup[:, :1], requires_grad=True).float().to(device)
            self.t_semigroup = torch.tensor(X_semigroup[:, 1:2], requires_grad=True).float().to(device)
            self.y_semigroup = torch.tensor(X_semigroup[:, 2:], requires_grad=True).float().to(device)
            self.history["loss_semigroup"] = []
            self.use_semigroup = True
        
        else:
            self.use_semigroup = False
        
        if X_smooth is not None:
            self.t_smooth = torch.tensor(X_smooth[:, :1], requires_grad=True).float().to(device)
            self.y_smooth = torch.tensor(X_smooth[:, 1:], requires_grad=True).float().to(device)
            self.history["loss_smooth"] = []
            self.use_smooth = True
        
        else:
            self.use_smooth = False
        
        if X_data is not None:
            if data is None:
                raise ValueError("The true/measured solution for the 'X_data' points has to be provided.")
            self.t_data = torch.tensor(X_data[:, :1], requires_grad=True).float().to(device)
            self.y_data = torch.tensor(X_data[:, 1:], requires_grad=True).float().to(device)
            self.data = torch.tensor(data, requires_grad=True).float().to(device)
            self.history["loss_data"] = []
            self.use_data = True
        
        else:
            self.use_data = False
        
        self.loss_names = list(self.history.keys())
    
    
    def net_y(self, t, y0):
        """
        Let N(t, y0) denote the value of the MLP at time t for initial value y0.
        Let M(t, y0) denote the value of the tcPINN at time t for initial value y0.
        
        We set M(t, y0) = y0 + t * N(t, y0) to guarantee continuity of the trajectory
        when stitching together solutions for large times.
        """
        y = y0 + t * self.mlp(torch.cat([t, y0], dim=1))
        
        return y
    
    
    def net_derivative(self, t, y0):
        """
        Pytorch automatic differentiation to compute the derivatives of the neural network
        with respect to time.
        
        Output:
        -------
        derivatives: list
            The i-th entry is the time derivative of the i-th output component of the neural network
            evaluated at all inputs (t, y0). Each input corresponds to one row in each tensor.
        """
        y = self.net_y(t, y0)
        
        # vectors for the autograd vector Jacobian product 
        # to compute the derivatives w.r.t. every output component
        vectors = [torch.zeros_like(y) for _ in range(self.ode_dimension)]
        
        for i, vec in enumerate(vectors):
            vec[:,i] = 1.
        
        derivatives = [
            torch.autograd.grad(
                y, t, 
                grad_outputs=vec,
                retain_graph=True,
                create_graph=True
            )[0]
            for vec in vectors
        ]
        
        return derivatives
    
    
    @abstractmethod
    def _loss_pinn(self):
        """
        The ODE-specific standard PINN loss.
        """
        pass
    
    
    def _loss_semigroup(self):
        """
        The general semigroup loss.
        """
        y_no_restart = self.net_y(self.s_semigroup + self.t_semigroup, self.y_semigroup)
        y_s = self.net_y(self.s_semigroup, self.y_semigroup)
        y_restart = self.net_y(self.t_semigroup, y_s)
        loss = self.w_semigroup * self.ode_dimension * torch.mean((y_no_restart - y_restart) ** 2)
        
        return loss
    
    
    def _loss_smooth(self):
        """
        The general smoothness loss.
        """
        deriv_below = self.net_derivative(self.t_smooth, self.y_smooth)
        y = self.net_y(self.t_smooth, self.y_smooth)
        deriv_above = self.net_derivative(torch.zeros_like(self.t_smooth, requires_grad=True), y)
        
        loss = .0
        
        for d1, d2 in zip(deriv_below, deriv_above):
            loss += torch.mean((d1 - d2) ** 2)
        
        loss *= self.w_smooth

        return loss
    
    
    def _loss_data(self):
        """
        The general data loss.
        """
        y = self.net_y(self.t_data, self.y_data)
        loss = self.w_data * self.ode_dimension * torch.mean((y - self.data) ** 2)

        return loss
    
    
    def _update_history(self, loss, loss_pinn=None, loss_semigroup=None, loss_smooth=None, loss_data=None):

        self.history["loss"].append(loss.item())

        if self.use_standard:
            self.history["loss_pinn"].append(loss_pinn.item())
        
        if self.use_semigroup:
            self.history["loss_semigroup"].append(loss_semigroup.item())

        if self.use_smooth:
            self.history["loss_smooth"].append(loss_smooth.item())
        
        if self.use_data:
            self.history["loss_data"].append(loss_data.item())
        
        if self.is_inverse:

            parameters = sorted(
                set(self.history.keys()) - {"loss", "loss_pinn", "loss_semigroup", "loss_smooth", "loss_data"}
            )

            for parameter in parameters:
                self.history[parameter].append(getattr(self.mlp, parameter).item())
    

    def _print_history(self):
        
        info = f"iteration {self.iter}"
        info += f"; loss: {self.history['loss'][-1]:.4f}"
        
        if self.use_standard:
            info += f", loss_pinn: {self.history['loss_pinn'][-1]:.4f}"
        
        if self.use_semigroup:
            info += f", loss_semigroup: {self.history['loss_semigroup'][-1]:.4f}"

        if self.use_smooth:
            info += f", loss_smooth: {self.history['loss_smooth'][-1]:.4f}"
        
        if self.use_data:
            info += f", loss_data: {self.history['loss_data'][-1]:.4f}"

        if self.is_inverse:
            
            loss_names = ["loss", "loss_semigroup", "loss_smooth", "loss_data"]
            parameters = sorted(set(self.history.keys()) - set(loss_names))

            for parameter in parameters:
                info += f", {parameter}: {self.history[parameter][-1]:.4f}"
        
        print(info)
    

    def loss_function(self):
        
        self.optimizer.zero_grad()
        loss_pinn, loss_semigroup, loss_smooth, loss_data = None, None, None, None
        loss = .0
        
        if self.use_standard:
            loss_pinn = self._loss_pinn()
            loss += loss_pinn
        
        if self.use_semigroup:
            loss_semigroup = self._loss_semigroup()
            loss += loss_semigroup
        
        if self.use_smooth:
            loss_smooth = self._loss_smooth()
            loss += loss_smooth
        
        if self.use_data:
            loss_data = self._loss_data()
            loss += loss_data
        
        loss.backward()
        self.scheduler.step(loss)
        self.iter += 1
        self._update_history(
            loss=loss, loss_pinn=loss_pinn, loss_semigroup=loss_semigroup, 
            loss_smooth=loss_smooth, loss_data=loss_data
        )
        
        if self.iter % 100 == 0:
            self._print_history()
        
        return loss    
    
    
    def train(self):
        """
        Train the MLP parameters with the LBFGS optimizer.
        """
        self.mlp.train()
        self.optimizer.step(self.loss_function)
    
    
    def predict(self, t, y0):
        """
        Evaluate the tcPINN at times 't' for initial values 'y0'.
        Each row (!) in 't' and 'y0' is one time point and intial value, respectively.
        """
        t = torch.tensor(t).float().to(device)
        y0 = torch.tensor(y0).float().to(device)
        
        self.mlp.eval()
        prediction = self.net_y(t, y0)
        prediction = prediction.detach().cpu().numpy()
        
        return prediction
    
    
    def predict_standard(self, max_t, delta_t, y0):
        """
        Predict the solution until time 'max_t' with step size 'delta_t' for
        a single (!) initial value 'y0'.
        
        The tcPINN is not (!) applied multiple times to stitch together trajectories.
        The predicted trajectory will therefore most likely be incorrect for all times 
        t > self.T.
        """        
        times = np.linspace(0, max_t, int(max_t / delta_t) + 1)
        times = times[:, np.newaxis]
        y0s = np.repeat(y0[np.newaxis,:], len(times), axis=0)
        
        trajectory = self.predict(times, y0s)
        
        return trajectory
    
    
    def predict_tc(self, max_t, delta_t, y0):
        """
        Predict the solution until time 'max_t' with step size 'delta_t' for
        a single (!) initial value 'y0'.
        
        The tcPINN is applied multiple times to stitch together trajectories.
        
        Note: 
            If delta_t does not devide self.T, the behavior of this function
            might be unexpected.
        """
        T = self.T.detach().cpu().numpy()
        times = np.arange(0, np.round(T + delta_t, 3), delta_t)[1:]
        times = times[:, np.newaxis]
        n_resets = int(np.ceil(max_t / self.T.detach().cpu().numpy()))
        
        trajectory = np.array([y0])
        
        for _ in range(n_resets):
            
            y0 = trajectory[-1]
            y0s = np.repeat(y0[np.newaxis,:], len(times), axis=0)
            segment =  self.predict(times, y0s)
            trajectory = np.vstack([trajectory, segment])
        
        return trajectory