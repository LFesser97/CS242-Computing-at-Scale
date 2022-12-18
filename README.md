# tcPINNs (Time-consistent Physics-informed Neural Networks)

------------------------


This is an implementation of Time-consistent Physics-informed Neural Networks (tcPINNs) using ***PyTorch***. We implemented the idea of solving multiple initial value problems by using an additional semigroup and smoothness loss function for three ODE systems. The code for each ODE system can be found in it's respective subfolder of this repository:

* A proof of concept (linear) ODE system of size two with the solutions being circles in two-dimensional space.

* The non-linear Lotka-Volterra ODE system of size two.

* The planar three-body problem. By exploiting the symmetry and scale invariance for restricted initial value problems, the training task was further simplified.

We demonstrate how to improve the accurcacy of tcPINNS by combining our approach with Neural Architecture Search (NAS). A general framework and an example of how to apply NAS to tcPINNs can be found in the `Neural Arichtecture Search` subfolder.

-------------------------------------------

## Attribute

**Original Work**: *Benedikt Geiger, Lukas Fesser, Hainan Xiong, Haoming Chen*

**Github Repo** : https://github.com/LFesser97/CS242-Computing-at-Scale

-------------------------------------------

## Dependencies

Major Dependencies:

 - ***PyTorch (for PyTorch Implementation)***: ```pip install --upgrade torch``
 - ***Jupyter Notebook/Lab***: ```pip install jupyterlab``` (JupyterLab) or ```pip install notebook```

Peripheral Dependencies:
 
 - ***numpy***: ```pip install numpy```
 - ***matplotlib***: ```pip install matplotlib```
