"""
main.py

Created on Sat Nov 26 14:20:00 2022

@author: Lukas

Main script that contains all the classes used in the experiments.
"""

# import packages

import numpy as np
import os
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# class for analytic ODE solutions

class ODE_analytic_sol:

    def __init__(self, name, dim, dynamic, y_true):
        """
        Parameters
        ----------
        name : string
            Name of the ODE.

        dim : int
            Dimension of the ODE.

        dynamic : string
            Type of the ODE.

        y_true : function
            Analytic solution of the ODE.

        Returns
        -------
        None.

        """
        self.name = name
        self.dim = dim
        self.dynamic = dynamic
        self.y_true = y_true


    def plot_sol(self, y_0, max_t, delta_t, ax = None):
        """
        Plots the analytic solution of the ODE.

        Parameters
        ----------
        y_0 : np.array
            Initial condition of the ODE.

        max_t : float
            Maximum time of the ODE.

        delta_t : float
            Time step of the ODE.

        ax : matplotlib.axes._subplots.AxesSubplot, optional

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
        
        """
        y = np.stack(np.array(time_series), axis=1) # need to adjust this
        ax.plot(y[0], y[1], '.-')
    
        return ax


    def __generate_training_data(sample_method):
        """
        Generates training data for the ODE.

        Parameters
        ----------
        sample_method : string
            Method used to sample the training data.

        Returns
        -------
        path_standard : pd.DataFrame

        path_dac : pd.DataFrame
        
        """
        # adapt data_standard_approach in proof_of_concept.ipynb ?!

        # adapt data_dac_approach in proof_of_concept.ipynb ?!

        pass


    def training_data_to_dataset(path_1, path_2):
        "to be implemented"
        pass


class PINN_ODE_Dataset:

    def __init__(self, delta_t):
        """
        Parameters
        ----------
        delta_t : float
            Time step.

        Returns
        -------
        None.

        """
        self.delta_t = delta_t
    
