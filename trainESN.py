# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 02:01:56 2022

@author: Leon Scharwächter
"""

from echoStateNetwork import ESN as ESNModel
from utils import scaleMatrix, sparseMaker, getEchoStateProperty
import numpy as np

def initialize_ESN(n_input = 1, n_reservoir = 40, n_output = 1, scalingfactor = 1e-8):
    Win = (np.random.rand(n_input, n_reservoir)-0.5)
    Wr = (np.random.rand(n_reservoir, n_reservoir)-0.5)
    Wfb = (np.random.rand(n_output, n_reservoir)-0.5)
    Wout = np.zeros((n_reservoir, n_output))
    
    # Scale matrices
    Win = scaleMatrix(Win, scalingFactor = 1.0) 
    Wr = scaleMatrix(Wr, scalingFactor = 0.01)
    Wfb = scaleMatrix(Wfb, scalingFactor = 0.01)
    
    # Transform matrix Wr to fulfil the Echo State Property
    Wr = getEchoStateProperty(Wr, spectralRadius = 0.75)
    
    # Create sparse matrices
    Win = sparseMaker(Win, 0.3)
    Wfb = sparseMaker(Wfb, 0.4)
    Wr = sparseMaker(Wr, 0.3)

    ESN = ESNModel(
        Win, Wr, Wout, Wfb,
        n_input=n_input, 
        n_reservoir=n_reservoir, 
        n_output=n_output
        )
    return ESN

def create_signal(T1=5, T2=2, A1=0.9, A2=0.4, timesteps=500):
    t = np.arange(0, timesteps, 1)
    signal = A1 * np.sin(t / T1) + A2 * np.sin(t / T2)
    return signal.reshape(1, -1)  # Shape: (1, timesteps)


if __name__ == "__main__":
    ESN = initialize_ESN()
    signal = create_signal()
    T, Wout_new = ESN.train(signal, 300)
    prediction, NRMSE = ESN.predict(signal, T, Wout_new, 300, 200)
    print('NRMSE:',NRMSE)
