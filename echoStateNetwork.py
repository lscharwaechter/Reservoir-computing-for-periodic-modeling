# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:39:30 2021

@author: Leon Scharwächter
"""
import matplotlib.pyplot as plt
import numpy as np

class ESN():
    def __init__(self, Win, Wr, Wout, Wfb,
                 n_input=1, n_reservoir=40, n_output=1):
        super().__init__()
        
        self.Win = Win
        self.Wr = Wr 
        self.Wout = Wout
        self.Wfb = Wfb 
        
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.n_output = n_output
        
    def resetStates(self, steps):
        # Initialize net input and activation of the reservoir and output layer with 0
        self.net_h = np.zeros((self.n_reservoir,steps))
        self.act_h = np.zeros((self.n_reservoir,steps))   
        self.net_hk = np.zeros((self.n_output,steps))
        self.act_hk = np.zeros((self.n_output,steps))
        
    def train(self, x, steps):
        self.resetStates(x.shape[1])
    
        prevt = 0  # Previous time step
        for t in range(steps):
            # Calculate activation of the reservoir
            self.net_h[:, t] = np.dot(self.Win.T, x[:, t])
    
            # Integrate recurrent input with previous timestep
            self.net_h[:, t] += np.dot(self.Wr, self.act_h[:, prevt])
            
            # Apply activation function
            self.act_h[:, t] = +np.tanh(0.9 * self.act_h[:, prevt] + self.net_h[:, t])
            self.act_h[:, t] = np.clip(self.act_h[:, t], -0.8, 0.8)
    
            # Calculate activation of the output layer
            self.net_hk[:, t] = np.dot(self.Wout.T, self.act_h[:, t])
            self.act_hk[:, t] = self.net_hk[:, t]  # No tanh nonlinearity
    
            prevt = t  # Update previous timestep
    
        # Teacher Forcing
        T = np.transpose(x[:, :steps])
    
        # Compute output weights (Wout_new)
        M = np.transpose(self.act_h[:, :steps])
        Wout_new = np.dot(np.linalg.pinv(M), T)  # Use pseudoinverse
            
        ### PLOT Original Signal ###
        plt.plot(T[:-2, 0])
        
        return T, Wout_new


    def predict(self, x, T, Wout_new, trainingSteps, testingSteps):
        cl_output = x[:, trainingSteps - 1]  # Begin with last training value
    
        prevt = trainingSteps
        for t in range(testingSteps):
            t = t + trainingSteps  # Update time step
            
            # Reset states before accumulation
            self.net_h[:, t] = 0  # Reset net input
            self.net_hk[:, t] = 0  # Reset output
            
            # Integrate feedback 
            self.net_h[:, t] += np.dot(self.Wfb.T, cl_output)
    
            # Integrate recurrent input from previous state
            self.net_h[:, t] += np.dot(self.Wr, self.act_h[:, prevt])
    
            # Apply activation function
            self.act_h[:, t] += np.tanh(0.9 * self.act_h[:, prevt] + self.net_h[:, t])
            self.act_h[:, t] = np.clip(self.act_h[:, t], -0.8, 0.8)
    
            # Compute accumulate output activation
            self.net_hk[:, t] += np.dot(Wout_new.T, self.act_h[:, t])
            self.act_hk[:, t] = self.net_hk[:, t]
    
            cl_output = self.act_hk[:, t] # Update feedback
            prevt = t # Update previous timestep
    
        # Calculate Normalized RMSE
        NRMSE = np.sqrt(np.mean((self.act_hk[:, trainingSteps:] - x[:, trainingSteps:]) ** 2) / np.var(x[:, trainingSteps:]))
    
        ### PLOT Learned Signal Echo ###
        x_values = np.arange(trainingSteps, trainingSteps + testingSteps)
        plt.plot(x_values, self.act_hk[0, trainingSteps:])
        plt.vlines(trainingSteps, -1.2, 1.2, colors='red')
        
        return self.act_hk, NRMSE
