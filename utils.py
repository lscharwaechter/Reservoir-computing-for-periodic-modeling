# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 21:55:05 2022

@author: Leon Scharwächter
"""

import numpy as np

def scaleMatrix(matrix, scalingFactor):
    '''
    Scales the entries of the input matrix 
    with the given scaling factor
    '''
    return matrix*scalingFactor

def sparseMaker(matrix, sparseness):
    '''
    Sets the proportion of entries given by sparsness 
    randomly to zero
    '''
    indices = np.random.choice(matrix.shape[0]*matrix.shape[1],
                               replace=False, 
                               size=int(matrix.shape[0]*matrix.shape[1]*sparseness))
    matrix[np.unravel_index(indices, matrix.shape)] = 0  
    return matrix

def getEchoStateProperty(matrix, spectralRadius):
    '''
    Implements the Echo State Property, i.e. divides the
    input matrix with its largest eigenvalue and scales it
    with a spectral radius < 1
    '''
    currentRadius = np.max(np.abs(np.linalg.eigvals(matrix)))
    if currentRadius == 0:
        currentRadius = 1e-6
    return matrix / currentRadius * spectralRadius
