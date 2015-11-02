# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 16:49:49 2015

utilities for standard kernel operations 

@author: Achille
"""

import numpy as np
from scipy.spatial.distance import pdist


def rbf_dot(X, Y, sigma):
    """
    computes Gram matrix K using a Gaussian kernel
    K_ij = exp(-||X[:,i]-Y[:,j]||^2/(2*sigma^2))
    
    arguments:
    X     --  nxd array
    Y     --  nxd array
    sigma --  kernel width
    
    returns:
    K     --  nxn Gram matrix     
    """
    
    if X.shape[1] != Y.shape[1] :
        raise ValueError('X and Y must have the same dimensions')
    
    nx, d = X.shape
    ny    = Y.shape[0]
    
    G = (X**2).sum(axis=1) 
    H = (Y**2).sum(axis=1)
    
    Q = np.tile(G[:,np.newaxis],(1,ny))
    R = np.tile(H, (nx,1))
    
    K = Q + R -2*np.dot(X, Y.T)
    
    K = np.exp(-K/(2*sigma**2))
    
    return K
 
def center_matrix(K):
    """
    computes the centered kernel matrix defined by Kc = H*K*H where
    H = I_n - (1/n)U_n where U_n is the 'all-ones' matrix
    
    arguments:
    K        -- Gram matrix
    
    returns:
    Kc       -- centered Gram matrix
    """
    
    n       = K.shape[0]
    colsumK = np.sum(K, axis=1)
    sumK    = np.sum(colsumK)
    
    return K - np.add.outer(colsumK, colsumK)/n - sumK/(n**2)
    
    
    
    
    
def estim_sigmakernel_median(X, nb_samples_max=300):
    """
    provides an estimate of the Gaussian kernel width parameter based on
    the median of the between-samples distance
    
    arguments:
    X               -- n x d array
    nb_samples_max  -- number of maximum samples used for the computation 
                        of the median
                        
    returns:
    sigma           -- Gaussian kernel scale factor
    
    """
    
    m = X.shape[0]
    
    if m > nb_samples_max:
        isub   = np.random.choice(m, nb_samples_max, replace=False)
        dist_X = pdist(X[isub,:])
    else:
        dist_X = pdist(X)
        
    sigma  = np.median(dist_X)
    
    return sigma
    
 
   