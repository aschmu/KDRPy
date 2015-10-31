# -*- coding: utf-8 -*-
"""

Kernel Dimension Reduction (KDR) routines

see "Kernel dimension reduction in regression" by K.Fukumizu, Bach F.
 and M.I. Jordan, Annals of Statistics, 2009

@author: Achille
"""

import numpy as np
from scipy import linalg
from kernelfun import rbf_dot, estim_sigmakernel_median
from kernelderiv import kernel_derivative, KDR_linesearch
from sklearn import preprocessing
from matplotlib import pyplot as plt


def kdr_optim(X, Y, K, max_loop, sigma_x, sigma_y, eps,
              eta, anl, verbose = True, tol=1e-9, 
              init_deriv = False):
    
        n, d  = X.shape
        
        if n != Y.shape[0]:
            raise ValueError, 'X and Y have incompatible dimensions'
         
        assert K<=d, 'dimension K must be lower than d !'
        assert sigma_x > 0 and sigma_y > 0, 'scale parameters must be positive!'
        assert tol > 0, 'tolerance factor must be >0'
        
        if init_deriv:
            print 'Initialization by derivative method...\n'
            B, t = kernel_derivative(X, Y, K, np.sqrt(anl)*sigma_x,
                                     sigma_y, eps)
        else:            
            B = np.random.randn(d, K)
        
        B = linalg.svd(B, full_matrices=False)[0]
                
        unit = np.ones((n,n))
        Q = np.eye(n) - unit/n
        
        """Gram matrix of Y"""
        Gy  = rbf_dot(Y, Y, sigma_y) #check this 
        Kyo = np.dot(np.dot(Q, Gy), Q)
        Kyo  = (Kyo + Kyo.T)/2
        
        """objective function initial value """
        Z = np.dot(X, B)
#        nZ = Z/np.sqrt(2)/sigma_x1
        Gz = rbf_dot(Z, Z, sigma_x)
        Kz = np.dot(np.dot(Q, Gz), Q)
        Kz = (Kz + Kz.T)/2
        
        mz = linalg.inv(Kz + eps*n*np.eye(n))
        tr = np.sum(Kyo*mz)
        
        if verbose:
            print '[0] trace = ', tr
        
        ssz2 = 2*sigma_x**2
        ssy2 = 2*sigma_y**2
        #careful h from 0 to maxloop-1, implement accordingly
        for h in xrange(max_loop): 
            sz2 = ssz2+(anl-1)*ssz2*(max_loop-h-1)/max_loop
            sy2 = ssy2+(anl-1)*ssy2*(max_loop-h-1)/max_loop
            
            Z  = np.dot(X, B)
            Kzw = rbf_dot(Z, Z, np.sqrt(sz2))
            Kz  = np.dot(np.dot(Q, Kzw), Q)
            Kzi = linalg.inv(Kz + eps*n*np.eye(n)) #
            
            Ky = rbf_dot(Y, Y, np.sqrt(sy2))
            Ky = np.dot(np.dot(Q, Ky), Q)
            Ky = (Ky + Ky.T)/2
            Kyzi = np.dot(Ky, Kzi) #inutile de le déclarer
            
            dB = np.zeros((d,K))
            #KziKyzi = np.dot(Kzi, np.dot(Ky, Kzi))
            
            for a in xrange(d):
                Xa = np.tile(X[:,a][:,np.newaxis], (1, n))
                XX = Xa - Xa.T
                for b in xrange(K):
                    Zb = np.tile(Z[:,b][:,np.newaxis], (1, n))
                    tt = XX*(Zb - Zb.T)*Kzw
                    dKB = np.dot(Q, np.dot(tt, Q))
                    dB[a, b] = np.trace(np.dot(Kzi.dot(Kyzi),dKB))  #np.sum(KziKyzi*dKB.T)
            
            nm = linalg.norm(dB, 2)
            if nm < tol:
                break
            B, tr = KDR_linesearch(X, Ky, sz2, B, dB/nm, eta, eps)
            B = linalg.svd(B, full_matrices=False)[0]
           
            """ compute trace with unannealed parameter"""
            if verbose:
                Z = np.dot(X, B)
                Kz = rbf_dot(Z, Z, sigma_x)
                Kz = np.dot(np.dot(Q, Kz), Q)
                Kz = (Kz + Kz.T)/2
                mz = linalg.inv(Kz + n*eps*np.eye(n))
                tr = np.sum(Kyo*mz)
                print '[%d] trace = %.6f \n'  % (h,tr) 
        
        return B
        
        
if __name__ == "__main__":
    
    max_iter = 50
    epsilon = 1e-4
    eta_linesearch = 10.0
    verbose = True
    annealing = 4
    r = 2 #dimension of SDR subspace
    
    
    print 'KDR demo using wine data from UCI Repository \n'
    
    data = np.genfromtxt(fname='./data/wine_data.csv', delimiter=",")    
    y = data[:, 0][:,np.newaxis]
    X = data[:, 1:]
    d = X.shape[1]
    N = X.shape[0]
    
    print d, 'features\n'
    print N, 'samples\n'

    #standardize data
    std_scaler = preprocessing.StandardScaler().fit(X)    
    Xscaled = std_scaler.transform(X)
    
    #estimate Gaussian scale parameter
    sigma_X = 0.5*estim_sigmakernel_median(Xscaled)
    sigma_y = estim_sigmakernel_median(y)
    
    B = kdr_optim(X=Xscaled, Y=y, K=r, max_loop=max_iter, sigma_x=sigma_X*np.sqrt(np.float(r)/d), 
                  sigma_y=sigma_y, eps=epsilon, eta=eta_linesearch, 
                  anl=annealing, init_deriv=True)
        
    r = 2 #SDR subspace dimension
    l = 3 #nb classes in data
    
    Z = np.dot(Xscaled, B)
    plt.scatter(Z[np.ravel(y)==1,0], Z[np.ravel(y)==1,1], color="blue")
    plt.scatter(Z[np.ravel(y)==2,0], Z[np.ravel(y)==2,1], color="red")
    plt.scatter(Z[np.ravel(y)==3,0], Z[np.ravel(y)==3,1], color="green")

    
    #fig = plt.figure()
    