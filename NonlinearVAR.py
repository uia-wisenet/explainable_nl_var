#import torch, 
import numpy as np
import matplotlib.pyplot as plt
import pickle
#from projection_simplex import projection_simplex_sort as proj_simplex
#from nlTools import NodalNonlinearity, nnl_randomInit

import pdb
#from utils_compare import kevin_computations # for the testing only

def _repair_zd(z_in, N):
    if z_in is None:
        return np.nan(N)
    assert z_in.shape == (N,)
    return z_in

class NonlinearVAR:
    def __init__(self, N, M, P, filename_prefix = 'model.nlv', \
        zl_desired = None, zu_desired = None, activation_type = 'sigmoid'):
        
        self.A = np.zeros([N, N, P])
        self.zl_desired = _repair_zd(zl_desired, N)
        self.zu_desired = _repair_zd(zu_desired, N)
        self.nnl = [nnl_randomInit(M, zl_desired[n], zu_desired[n]\
            , activation_type=activation_type) for n in range(N)] 
        # TODO: different initializations 
        #self.nnl = [NodalNonlinearity(M) for n in range(N)]
        self.filename_prefix = filename_prefix     
     
    def topshow(self):
        m_A = np.linalg.norm(self.A, axis=2)
        plt.matshow(m_A)
        plt.xlabel('(past) Node index')
        plt.ylabel('(predicting) Node index')

  