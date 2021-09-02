import numpy as np
import pdb

class LinearVAR:
    def __init__(self, N, P):
            self.A = np.zeros([N, N, P])
