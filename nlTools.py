import numpy as np

def sigmoid(x): # stable sigmoid
    return np.where(x >= 0, 
        1 / (1 + np.exp(-x)), 
        np.exp(x) / (1 + np.exp(x)))

def sigprime(x):
    return sigmoid(x)*sigmoid(-x)
    
class NodalNonlinearity:
    def __init__(self, M, activation=sigmoid, activation_prime = sigprime):
        self.M = M
        self.alpha = np.zeros(M)
        self.k     = np.zeros(M)
        self.w     = np.zeros(M)
        self.b     = np.zeros(1)
        self.activation = activation
        self.activation_prime = activation_prime