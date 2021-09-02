#from NonlinearVAR import NonlinearVAR
import matplotlib.pyplot as plt
import numpy as np
import pickle
#from utils_exp import *

def load_model_from_pickle(filename): # LOADER
    infile = open(filename,'rb')
    new_obj = pickle.load(infile)
    infile.close()
    return new_obj

def topshow(m_A_in):
        m_A = np.linalg.norm(m_A_in, axis=2)
        plt.matshow(m_A)
        plt.xlabel('(past) Node index')
        plt.ylabel('(predicting) Node index')

folderName = 'data_1sep/'
nlv_hat  = load_model_from_pickle(folderName+'true_17.nlv')
nlv_true = load_model_from_pickle(folderName+'nlv_true_0')
lv_hat   = load_model_from_pickle(folderName+'hat0_linear')

topshow(nlv_true.A)
plt.title('True adjacency')
topshow(nlv_hat.A)
plt.title('Estimated adjacency (proposed method)')
topshow(lv_hat.A)
plt.title('Estimated adjacency (linear VAR)')


plt.show()