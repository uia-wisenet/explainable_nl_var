import numpy as np
import matplotlib.pyplot as plt

folderName = 'results'
def myload(str_in):
     return np.loadtxt(folderName+'/'+str_in)

nmse_train   = myload('nmse_train')
nmse_test    = myload('nmse_test')
nmse_Ltrain  = myload('nmse_Ltrain')
nmse_Ltest   = myload('nmse_Ltest')
nmse_LNtrain = myload('nmse_LNtrain')
nmse_LNtest  = myload('nmse_LNtest')

#Plotting error metrics
plt.figure(1)
plt.plot( nmse_train  ,label='Nonlinear. Training NMSE')
plt.plot( nmse_test   ,label='Nonlinear. Test NMSE')
plt.plot( nmse_Ltrain ,\
     label='Linear. Training NMSE')
plt.plot( nmse_Ltest  ,\
     label='Linear. Test NMSE')
#plt.plot( nmse_LNtrain ,\
     #label='Normalization,linear. Training NMSE')
#plt.plot( nmse_LNtest  ,\
     #label='Normalization,linear. Test NMSE')

plt.grid()
plt.legend()
plt.title('Training and test NMSE')
plt.xlabel('epoch')
plt.ylabel('NMSE')

# plotting functions
#compare_transfers(nlv_true, nlv_hat)
plt.show()