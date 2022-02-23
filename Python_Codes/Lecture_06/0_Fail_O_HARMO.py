# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 14:34:00 2021

@author: mendez
"""


# Test the function-like implementation of the GA
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import optimize # nice optimization tools
from scipy.optimize import minimize

#%% Function Definition 

def harmo(X):  # The Function to Minimize  
    """Function to be minimized"""  # new        
    C=X[0]*np.sin(4*X[0])+1.1*X[1]*np.sin(2*X[1])
    return C 



def Grad_Desc(func,X_0,N_ITER=50,eta=0.001):
    # Inputs: the function, the starting point, the number of iterations
    # and the learning rate.
    # Output: The minima found
    X_S=X_0; eps=1e-6
    X_S_PATH=np.zeros((len(X_0),N_ITER))
    print('Running Grad Descent on '+ str(func))
    for j in range(0,N_ITER):
     GRAD=optimize.approx_fprime(X_S, func,eps)
     X_S=X_S-eta*GRAD
     X_S_PATH[:,j]=X_S
     # Create Table for Output     
     print('j= '+str(j)+' Err='+str(func(X_S)))
    print('Solution X='+str(X_S))
    return X_S, X_S_PATH



#%% Main GA, given the input parameters here below

x = np.linspace(-8, 8, 200)
y = np.linspace(-8, 8, 200)
X, Y = np.meshgrid(x, y)
COST=Y*0
# Evaluate the cost function
for i in range(0,len(x)):
  for j in range(0,len(x)):
   XX=[X[i,j],Y[i,j]]   
   COST[i,j]=harmo(XX)                # ---------- PUT COST FUNCTION HERE
      

 
#%% Gradient Descent Attempts
X_0=np.array([0,2])
# Set of attempts: Gradient Descent, BFGS, NM
X_SS,X_S_Path=Grad_Desc(harmo,X_0,N_ITER=1000,eta=0.003)

res_BFGS = minimize(harmo, X_0, method='BFGS',
               options={'gtol': 1e-6, 'disp': True})
print('BFGS Result is '+str(res_BFGS.x))

res_NM = minimize(harmo, X_0, method='Nelder-Mead',
               options={'gtol': 1e-6, 'disp': True})
print('NM Result is '+str(res_NM.x))


# Find easily the objective:
obb=COST.min()
ID=np.where(COST == obb)

fig= plt.figure(figsize=(5, 5)) # This creates the figure
plt.contourf(X, Y, COST, extend='both', alpha=0.5) 
plt.plot(X[ID],Y[ID],'ro',markersize=5)
plt.plot(-X[ID],Y[ID],'ro',markersize=5)
plt.plot(-X[ID],-Y[ID],'ro',markersize=5)
plt.plot(X[ID],-Y[ID],'ro',markersize=5)
plt.plot(X_SS[0],X_SS[1],'wo',markersize=5,label='GD')
plt.plot(X_S_Path[0,:],X_S_Path[1,:],'k--')
plt.plot(res_BFGS.x[0],res_BFGS.x[1],"bo",markersize=5,label='BFGS')
plt.plot(res_NM.x[0],res_NM.x[1],'d',markersize=5,label='NM')
plt.xlim([-8, 8])
plt.ylim([-8, 8])  
plt.legend() 
plt.tight_layout()
plt.savefig('HARMO_GD_BFS_NM_ATTEMPS.png', dpi=100)
plt.show()



