# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 18:24:40 2022

@author: mendez
"""

import numpy as np
import matplotlib.pyplot as plt

# Configuration for plots
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

# Create the vector for the range
x=np.linspace(-1,1,100)

#%% Sigmoid Basis function
def sigmoid(x,x_r=0,c_r=0.1):
    z=(x-x_r)/c_r;
    phi_r=1/(1+np.exp(-z))
    return phi_r

# Define some collocation points in [-1,1]
x_r=np.linspace(-1,1,5)

fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
for j in range(len(x_r)):
    plt.plot(x,sigmoid(x,x_r[j],c_r=0.05))

ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)
plt.title('Sigmoid Functions',fontsize=16)

Name='Sigmoid_Functions.png'
plt.tight_layout()
plt.savefig(Name, dpi=200) 


#%% Gaussian Basis function
def Gauss_RBF(x,x_r=0,c_r=0.1):
    d=x-x_r # Get distance
    phi_r=np.exp(-c_r**2*d**2)
    return phi_r

fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
for j in range(len(x_r)):
    plt.plot(x,Gauss_RBF(x,x_r[j],c_r=4))

ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)
Name='Gaussians_Functions.png'
plt.title('Gaussian RBFs',fontsize=16)
plt.tight_layout()
plt.savefig(Name, dpi=200) 


#%% Custom Basis function
def C4_Compact_RBF(x,x_r=0,c_r=0.1):
    d=x-x_r # Get distance
    phi_r=(1+d/c_r)**5*(1-d/c_r)**5
    phi_r[np.abs(d)>c_r]=0
    return phi_r

fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
for j in range(len(x_r)):
    plt.plot(x,C4_Compact_RBF(x,x_r[j],c_r=0.4))

ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)
Name='C4_RBF_Functions.png'
plt.title('C4 RBF',fontsize=16)
plt.tight_layout()
plt.savefig(Name, dpi=200) 

#%% Test Standardization

x_test=np.random.uniform(low=-5,high=4,size=100)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Create a scaler object and then fit
scaler = MinMaxScaler(); 
# or
scaler=StandardScaler()



# Note that this function is designed to work with (large)
# matrices. Even a vector should be defined as a matrix.
# We can reshape the data using .reshape(-1,1)
scaler.fit_transform(x_test.reshape(-1,1))
x_prime=scaler.transform(x_test.reshape(-1,1)) # Scale
x_back=scaler.inverse_transform(x_prime)[:,0] # Invert

#%% Test Scale on Matrix like data
SIZE=1000
X_feature=np.zeros((SIZE,3))
# Note that in scipy the feature matrix is n_samples x n_features
# so the scaler is here acting column by column

X_feature[:,0]=np.random.uniform(low=-5,high=4,size=SIZE)
X_feature[:,1]=np.random.uniform(low=-1,high=3,size=SIZE)
X_feature[:,2]=np.random.normal(loc=2,scale=2,size=SIZE)

scal_Matrix=StandardScaler()
scal_Matrix.fit(X_feature)
X_prime=scal_Matrix.transform(X_feature)


