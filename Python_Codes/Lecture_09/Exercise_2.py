# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 19:53:02 2022

@author: mendez
"""

import numpy as np

import matplotlib.pyplot as plt

# Configuration for plots
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

#%% We seek to solve a boundary value problem in a meshless fashion
# Generate the 'domain'
n_x=500; n_b=200
x=np.linspace(0,1,n_x)
x_b=np.linspace(0,1,n_b)


#%% Step 1: Create the matrix L, Phi_Gamma, c
# We need Phi, Phi_x and Phi_xx
def C4_RBF(x,x_r=0,c_r=0.1):
    d=x-x_r # Get distance
    phi_r=(1+d/c_r)**5*(1-d/c_r)**5
    phi_r[np.abs(d)>c_r]=0
    return phi_r

def C4_RBF_x(x,x_r=0,c_r=0.1):
    d=x-x_r # Get distance
    phi_r=(-10*d*(d-c_r)**4*(d+c_r)**4)/(c_r**10)
    phi_r[np.abs(d)>c_r]=0
    return phi_r

def C4_RBF_xx(x,x_r=0,c_r=0.1):
    d=x-x_r # Get distance
    phi_r=(-10*(d-c_r)**3*(d+c_r)**3)*(3*d-c_r)*(3*d+c_r)/(c_r**10)
    phi_r[np.abs(d)>c_r]=0
    return phi_r


phi=C4_RBF(x,0.5,c_r=0.05)
phi_x_a=C4_RBF_x(x,0.5,c_r=0.05)
phi_xx_a=C4_RBF_xx(x,0.5,c_r=0.05)

# import matplotlib.pyplot as plt

# # If you want to check the derivatives numerically:
# phi_x_n=np.gradient(phi,x)
# phi_xx_n=np.gradient(phi_x_a,x)

# plt.plot(phi_x_a)
# plt.plot(phi_x_n)

# plt.plot(phi_xx_a)
# plt.plot(phi_xx_n)

# Function that prepares the three derivatives
def PHI_C4_X(x_in, x_b, c_r=0.1):
 n_x=np.size(x_in); n_b=len(x_b)
 Phi=np.zeros((n_x,n_b+1)) # Initialize Basis Matrix on x
 Phi_x=np.zeros((n_x,n_b+1)) # Initialize Basis Matrix on x
 Phi_xx=np.zeros((n_x,n_b+1)) # Initialize Basis Matrix on x
  
 # Add a constant and a linear term on Phi, which means:
 Phi[:,0]=x_in; Phi[:,0]=np.ones(len(x_in)); Phi[:,0]=np.zeros(len(x_in)) 
 for j in range(0,n_b): # Loop to prepare the basis matrices (inefficient)
  # Prepare all the terms for Phi, Phi_x, Phi_xx
  Phi[:,j+1]=C4_RBF(x_in,x_r=x_b[j],c_r=c_r)
  Phi_x[:,j+1]=C4_RBF_x(x_in,x_r=x_b[j],c_r=c_r)
  Phi_xx[:,j+1]=C4_RBF_xx(x_in,x_r=x_b[j],c_r=c_r)
 
 return Phi, Phi_x, Phi_xx


Phi, Phi_x, Phi_xx=PHI_C4_X(x, x_b, c_r=0.1)


fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure

plt.plot(x,Phi_x[:,150],'b',label='dydx')
plt.plot(x,Phi_xx[:,100],'r',label='d2ydx2')

ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)
plt.title('RBFs derivatives',fontsize=16)

plt.legend()

Name='RBF_Derivatives.png'
plt.tight_layout()
plt.savefig(Name, dpi=200) 



#%% RBF Solution procedure
Phi, Phi_x, Phi_xx=PHI_C4_X(x, x_b, c_r=0.1)
# Create the matrix L and the bloc A
L=Phi_xx+2*Phi_x+Phi; A=2*L.T@L
# Create the matrix Gamma_T
x_gamma=np.array([0,1])
Phi_Gamma, Phi_x_Gamma, Phi_xx_Gamma=PHI_C4_X(x_gamma, x_b, c_r=0.1)
# Assembly the global system
Rows_H_1=np.hstack((A,Phi_Gamma.T))
Rows_H_2=np.hstack((Phi_Gamma,np.zeros((2,2))))
A_star=np.vstack((Rows_H_1,Rows_H_2))
b_star=np.hstack((np.zeros(n_b+1),np.array([1,3]))).T  
# Solve the system approximately using the Pseudo Inverse
x_sol=np.linalg.pinv(A_star,rcond=1e-15).dot(b_star)

# Other methods to solve linear systems
# x_sol= np.linalg.lstsq(A_star, b_star,rcond=None)
# x_sol=np.linalg.solve(A_star,b_star)

# Get the norm of b for relative error:
r_error=np.linalg.norm(A_star.dot(x_sol)-b_star)
print(r_error)

# Get the weights
w=x_sol[:n_b+1]
# Assembly the solution
y_Sol_RBF=Phi.dot(w)
# Prepare the Analytic solution for comparison
y_Sol_Analyt=np.exp(-x)+(3*np.e-1)*x*np.exp(-x)

# This creates the figure
fig, ax = plt.subplots(figsize=(5, 3)) 
plt.plot(x,y_Sol_RBF,'b',label='RBF Solution')
plt.plot(x,y_Sol_Analyt,'r',label='Analytic Solution')
ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)
plt.title('Analytic vs RBF Solution',fontsize=16)
Name='RBF_Approximation.png'
plt.tight_layout()
plt.savefig(Name, dpi=200) 









