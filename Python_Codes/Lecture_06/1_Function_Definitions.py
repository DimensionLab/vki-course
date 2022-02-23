# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 15:03:41 2022

@author: mendez
"""

import numpy as np
import matplotlib.pyplot as plt


#%% Preamble: customization of matplotlib
# Configuration for plots
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

#%% Harmonic function definition
def harmo(X):  # The Function to Minimize  
    """Function to be minimized"""  # new        
    C=X[0]*np.sin(4*X[0])+1.1*X[1]*np.sin(2*X[1])
    return C 


def Plot_Harmo(x_1m,x_1M,x_2m,x_2M,n_p):
  # Create the vectors for the grid  
  x = np.linspace(x_1m, x_1M, n_p)
  y = np.linspace(x_2m, x_2M, n_p)
  X, Y = np.meshgrid(x, y) # Create grid
  COST=np.zeros((n_p,n_p)) # Initialize the cost func
  # Evaluate the cost function
  for i in range(0,len(x)):
   for j in range(0,len(x)):
    XX=np.array([X[i,j],Y[i,j]]) # Get Point Loc   
    COST[i,j]=harmo(XX) # Interrogate the func   
  return X,Y, COST 

#%% Rosenbrock function definition
def rosenbrock(X):  # The Function to Minimize  
    """Function to be minimized"""  # new        
    x = X[0]; y = X[1]
    C= 100*(y-x**2)**2+(1-x)**2.   
    return C 


def Plot_Rosenbrock(x_1m,x_1M,x_2m,x_2M,n_p):
  # Create the vectors for the grid  
  x = np.linspace(x_1m, x_1M, n_p)
  y = np.linspace(x_2m, x_2M, n_p)
  X, Y = np.meshgrid(x, y) # Create grid
  COST=np.zeros((n_p,n_p)) # Initialize the cost func
  # Evaluate the cost function
  for i in range(0,len(x)):
   for j in range(0,len(x)):
    XX=np.array([X[i,j],Y[i,j]]) # Get Point Loc   
    COST[i,j]=rosenbrock(XX) # Interrogate the func   
  return X,Y, COST 



#%% Plot Harmonic function in 2D and in 3D

# Prepare the grid information
X_h,Y_h,COST_h=Plot_Harmo(-8,8,-8,8,200)

from matplotlib import cm
#Look at the Cost Function in 3D
fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(3, 3))
# Plot the surface.
surf = ax.plot_surface(X_h, Y_h, COST_h, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.dist = 12

plt.tight_layout()
plt.savefig('HARMO_3D_Cost_View.png', dpi=300)
plt.close("all")

#% Plot the fame function in 2D  

# Find easily the objective:
obb=COST_h.min()
ID=np.where(COST_h == obb)


fig, ax = plt.subplots(figsize=(3, 3)) # This creates the figure
plt.contourf(X_h, Y_h, COST_h,cmap=cm.coolwarm, alpha=0.8) 
plt.plot(X_h[ID],Y_h[ID],'wo',markersize=5)
plt.plot(-X_h[ID],Y_h[ID],'wo',markersize=5)
plt.plot(-X_h[ID],-Y_h[ID],'wo',markersize=5)
plt.plot(X_h[ID],-Y_h[ID],'wo',markersize=5)
plt.xlim([-8, 8])
plt.ylim([-8, 8])   
plt.xticks(np.arange(-8,10,4))
plt.yticks(np.arange(-8,10,4))

plt.tight_layout()
plt.savefig('HARMO_2D_Cost_View.png', dpi=300)

#%% Plot Rosenbrock in 2D and 3D
X_r,Y_r,COST_r=Plot_Rosenbrock(-2,2,-0.5,3,200)

#Look at the Cost Function in 3D
fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(4, 4))
# Plot the surface.
surf = ax.plot_surface(X_r, Y_r, COST_r, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.view_init(elev=38, azim=50)
ax.dist = 12

plt.tight_layout()
plt.savefig('ROSEN_3D_Cost_View.png', dpi=300)
plt.close("all")


#% Plot the fame function in 2D  

# Find easily the objective:
obb=COST_r.min()
ID=np.where(COST_r == obb)


fig, ax = plt.subplots(figsize=(3, 3)) # This creates the figure
plt.contourf(X_r, Y_r, COST_r,cmap=cm.coolwarm, alpha=0.8) 
plt.plot(X_r[ID],Y_r[ID],'wo',markersize=5)
plt.xlim([-2, 2])
plt.ylim([-0.5, 3])   
plt.xticks(np.arange(-2,2.5,1))
plt.yticks(np.arange(-0.5,3.5,1))

plt.tight_layout()
plt.savefig('ROSE_2D_Cost_View.png', dpi=300)



