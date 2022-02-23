# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 09:06:43 2022

@author: pedro
"""

#%% Import packages

import numpy as np
import matplotlib.pyplot as plt

from Lecture_5_Functions import cost_contour
from Lecture_5_Functions import cost_rosenbrock, grad_rosenbrock
from Lecture_5_Functions import grad_descent, line_search_grad_descent, conj_grad_method, BFGS

#%% Preamble: customization of matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

#%% Plot cost function

# Cost function map parameters
n_x = 200; n_y = 200; x_min=-2; x_max=2; y_min=-0.5; y_max=3
# Draw 2D cost function map
COST = cost_contour(n_x,n_y,x_min,x_max,y_min,y_max,cost_rosenbrock)

plt.figure()
# Cost function contour levels and filled map
plt.contour(COST,extent=[x_min,x_max,y_min,y_max],levels=np.logspace(0,3,5),colors='white',alpha=0.5,zorder=1)
plt.imshow(COST,extent=[x_min,x_max,y_min,y_max],origin='lower',cmap='viridis',zorder=0)
# Axes labels & formatting
plt.xlabel(r'$w_0$'); plt.ylabel(r'$w_1$')
plt.axis('scaled'); plt.tight_layout()

#%% Optimization inputs

# Initial guess
w_0 = [0,2.5]
# Learning rate guess
eta_0 = 0.00125
# Number of iterations for the gradient descent
n_iter = 10000
# Constant for the Strong Wolfe conditions (line-search)
c1 = 1e-6; c2 = 1.0
# Tolerance/precision in the gradient computation
eps = 1e-16

#%% Optimization methods

# Gradient descent with constant learning rate
Grad_J_1,J_1,w_1 = grad_descent(n_iter,w_0,eta_0,cost_rosenbrock,grad_rosenbrock)

# # Gradient descent with line-search
eta_2,Grad_J_2,J_2,w_2 = line_search_grad_descent(n_iter,w_0,c1,c2,eps,cost_rosenbrock,grad_rosenbrock)

# # Conjugate-gradients
eta_3,Grad_J_3,J_3,w_3 = conj_grad_method(n_iter,w_0,c1,c2,eps,cost_rosenbrock,grad_rosenbrock)

# # Quasi-Newton BFGS
eta_4,Grad_J_4,J_4,w_4 = BFGS(n_iter,w_0,c1,c2,eps,cost_rosenbrock,grad_rosenbrock)


#%% Plot cost function and optimization procedures

plt.figure()

# Gradient descent solution history
plt.plot(w_1[:,0],w_1[:,1],color='C0',linestyle='solid',linewidth=3,alpha=0.8,marker='',zorder=6)
plt.scatter(w_1[-1,0],w_1[-1,1],color='C0',marker='o',label=r'Grad. descent',zorder=6)

# # Gradient descent solution history
plt.plot(w_2[:,0],w_2[:,1],color='C1',linestyle='solid',linewidth=3,alpha=0.8,marker='',zorder=5)
plt.scatter(w_2[-1,0],w_2[-1,1],color='C1',marker='v',label=r'Grad. descent \& line-search',zorder=5)

# # Conjugate gradient method solution history
plt.plot(w_3[:,0],w_3[:,1],color='C2',linestyle='solid',linewidth=3,alpha=0.8,marker='',zorder=4)
plt.scatter(w_3[-1,0],w_3[-1,1],color='C2',marker='s',label=r'Conjugate gradients method',zorder=4)

# # Quasi-Newton BFGS method solution history
plt.plot(w_4[:,0],w_4[:,1],color='C3',linestyle='solid',linewidth=3,alpha=0.8,marker='',zorder=3)
plt.scatter(w_4[-1,0],w_4[-1,1],color='C3',marker='>',label=r'Quasi-Newton: BFGS',zorder=6)

# True minimizer w*
plt.scatter(1,1,color='white',marker='x',s=100,label=r'True minimizer: $\mathbf{w}^*$',alpha=0.8,zorder=2)
# Cost function contour levels and filled map
plt.contour(COST,extent=[x_min,x_max,y_min,y_max],levels=np.logspace(0,3,5),colors='white',alpha=0.5,zorder=1)
plt.imshow(COST,extent=[x_min,x_max,y_min,y_max],origin='lower',cmap='viridis',alpha=0.6,zorder=0)

# Axes labels & formatting
plt.xlabel(r'$w_0$'); plt.ylabel(r'$w_1$')
plt.legend(); plt.axis('scaled'); plt.tight_layout()

#%% Convergence analysis

plt.figure()
plt.subplot(2,1,1)
plt.plot(J_1,color='C0',label='Grad. descent')
plt.plot(J_2,color='C1',label='Grad. descent \& line-search')
plt.plot(J_3,color='C2',label='Conjugate gradient method')
plt.plot(J_4,color='C3',label='Quasi-Newton: BFGS')
plt.yscale('log')
plt.ylabel(r'Cost function: $J{(\mathbf{w})}$')
plt.xlabel('Number of iterations [-]'); plt.xlim(0,n_iter-1)
plt.grid(alpha=0.5); plt.legend(); plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(np.linalg.norm(Grad_J_1,axis=1),color='C0',label='Grad. descent')
plt.plot(np.linalg.norm(Grad_J_2,axis=1),color='C1',label='Grad. descent \& line-search')
plt.plot(np.linalg.norm(Grad_J_3,axis=1),color='C2',label='Conjugate gradient method')
plt.plot(np.linalg.norm(Grad_J_4,axis=1),color='C3',label='Quasi-Newton: BFGS')
plt.yscale('log')
plt.ylabel(r'Gradient norm: $||\nabla_w J{(\mathbf{w})}||$')
plt.xlabel('Number of iterations [-]'); plt.xlim(0,n_iter-1)
plt.grid(alpha=0.5); plt.tight_layout()
