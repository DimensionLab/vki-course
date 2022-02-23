# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 14:20:02 2022

@author: pedro
"""

#%% Import packages

import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.optimize  import minimize

from Lecture_5_Functions import cost_contour
from Lecture_5_Functions import num_response, theor_response, Error, Grad_Error
from Lecture_5_Functions import grad_descent, line_search_grad_descent, conj_grad_method, BFGS

#%% Preamble: customization of matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

#%% Input parameters: 2nd Order System

# Damping rate [-] and natural freq. [rad/s]
xi = 0.7; w_n = 1
# Initial conditions [y, dy/dt]
y0 = [0, 0]
# Number of time-steps [-], start-time [s], end-time [s]
n_t = 3000; t_s = 0; t_e = 30
# Temporal array [s]
ts = np.linspace(t_s, t_e, n_t)
# System response via numerical integration
ys_num   = odeint(num_response, y0, ts, args=(xi,w_n))[:,0]
# System response via analytical step response
ys_theor = theor_response(ts,xi,w_n)

plt.figure()
# Numerical solution
plt.plot(ts,ys_num,color='b',label='Numerical integration',marker='o',markerfacecolor='none',markevery =40)
# Theoretical solution
plt.plot(ts,ys_theor,color='r',label='Analytical response',marker='d',markerfacecolor='none',markevery =40)
# Input step with amplitude 1 at t=0
plt.plot(ts,np.heaviside(ts,1),color='k',label='Impulse')
# Axes labels & formatting
plt.ylabel(r'$y_s$'); plt.xlabel(r'Time [s]');
plt.xlim(ts[0],ts[-1]); plt.ylim(0); plt.grid(); plt.legend(); plt.tight_layout()

#%% Signal generation

# Select seed for random number generation
np.random.seed(0)
# Generate noisy synthetic data by adding a normally distributed random perturbation
y_data = ys_theor + np.random.normal(loc=0,scale=0.05,size=n_t)

plt.figure()
# Nosy synthetic data
plt.plot(ts,y_data,color='k',label='Noisy data')
# Theoretical solution
plt.plot(ts,ys_theor,color='r',label='Analytical response')
# Axes labels & formatting
plt.ylabel(r'$y_s$'); plt.xlabel(r'Time [s]');
plt.xlim(ts[0],ts[-1]); plt.ylim(0); plt.grid(); plt.legend(); plt.tight_layout()

#%% Optimization inputs

# Initial guess
w_0 = [1.5,1.5]
# Learning rate guess
eta_0 = 1e-3
# Number of iterations for the optimizers
n_iter = 2000
# Constant for the Strong Wolfe conditions (line-search)
c1 = 1e-6; c2 = 1.0
# Tolerance/precision in the gradient convergence
eps = 1e-16

#%% Optimization methods

# Gradient descent with constant learning rate
t_1 = time.time() # time at the start
Grad_J_1, J_1, w_1 = grad_descent(n_iter=n_iter,w_0=w_0,eta=eta_0,cost=Error,grad=Grad_Error,args=(y_data,num_response,y0,ts))
t_1 = time.time() - t_1 # time during execution

# Gradient descent with line-search
t_2 = time.time() # time at the start
eta_2,Grad_J_2,J_2,w_2 = line_search_grad_descent(n_iter=n_iter,w_0=w_0,c1=c1,c2=c2,eps=eps,cost=Error,grad=Grad_Error,args=(y_data,num_response,y0,ts))
t_2 = time.time() - t_2 # time during execution

# Conjugate-gradients
t_3 = time.time() # time at the start
eta_3,Grad_J_3,J_3,w_3 = conj_grad_method(n_iter=n_iter,w_0=w_0,c1=c1,c2=c2,eps=eps,cost=Error,grad=Grad_Error,args=(y_data,num_response,y0,ts))
t_3 = time.time() - t_3 # time during execution

# Quasi-Newton BFGS
t_4 = time.time() # time at the start
eta_4,Grad_J_4,J_4,w_4 = BFGS(n_iter=n_iter,w_0=w_0,c1=c1,c2=c2,eps=eps,cost=Error,grad=Grad_Error,args=(y_data,num_response,y0,ts))
t_4 = time.time() - t_4 # time during execution

#%%
plt.figure()

# plt.scatter(res.x[0],res.x[1],c='r',label=r'$(%.2f,%.2f)$' %(res.x[0],res.x[1]),zorder=2)

# Gradient descent solution history
plt.plot(w_1[:,0],w_1[:,1],color='C0',linestyle='solid',linewidth=3,alpha=0.8,marker='',zorder=6)
plt.scatter(w_1[-1,0],w_1[-1,1],color='C0',marker='o',label=r'Grad. descent',zorder=6)

# Gradient descent solution history
plt.plot(w_2[:,0],w_2[:,1],color='C1',linestyle='solid',linewidth=3,alpha=0.8,marker='',zorder=5)
plt.scatter(w_2[-1,0],w_2[-1,1],color='C1',marker='v',label=r'Grad. descent \& line-search',zorder=5)

# Conjugate gradient method solution history
plt.plot(w_3[:,0],w_3[:,1],color='C2',linestyle='solid',linewidth=3,alpha=0.8,marker='',zorder=6)
plt.scatter(w_3[-1,0],w_3[-1,1],color='C2',marker='s',label=r'Conjugate gradients method',zorder=6)

# Quasi-Newton BFGS method solution history
plt.plot(w_4[:,0],w_4[:,1],color='C3',linestyle='solid',linewidth=3,alpha=0.8,marker='',zorder=7)
plt.scatter(w_4[-1,0],w_4[-1,1],color='C3',marker='>',label=r'Quasi-Newton: BFGS',zorder=7)

# True minimizer w*
plt.scatter(xi,w_n,color='white',marker='x',s=100,label=r'Theoretical',alpha=0.8,zorder=2)
# Cost function map parameters
n_x = 50; n_y = 50; x_min=0; x_max=2; y_min=0; y_max=2
# Draw 2D cost function map
COST = cost_contour(n_x,n_y,x_min,x_max,y_min,y_max,Error,args=(y_data,num_response,y0,ts))
# Cost function contour levels and filled map
plt.contour(COST,extent=[x_min,x_max,y_min,y_max],levels=np.logspace(0,1.5,20),colors='white',alpha=0.5,zorder=1)
plt.imshow(COST,extent=[x_min,x_max,y_min,y_max],origin='lower',interpolation='nearest',aspect='auto',cmap='viridis',alpha=0.6,zorder=0)

# Axes labels & formatting
plt.xlabel(r'$w_0=\xi$'); plt.ylabel(r'$w_1=\omega_n$')
plt.legend(); plt.tight_layout()

#%% Convergence analysis

plt.figure()
plt.subplot(2,1,1)
plt.plot(J_1,color='C0',label='Grad. descent')
plt.plot(J_2,color='C1',label='Grad. descent \& line-search')
plt.plot(J_3,color='C2',label='Conjugate gradient method')
plt.plot(J_4,color='C3',label='Quasi-Newton: BFGS')
plt.xlabel('Number of iterations [-]'); plt.ylabel(r'Cost function: $J{(\mathbf{w})}$')
plt.xlim(0,n_iter-1); plt.ylim(min(J_1[-1],J_2[-1],J_3[-1],J_4[-1]))
plt.grid(alpha=0.5); plt.legend(); plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(np.linalg.norm(Grad_J_1,axis=1),color='C0',label='Grad. descent')
plt.plot(np.linalg.norm(Grad_J_2,axis=1),color='C1',label='Grad. descent \& line-search')
plt.plot(np.linalg.norm(Grad_J_3,axis=1),color='C2',label='Conjugate gradient method')
plt.plot(np.linalg.norm(Grad_J_4,axis=1),color='C3',label='Quasi-Newton: BFGS')
plt.yscale('log')
plt.xlabel('Number of iterations [-]'); plt.ylabel(r'Gradient norm: $||\nabla_w J{(\mathbf{w})}||$')
plt.xlim(0,n_iter-1); plt.ylim(min(np.linalg.norm(Grad_J_1,axis=1)[-1],
                                   np.linalg.norm(Grad_J_2,axis=1)[-1],
                                   np.linalg.norm(Grad_J_3,axis=1)[-1],
                                   np.linalg.norm(Grad_J_4,axis=1)[-1]))
plt.grid(alpha=0.5); plt.tight_layout()

#%% Learning rate analysis

plt.figure()
plt.plot(1e-2*np.ones(n_iter),color='C0',label='Grad. descent')
plt.plot(eta_2,color='C1',label='Grad. descent \& line-search')
plt.plot(eta_3,color='C2',label='Conjugate gradient method')
plt.plot(eta_4,color='C3',label='Quasi-Newton: BFGS')
plt.yscale('log')
plt.ylabel(r'Learning rate: $\eta^{(k)}$')
plt.xlabel('Number of iterations [-]'); plt.xlim(0,n_iter-1)
plt.grid(True, which="both", ls="-", alpha=0.4); plt.legend(); plt.tight_layout()

#%% Compare system response

plt.figure()
# Nosy synthetic data
plt.plot(ts,y_data,color='k',label='Noisy data')
# Response obtained from the optimization step
y_1 = odeint(num_response, y0, ts, args=(w_1[-1,0],w_1[-1,1]))[:,0]
y_2 = odeint(num_response, y0, ts, args=(w_2[-1,0],w_2[-1,1]))[:,0]
y_3 = odeint(num_response, y0, ts, args=(w_3[-1,0],w_3[-1,1]))[:,0]
y_4 = odeint(num_response, y0, ts, args=(w_4[-1,0],w_4[-1,1]))[:,0]

plt.plot(ts,y_1,color='C0',label='Grad. descent')
plt.plot(ts,y_2,color='C1',label='Grad. descent \& line-search')
plt.plot(ts,y_3,color='C2',label='Conjugate gradient method')
plt.plot(ts,y_4,color='C3',label='Quasi-Newton: BFGS')

# Axes labels & formatting
plt.ylabel(r'$y_s$'); plt.xlabel(r'Time [s]');
plt.xlim(ts[0],ts[-1]); plt.ylim(0); plt.grid(); plt.legend(); plt.tight_layout()

#%% Compare execution times

plt.figure()
plt.bar([1,2,3,4],[t_1,t_2,t_3,t_4],edgecolor='k',color=['C0','C1','C2', 'C3'])
plt.ylabel('Execution time [s]')
plt.xticks([1,2,3,4], ['Grad. Descent', 'GD + Line S.', 'Conj. Grad', 'BFGS'])
plt.tight_layout()

#%% Compare with SciPy functions

# SciPy BFGS
t_bfgs = time.time() # time at the start
res_bfgs = minimize(fun=Error,
                    x0=w_0,
                    jac=Grad_Error,
                    method='BFGS',
                    tol=eps,
                    args=(y_data,num_response,y0,ts)
                    )
t_bfgs = time.time() - t_bfgs # time during execution

# SciPy Conjugate Gradients Method
t_cg = time.time() # time at the start
res_cg = minimize(fun=Error,
                  x0=w_0,
                  jac=Grad_Error,
                  method='CG',
                  tol=eps,
                  args=(y_data,num_response,y0,ts)
                  )
t_cg = time.time() - t_cg # time during execution


#%%
plt.figure()
plt.bar([1,2,3,4],[t_3,t_cg,t_4,t_bfgs],edgecolor='k',color=['C2','C2','C3','C3'])
plt.ylabel('Execution time [s]')
plt.xticks([1,2,3,4], ['Conj. Grad', 'Conj. Grad (SciPy)', 'BFGS', 'BFGS (SciPy)'])
plt.tight_layout()
