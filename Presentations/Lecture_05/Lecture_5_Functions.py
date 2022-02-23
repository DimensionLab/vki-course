# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 09:06:43 2022

@author: pedro
"""
#%% Import packages

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import line_search, approx_fprime

#%% Analytical cost functions and gradients

def cost_rosenbrock(w):
    J = (1 - w[0])**2 + 100*(w[1] - w[0]**2)**2
    return J

def grad_rosenbrock(w):
    dJdw1 = -2*(1 - w[0]) - 400*(w[1] - w[0]**2)
    dJdw2 = 200*(w[1] - w[0]**2)
    return np.array([dJdw1, dJdw2])

# grad_rosenbrock_n
# -> finite difference approximation
# -> compare with the theoretical one

def hess_rosenbrock(w):
    H11 = 2 -400*(w[1] - w[0]**2) + 800*w[0]**2
    H12 = -400*w[0]
    H21 = -400*w[0]
    H22 = 200
    return np.array([[H11, H12],[H21, H22]])

#%% Gradient-based methods

#%%% Vanilla gradient descent
def grad_descent(n_iter,w_0,eta,cost,grad,args=()):
    print('-- Grad. descent --')
    # Initialize solution history
    w = np.zeros((int(n_iter),len(w_0)))
    # Initialize cost function history
    J = np.zeros(int(n_iter)); J[0] = cost(w_0,*args)
    # Initialize gradient history
    Grad_J = np.zeros((int(n_iter),len(w_0))); Grad_J[0,:] = grad(w_0,*args)
    # Assign initial guess
    w[0,:] = w_0
    for i in range(int(n_iter-1)):
        # Update the solution
        w[i+1,:]      = w[i,:] - eta*Grad_J[i,:]
        J[i+1]        = cost(w[i+1,:],*args)
        Grad_J[i+1,:] = grad(w[i+1,:],*args)
        # Print status to the terminal
        print('Iteration: {:d}; Cost: {:.3f}; Grad_abs: {:.3f}'
              .format(i,J[i],np.linalg.norm(Grad_J[i+1,:])))
    print('Grad. descent terminated')
    # Return cost-function & solution history
    return Grad_J, J, w


#%%% Line-search gradient descent
def line_search_grad_descent(n_iter,w_0,c1,c2,eps,cost,grad,args=()):
    print('-- Grad. descent with line search --')
    # Initialize solution history
    w = np.zeros((int(n_iter),len(w_0))); w[0,:] = w_0
    # Initialize optimal learning rate history
    eta = np.zeros(int(n_iter))
    # Initialize cost function history
    J = np.zeros(int(n_iter)); J[0] = cost(w_0,*args); print(J[0])
    # Initialize gradient history
    Grad_J = np.zeros((int(n_iter),len(w_0))); Grad_J[0,:] = grad(w_0,*args); print(Grad_J[0,:])
    
    for i in range(int(n_iter-1)):
        
        # Line-search algorithm for the optimal learning rate
        eta[i],_,_,_,_,_=line_search(f=cost,
                                     myfprime=grad,
                                     xk=w[i,:],
                                     pk=-Grad_J[i,:],
                                     gfk=Grad_J[i,:],
                                     old_fval=J[i],
                                     c1=c1,
                                     c2=c2,
                                     maxiter=10000,
                                     args=args
                                     )

        # Update the solution
        w[i+1,:]      = w[i,:] - eta[i]*Grad_J[i,:]
        J[i+1]        = cost(w[i+1,:],*args)
        Grad_J[i+1,:] = grad(w[i+1,:],*args)
        
        if (np.isnan(w[i+1,:]).any()      == True or \
            np.isnan(J[i+1]).any()        == True or \
            np.isnan(Grad_J[i+1,:]).any() == True):
            # Must set an early termination condition because the Hessian
            # approximation may give a divide-by-zero warning. This means
            # the solution has already converged, so there is no need to
            # continue the algorithm
            print('Grad. descent + line search terminated: Minimum has been reached')
            eta    = eta[0:i+1]
            Grad_J = Grad_J[0:i+1,:]
            J      = J[0:i+1]
            w      = w[0:i+1]
            break
        elif np.linalg.norm(Grad_J[i+1,:])<=eps:
            print('Grad. descent + line search terminated: Gradient norm is under the tolerance')
            eta    = eta[0:i+2]
            Grad_J = Grad_J[0:i+2,:]
            J      = J[0:i+2]
            w      = w[0:i+2]
            break
        
        # Print status to the terminal
        print('Iteration: {:d}; Cost: {:.3f}; Grad_abs: {:.3f}'.format(i,J[i+1],np.linalg.norm(Grad_J[i+1,:])))
        
    # Return learning-rate, cost-function & solution history
    return eta, Grad_J, J, w


#%%% Conjugate gradient descent method
def conj_grad_method(n_iter,w_0,c1,c2,eps,cost,grad,args=()):
    print('-- Conjugate gradient descent --')
    # Initialize solution history
    w = np.zeros((int(n_iter),len(w_0))); w[0,:] = w_0
    # Initialize optimal learning rate history
    eta = np.zeros(int(n_iter))
    # Initialize cost function history
    J = np.zeros(int(n_iter)); J[0] = cost(w_0,*args)
    # Initialize gradient history
    Grad_J = np.zeros((int(n_iter),len(w_0))); Grad_J[0,:] = grad(w_0,*args)
    # Initialize search direction history
    d = np.zeros((int(n_iter),len(w_0))); d[0,:] = -Grad_J[0,:]
    
    for i in range(int(n_iter-1)):

        # Line-search algorithm for the optimal learning rate
        eta[i],_,_,_,_,_=line_search(f=cost,
                                     myfprime=grad,
                                     xk=w[i,:],
                                     pk=d[i,:],
                                     gfk=Grad_J[i,:],
                                     old_fval=J[i],
                                     c1=c1,
                                     c2=c2,
                                     maxiter=10000,
                                     args=args
                                     )
        
        # Update the solution
        w[i+1,:]      = w[i,:] + eta[i]*d[i,:]
        J[i+1]        = cost(w[i+1,:],*args)
        Grad_J[i+1,:] = grad(w[i+1,:],*args)        
        
        if (np.isnan(w[i+1,:]).any()      == True or \
            np.isnan(J[i+1]).any()        == True or \
            np.isnan(Grad_J[i+1,:]).any() == True):
            # Must set an early termination condition because the Hessian
            # approximation may give a divide-by-zero warning. This means
            # the solution has already converged, so there is no need to
            # continue the algorithm
            print('Conj. gradient descent terminated: Minimum has been reached')
            eta    = eta[0:i+1]
            Grad_J = Grad_J[0:i+1,:]
            J      = J[0:i+1]
            w      = w[0:i+1]
            break
        elif np.linalg.norm(Grad_J[i+1,:])<=eps:
            print('Conj. gradient descent terminated: Gradient norm is under the tolerance')
            eta    = eta[0:i+2]
            Grad_J = Grad_J[0:i+2,:]
            J      = J[0:i+2]
            w      = w[0:i+2]
            break
        
        # # Restart coefficient (p.125 Nocedal)
        # nu = 0.1
        # if i > 0:
        #     # Check if we need to restart the search direction based on
        #     # the orthogonality of the gradients between the current and
        #     # previous iterations
        #     ortho_check = np.abs(np.dot(Grad_J[i,:].T,Grad_J[i-1,:]))/np.abs(np.dot(Grad_J[i,:].T,Grad_J[i,:]))
            
        #     if ortho_check >= nu:
        #         beta = 0
        #     else:
        #         # Compute "beta" (Fletcher Reeves)
        #         beta = np.dot(Grad_J[i+1].T,Grad_J[i+1])/np.dot(Grad_J[i,:].T,Grad_J[i,:])
        # else:
        #     # Assume steepest descent for the first iteration
        #     beta = 0
        
        # Compute "beta" (Fletcher Reeves)
        beta = np.dot(Grad_J[i+1].T,Grad_J[i+1])/np.dot(Grad_J[i,:].T,Grad_J[i,:])
        
        # Compute new search direction
        d[i+1,:] = - Grad_J[i+1] + beta*d[i,:]
        
        # Print status to the terminal
        print('Iteration: {:d}; Cost: {:.3f}; Grad_abs: {:.3f}'.format(i,J[i+1],np.linalg.norm(Grad_J[i+1,:])))
    
    # Return cost-function & solution history
    return eta, Grad_J, J, w


#%%% Quasi-Newton: BFGS
def BFGS(n_iter,w_0,c1,c2,eps,cost,grad,args=()):
    print('-- Quasi-Newton Method: BFGS --')
    # Initialize solution history
    w = np.zeros((int(n_iter),len(w_0))); w[0,:] = w_0
    # Initialize optimal learning rate history
    eta = np.zeros(int(n_iter))
    # Initialize cost function history
    J = np.zeros(int(n_iter)); J[0] = cost(w_0,*args)
    # Initialize gradient history
    Grad_J = np.zeros((int(n_iter),len(w_0))); Grad_J[0,:] = grad(w_0,*args)
    # Initialize Hessian approximation as the identity matrix
    H = np.eye(len(w_0)); I = np.eye(len(w_0))
    # Initialize search direction history
    d = np.zeros((int(n_iter),len(w_0))); d[0,:] = -np.dot(H,Grad_J[0,:].T)
    
    for i in range(int(n_iter-1)):
        # Line-search algorithm for the optimal learning rate
        eta[i],_,_,_,_,_=line_search(cost,
                                     grad,
                                     w[i,:],
                                     d[i,:],                                       
                                     gfk=Grad_J[i,:],
                                     old_fval=J[i],
                                     c1=c1,
                                     c2=c2,
                                     maxiter=10000,
                                     args=args
                                     )

        # Update the solution
        w[i+1,:]      = w[i,:] + eta[i]*d[i,:]
        J[i+1]        = cost(w[i+1,:],*args)
        Grad_J[i+1,:] = grad(w[i+1,:],*args)
        
        if (np.isnan(w[i+1,:]).any()      == True or \
            np.isnan(J[i+1]).any()        == True or \
            np.isnan(Grad_J[i+1,:]).any() == True):
            # Must set an early termination condition because the Hessian
            # approximation may give a divide-by-zero warning. This means
            # the solution has already converged, so there is no need to
            # continue the algorithm
            print('BFGS terminated: Minimum has been reached')
            eta    = eta[0:i+1]
            Grad_J = Grad_J[0:i+1,:]
            J      = J[0:i+1]
            w      = w[0:i+1]
            break
        elif np.linalg.norm(Grad_J[i+1,:])<=eps:
            print('BFGS terminated: Gradient norm is under the tolerance')
            eta    = eta[0:i+2]
            Grad_J = Grad_J[0:i+2,:]
            J      = J[0:i+2]
            w      = w[0:i+2]
            break
        
        # Compute inverse-Hessian approximation
        s_k   =      w[i+1,:] -      w[i,:]; s_k=s_k.reshape((2,1)) # [nf x 1]
        y_k   = Grad_J[i+1,:] - Grad_J[i,:]; y_k=y_k.reshape((2,1)) # [nf x 1]
        rho_k = 1/np.dot(y_k.T,s_k) # [1 x 1]
        # BFGS construction of the Hessian approximation\
        H_aux_1 = I - rho_k*np.dot(s_k,y_k.T)
        H_aux_2 = I - rho_k*np.dot(y_k,s_k.T)
        H_aux_3 = rho_k*np.dot(s_k,s_k.T)
        H = np.dot(np.dot(H_aux_1,H),H_aux_2) + H_aux_3
        
        # Compute new search direction
        d[i+1,:] = - np.dot(H,Grad_J[i+1,:].T)
        
        # Print status to the terminal
        print('Iteration: {:d}; Cost: {:.3f}; Grad_abs: {:.3e}'.format(i,J[i+1],np.linalg.norm(Grad_J[i+1,:])))
    
    # Return cost-function & solution history
    return eta, Grad_J, J, w

#%% Second order system

def num_response(x,t,xi,w_n):
    # 2nd order system: step response (via numerical integration)
    y = x[0]
    z = x[1]
    dydt = z
    dzdt = -2*xi*w_n*z - w_n*w_n*y + w_n*w_n*np.heaviside(t,1)
    return [dydt,dzdt]

def theor_response(t,xi,w_n):
    # 2nd order system: step response (analytical response)
    if xi < 1:
        y_s = 1 - np.exp(-xi*w_n*t)*( np.cos(np.sqrt(1 - xi**2)*w_n*t) + (xi/np.sqrt(1 - xi**2))*np.sin(np.sqrt(1-xi**2)*w_n*t))
    elif xi > 1:
        y_s = 1 - np.exp(-xi*w_n*t)*( np.cosh(np.sqrt(xi**2 - 1)*w_n*t) + (xi/np.sqrt(xi**2 -1))*np.sinh(np.sqrt(xi**2 -1)*w_n*t))
    elif xi == 1:
        y_s = 1 - np.exp(-xi*w_n*t)*(1 + w_n*t)
    return y_s


def Error(w,y_s,num_response,y0,ts):
    xi = w[0]; w_n = w[1]
    # Evaluate numerical solution for w[0] = xi & w[1] = w_n
    y_num = odeint(num_response, y0, ts, args=(xi,w_n))[:,0]
    # L2 norm of the error between the synthetic data and the numerical solution
    Err = np.linalg.norm(y_s - y_num)
    return Err

def Grad_Error(w,y_s,num_response,y0,ts,dxi=1e-4,dwn=1e-4):
    xi = w[0]; w_n = w[1]
    # Evaluate numerical solution for w[0] = xi & w[1] = w_n
    y             = odeint(num_response, y0, ts, args=(xi    ,w_n    ))[:,0]
    y_xi_step_pos = odeint(num_response, y0, ts, args=(xi+dxi,w_n    ))[:,0]
    y_wn_step_pos = odeint(num_response, y0, ts, args=(xi    ,w_n+dwn))[:,0]
    # y_xi_step_neg = odeint(num_response, y0, ts, args=(xi-dxi,w_n    ))[:,0]
    # y_wn_step_neg = odeint(num_response, y0, ts, args=(xi    ,w_n-dwn))[:,0]
    
    # L2 norm of the error between the synthetic data and the numerical solution
    f             = np.linalg.norm(y_s - y)
    f_xi_step_pos = np.linalg.norm(y_s - y_xi_step_pos)
    f_wn_step_pos = np.linalg.norm(y_s - y_wn_step_pos)
    # f_xi_step_neg = np.linalg.norm(y_s - y_xi_step_neg)
    # f_wn_step_neg = np.linalg.norm(y_s - y_wn_step_neg)
    
    # 1st Order Forward differences
    dfdxi = (f_xi_step_pos - f)/(dxi)
    dfdwn = (f_wn_step_pos - f)/(dwn)
    
    # # 1st Order Backward differences
    # dfdxi = (f - f_xi_step_neg)/(dxi)
    # dfdwn = (f - f_wn_step_neg)/(dwn)
    
    # # 2nd Order Central differences
    # dfdxi = (f_xi_step_pos - f_xi_step_neg)/(2*dxi)
    # dfdwn = (f_wn_step_pos - f_wn_step_neg)/(2*dwn)
    
    return np.array([dfdxi, dfdwn])
    

#%% Plot 2D contour of the cost function

def cost_contour(n_x,n_y,x_min,x_max,y_min,y_max,cost,args=()):
    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(y_min, y_max, n_y)
    # Initialize cost function and gradient
    J = np.zeros((n_y,n_x))
    # Initialize two-dimensional grid space
    X,Y = np.meshgrid(x,y)
    # Evaluate the cost function on the "parameter" space
    print('Drawing 2D cost function map')
    for i in range(0,n_y):
      for j in range(0,n_x):
          print('(%.03i,%.03i) ' %(i,j), end =" ")
          XY        = [X[i,j],Y[i,j]]   
          J[i,j] = cost(XY,*args)
    # Return 
    return J