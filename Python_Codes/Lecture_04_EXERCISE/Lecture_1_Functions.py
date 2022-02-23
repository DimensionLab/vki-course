# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 14:47:18 2022

@author: admin
"""

# This file contains all the functions used in lecture 1

import numpy as np


def Batch_GD(cost,grad,w0,X,y,eta_0,decay,n_epochs,n_batch):
    '''
    This function implements the batch gradient descent
    for the function 'cost', having gradient 'grad' a decaying learning 
    schedule of the form eta_0/(1+k*d). The minimization is performed
    on the dataset X, of size n_p x n_f, with n_p points and n_f features
    
    :param cost: func
                function to be minimized
    :param grad: func
                gradient of the function cost
    :param w0: float
                initial weights         
    :param X: float
                 Design Matrix, n_p x n_f 
    :param y: float
                 Targets, n_p x 1              
    :param eta_0: float
                initial learning rate
    :param decay: float
                decay term in the learning schedule
    :param n_epochs: float
                 number of epochs
    :param n_batch: float
                 batch size
                 
                    
    :return: 
    :param w_opt: float
              final solution found
    :param w_evolution: float, array
              evolution of the weights
    :param Err_SGD: float, array
                     evolution of the errors (in iterations)
              
              
    
    '''
    
    # Prepare the loop per epoch
    n_iter=n_epochs*n_batch
    # number of points and features
    n_p,n_f=np.shape(X)
    # Initialize batch sample Design Matrix
    X_b=np.zeros((n_batch,2))
    # Current estimate of w
    w=w0
    # Initialize the weight evolution and the error evolution
    Err_SGD=np.zeros(n_iter); #Err_SGD[0]=cost(w,X,y)
    w_evolution=np.zeros((n_f,n_iter)); #w_evolution[:,0]=w0
    
    for j in range(n_iter):      
      # Select randomly some data points for the batch
      # Note that replace=False means that there is no repetition
      Indices=np.random.choice(n_p, n_batch,replace=False)
      # Construct the matrix X_b
      X_b=X[Indices,:]; y_b=y[Indices]  
      # Get the current cost
      Err_SGD[j]=cost(w,X_b,y_b)
      #Get the gradient
      Nabla_J_w=grad(w,X_b,y_b)
      # Compute the learning rate
      eta=eta_0/(1+decay*j)
      # Weght update
      w=w-eta*Nabla_J_w
      # Store the result in the history
      w_evolution[:,j]=w
      # Message
      Mex='Iteration: {:d}; Epoch: {:d}; Cost: {:.3f}; Grad_abs: {:.3f}'\
          .format(j,j//n_batch,Err_SGD[j],np.linalg.norm(Nabla_J_w))
      print(Mex)    
      
    w_opt=w # Final result on the weight  
      
    
    return w_opt, w_evolution, Err_SGD


# Useful function for the splitting into training and testing
from sklearn.model_selection import train_test_split


def Boot_Strap_Model_1_esamble(X,y,x_g,n_e=500,tp=0.3):
    '''
    This function implements ensamble statistics from the posterior distributions
    for a linear model with feature matrix X.
        
       
    :param X: float
                 Design Matrix, n_p x n_f 
    :param y: float
                 Targets, n_p x 1              
    :param n_e: number of elements in the ensemble
    :param tp: float
                percentage for the testing portion
    :param x_g: float array
                grid for the prediction
                 
                    
    :return: 
    :param mu_y: array of floats
              ensemble mean for the coefficients
    :param sigma_y: array of floats
              Uncertainty in each prediction (1.96*sigma_y)
     :param J_i_mean: array of floats
              Mean in sample error
     :param J_o_mean: array of floats
              Mean in sample error
              
    
    '''
  #  n_p=np.shape(y)[0] # size of the the set
    # Construct the X on the grid for the predition
    X_g=np.zeros((len(x_g),2));  
    X_g[:,0]=np.ones(len(x_g)); X_g[:,1]=x_g
    
    Pop_Y=np.zeros((len(x_g),n_e)) # posterior population
    mu_y=np.zeros(len(x_g)) # mean prediction
    Unc_y=np.zeros(len(x_g)) # Uncertainty in the prediction
    J_i_mean=np.zeros(n_e) # in sample error
    J_o_mean=np.zeros(n_e) # out of sample error
    w_e=np.zeros((2,n_e)) # Output distribution of weights

    # Take back the x vector
    x=X[:,1]
    
    
    # Loop over the ensamble    
    for j in range(n_e):    
     # Split the    
     xs, xss, ys, yss = train_test_split(x,y, test_size=tp)   
     # prepare the feature matrix for the fitting of a linear model
     X_s=np.zeros((len(xs),2));  X_ss=np.zeros((len(xss),2));
     # Fill in the columns for the Xs
     X_s[:,0]=np.ones(len(xs)); X_s[:,1]=xs
     # Fill in the columns for the Xss
     X_ss[:,0]=np.ones(len(xss)); X_ss[:,1]=xss
     # Fit the weights
     w_s=np.linalg.inv(X_s.T.dot(X_s)).dot(X_s.T).dot(ys)
     # Assign vectors to the distributions
     w_e[:,j]=w_s 
     # Make in-sample prediction
     y_p_s=X_s.dot(w_s); J_i_mean[j]=1/len(xs)*np.linalg.norm(y_p_s-ys)**2
     # Make out-of sample prediction (and errors)
     y_p_ss=X_ss.dot(w_s); J_o_mean[j]=1/len(xss)*np.linalg.norm(y_p_ss-yss)**2
     # Fill the population matrix
     Pop_Y[:,j]=X_g.dot(w_s); 
     
    # Compute the mean and the uncertainty
    mu_y=np.mean(Pop_Y,1) # Mean prediction
    Var_y=(np.mean(J_i_mean))  # Variance from the residuals
    Var_Y_model=np.std(Pop_Y,1)**2
    Var_y_tot=Var_y+Var_Y_model
    Unc_y=1.96*np.sqrt(Var_y_tot)
    
    return mu_y, Unc_y, J_i_mean, J_o_mean,w_e



def Boot_Strap_Model_2_esamble(X,y,x_g,n_e=500,tp=0.3):
    '''
    This function implements ensamble statistics from the posterior distributions
    for a linear model with feature matrix X.
        
    Note: Here we use polyfit and polyval, as these are slightly regularized
        
    :param X: float
                 Design Matrix, n_p x n_f 
    :param y: float
                 Targets, n_p x 1              
    :param n_e: number of elements in the ensemble
    :param tp: float
                percentage for the testing portion
    :param x_g: float array
                grid for the prediction
                 
                    
    :return: 
    :param mu_y: array of floats
              ensemble mean for the coefficients
    :param sigma_y: array of floats
              Uncertainty in each prediction (1.96*sigma_y)
     :param J_i_mean: array of floats
              Mean in sample error
     :param J_o_mean: array of floats
              Mean in sample error
              
    
    '''
    # n_p=np.shape(y)[0] # size of the the set
    # Construct the X on the grid for the predition

    Pop_Y=np.zeros((len(x_g),n_e)) # posterior population
    mu_y=np.zeros(len(x_g)) # mean prediction
    Unc_y=np.zeros(len(x_g)) # Uncertainty in the prediction
    J_i_mean=np.zeros(n_e) # in sample error
    J_o_mean=np.zeros(n_e) # out of sample error
    w_e=np.zeros((3,n_e)) # Output distribution of weights
    # Take back the x vector
    x=X[:,1]
    
    
    # Loop over the ensamble    
    for j in range(n_e):    
     # Split the    
     xs, xss, ys, yss = train_test_split(x,y, test_size=tp)   
     # Fit the weights
     w_s=np.polyfit(xs,ys,2)
     # Assign vectors to the distributions
     w_e[:,j]=w_s 
     # Make in-sample prediction
     y_p_s=np.polyval(w_s,xs); J_i_mean[j]=1/len(xs)*np.linalg.norm(y_p_s-ys)**2
     # Make out-of sample prediction (and errors)
     y_p_ss=np.polyval(w_s,xss); J_o_mean[j]=1/len(xss)*np.linalg.norm(y_p_ss-yss)**2
     # Fill the population matrix
     Pop_Y[:,j]=np.polyval(w_s,x_g); 
     
    # Compute the mean and the uncertainty
    mu_y=np.mean(Pop_Y,1) # Mean prediction
    Var_y=(np.mean(J_i_mean))  # Variance from the residuals
    Var_Y_model=np.std(Pop_Y,1)**2
    Var_y_tot=Var_y+Var_Y_model
    Unc_y=1.96*np.sqrt(Var_y_tot)
    
    return mu_y, Unc_y, J_i_mean, J_o_mean, w_e

# # Testing/debugging code
# W=np.polyfit(X_s[:,1],ys,2)
# plt.plot(X_s[:,1],ys,'ro')
# plt.plot(X_g[:,1],np.polyval(W,X_g[:,1]),'k--');


# W=np.linalg.inv(X_s.T.dot(X_s)).dot(X_s.T).dot(ys)
# plt.plot(X_s[:,1],ys,'ro')
# plt.plot(X_g[:,1],np.polyval(W,X_g[:,1]),'k--');



