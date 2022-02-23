# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 14:46:14 2022

@author: Miguel Alfonso Mendez
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Preamble: customization of matplotlib
# Configuration for plots
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)


# Generate data
n_p=100
np.random.seed(10)
x_s=np.random.uniform(0,10,n_p)
y_s=2*x_s+2+np.random.normal(loc=0,scale=10,size=len(x_s))+x_s**2

# Show the results
fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
plt.scatter(x_s,y_s,c='black',marker='o',edgecolor='black',s=16)
ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)
Name='Exercise_1_data.png'
plt.tight_layout()
plt.savefig(Name, dpi=200) 


#%% Step 2: Test Analytical Solution vs polyfit
# Python's scipy offers the polyfit function like in matlab.
# Here's the fit function from numpy
w_s=np.polyfit(x_s,y_s,1)    
# We now want to test our model on a new regular grid
x_t=np.linspace(0,10,200)
# The prediction would thus be
y_t=np.polyval(w_s,x_t)
# Show the result of the fit 
fig, ax = plt.subplots(figsize=(5, 3)) 
plt.scatter(x_s,y_s,c='black',
            marker='o',edgecolor='black',s=16)
plt.plot(x_t,y_t,'r--',linewidth=2)
ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)
Name='Exercise_1_data_fit.png'
plt.tight_layout()
plt.savefig(Name, dpi=200) 

#%% Step 3: Run the stochastic gradient descent and make a movie

# Create the Design Matrix
X=np.zeros((n_p,3))
X[:,0]=1 # First column
X[:,1]=x_s # Second Column
X[:,2]=x_s**2 # Second Column

# Analytical Solution 
w_s2=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_s)   


# Create a function for the analytical solution with minimal
# regularization

def line_s_fit(x,y):    ######################  1.YOUR CODE GOES HERE ########
    # take input vectors x,y and fit a line
    # with coefficients w_0 and w_1 

    return w    

def Quadratic_s_fit(x,y):    #################  2.YOUR CODE GOES HERE ########
    # take input vectors x,y and fit a line
    # with coefficients w_0 and w_1 

    return w    

def line_s_fit_red(x,y,k_A_l=1000):
    # take input vectors x,y and fit a line
    # with coefficients w_0 and w_1 
    n_p=len(x)
    X=np.zeros((n_p,2))
    X[:,0]=1 # First column
    X[:,1]=x # Second Column
    # Compute the Hessian
    H=X.T@(X)
    w=np.linalg.inv(H).dot(X.T).dot(y)
    return w  
    
def Quadratic_s_fit_red(x,y,k_A_l=1000):
     # take input vectors x,y and fit a line
     # with coefficients w_0 and w_1 
     n_p=len(x)
     X=np.zeros((n_p,3))
     X[:,0]=
     X[:,1]=
     X[:,2]=
     # Compute the Hessian
     H=X.T@(X)
     # compute the eigenvalues, the largest (l_M) and the smallest (l_m)
     Lambd=np.linalg.eig(H); l_M=np.max(Lambd[0]); l_m=np.min(Lambd[0]);
     alpha=
     H_p=H+alpha*np.identity(np.shape(H)[0])
     w=np.linalg.inv(H_p).dot(X.T).dot(y)
     return w     


w_OLS=Quadratic_s_fit(x_s,y_s)
print(w_OLS)

w_Q=Quadratic_s_fit_red(x_s,y_s,k_A_l=1000)
print(w_Q)

w_P=np.polyfit(x_s,y_s,2)
print(w_P)


     


#%% Step 3: Run the boot strapping 
 #################  3. YOUR CODE GOES HERE ###################################
 
# Useful function for the splitting into training and testing
from sklearn.model_selection import train_test_split

def Boot_Strap_Model_2_esamble(X,y,x_g,n_e=500,tp=0.3):
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
     w_s=
     # Assign vectors to the distributions
     w_e[:,j]=w_s 
     # Make in-sample prediction---------------------------
     y_p_s=
     # In-sample error
     J_i_mean[j]=
     # Make out-of sample prediction (and errors)
     y_p_ss=
     # Out of sample error
     J_o_mean[j]=
     # Fill the population matrix
     Pop_Y[:,j]=
     
    # Compute the mean and the uncertainty
    mu_y=
    Var_y=
    Var_Y_model=
    Var_y_tot=
    Unc_y=1.96*np.sqrt(Var_y_tot)
    
    return mu_y, Unc_y, J_i_mean, J_o_mean, w_e



def Boot_Strap_Model_2_esamble_reg(X,y,x_g,n_e=500,tp=0.3,K_A=1000):
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
     w_s=Quadratic_s_fit_red(xs,ys,k_A_l=K_A)
     # Assign vectors to the distributions
     w_e[:,j]=w_s 
     # Make in-sample prediction---------------------------
     y_p_s=np.polyval(w_s,xs); 
     # In-sample error
     J_i_mean[j]=1/len(xs)*np.linalg.norm(y_p_s-ys)**2
     # Make out-of sample prediction (and errors)
     y_p_ss=np.polyval(w_s,xss);
     # Out of sample error
     J_o_mean[j]=1/len(xss)*np.linalg.norm(y_p_ss-yss)**2
     # Fill the population matrix
     Pop_Y[:,j]=np.polyval(w_s,x_g); 
     
    # Compute the mean and the uncertainty
    mu_y=np.mean(Pop_Y,1) # Mean prediction
    Var_y=(np.mean(J_i_mean))  # Variance from the residuals
    Var_Y_model=np.std(Pop_Y,1)**2
    Var_y_tot=Var_y+Var_Y_model
    Unc_y=1.96*np.sqrt(Var_y_tot)
    
    return mu_y, Unc_y, J_i_mean, J_o_mean, w_e



# Repeat the same for a second order model
X=np.zeros((n_p,3))
X[:,0]=1 # First column
X[:,1]=x_s # Second Column
X[:,2]=x_s**2 # Third Column

x_g=np.linspace(0,10,100)

mu_y, Unc_y, J_i, J_o, w_e=Boot_Strap_Model_2_esamble(X,y_s,x_g,n_e=500,tp=0.3)

J_i_mean=np.mean(J_i)
J_o_mean=np.mean(J_o)

Mex='J_i_mean:{:.1f}; J_o_mean: {:.1f}'.format(J_i_mean,J_o_mean)
print(Mex)   


# Plot linear model with Uncertainties
fig, ax = plt.subplots(figsize=(5, 3)) 
plt.scatter(x_s,y_s,c='black',
            marker='o',edgecolor='black',s=16)
plt.plot(x_g,mu_y,'r--',linewidth=2)
plt.fill_between(x_g, mu_y + Unc_y,mu_y - Unc_y, alpha=0.5)

ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)
Name='Model_2_unc_unreg.png'
plt.tight_layout()
plt.savefig(Name, dpi=200) 

# Distribution of weights 1
fig, ax = plt.subplots(figsize=(5, 3)) 
plt.hist(w_e[0,:],100,label='$p(w_0|\mathbf{x})$',alpha=0.8)
plt.hist(w_e[1,:],100,label='$p(w_1|\mathbf{x})$',alpha=0.6)
plt.hist(w_e[2,:],100,label='$p(w_2|\mathbf{x})$',alpha=0.4)
ax.set_xlabel('$w_0$, $w_1$, $w_2$',fontsize=14)
ax.set_ylabel('$p(w_0|\mathbf{x})$, $p(w_1|\mathbf{x})$, \
               $p(w_2|\mathbf{x})$',fontsize=14)
plt.legend()
plt.tight_layout()
Name='Posteriors_2_unreg.png'
plt.tight_layout()
plt.savefig(Name, dpi=300)
plt.close("all")

####### REGULIRIZED CASE ####################################################


# Repeat the same for a second order model
X=np.zeros((n_p,3))
X[:,0]=1 # First column
X[:,1]=x_s # Second Column
X[:,2]=x_s**2 # Third Column

x_g=np.linspace(0,10,100)

mu_y,Unc_y,J_i,J_o,w_e=Boot_Strap_Model_2_esamble_reg(X,y_s,\
                                                      x_g,n_e=500,\
                                                          tp=0.3,K_A=1000)

J_i_mean=np.mean(J_i)
J_o_mean=np.mean(J_o)

Mex='J_i_mean:{:.1f}; J_o_mean: {:.1f}'.format(J_i_mean,J_o_mean)
print(Mex)   


# Plot linear model with Uncertainties
fig, ax = plt.subplots(figsize=(5, 3)) 
plt.scatter(x_s,y_s,c='black',
            marker='o',edgecolor='black',s=16)
plt.plot(x_g,mu_y,'r--',linewidth=2)
plt.fill_between(x_g, mu_y + Unc_y,mu_y - Unc_y, alpha=0.5)

ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)
Name='Model_2_unc_reg.png'
plt.tight_layout()
plt.savefig(Name, dpi=200) 

# Distribution of weights 1
fig, ax = plt.subplots(figsize=(5, 3)) 
plt.hist(w_e[0,:],100,label='$p(w_0|\mathbf{x})$',alpha=0.8)
plt.hist(w_e[1,:],100,label='$p(w_1|\mathbf{x})$',alpha=0.6)
plt.hist(w_e[2,:],100,label='$p(w_2|\mathbf{x})$',alpha=0.4)
ax.set_xlabel('$w_0$, $w_1$, $w_2$',fontsize=14)
ax.set_ylabel('$p(w_0|\mathbf{x})$, $p(w_1|\mathbf{x})$, \
               $p(w_2|\mathbf{x})$',fontsize=14)
plt.legend()
plt.tight_layout()
Name='Posteriors_2_reg.png'
plt.tight_layout()
plt.savefig(Name, dpi=300)
plt.close("all")



#%% Step 4 Gradient Descent
 #################  4. YOUR CODE GOES HERE ###################################

# Create a cost function

def cost(w,X,y):
    
    return J

# Create a gradient function

def grad(w,X,y):

    return Nabla_J

    
def Batch_GD(cost,grad,w0,X,y,eta_0,decay,n_epochs,n_batch):
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








