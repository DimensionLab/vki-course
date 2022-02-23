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

#%% Rosenbrock function definition
def rosenbrock(X):  # The Function to Minimize  
    """Function to be minimized"""  # new        
    x = X[0]; y = X[1]
    C= 100*(y-x**2)**2+(1-x)**2.   
    return C 


#%% Import the key functions
from GA_Functions import Initialize_POP
from GA_Functions import Evaluate_POP
from GA_Functions import Update_POP
from GA_Functions import GA

# Test these separately to make sure you understand how they function!


#%% Run the GA optimization
# Define the input parameters
# ----------------------------------------
# Select a population of n_proc*100 elements
N_POP=100 # Total Population
# Boundaries of the function to be optimized
X_Bounds=[(-2,2),(-0.5,3)] 
N_ITER=50 # Number of iterations
mu_I=0.02; mu_F=0.0001 # Initial and Final Mutation Rates    
p_M=1 # Portion of the Chromosome subject to Mutation
n_E=0.05 # Portion subjet to Elitism
Func=rosenbrock # Function to minimize
# ----------------------------------------

X_S, X_U, X_V=GA(Func,X_Bounds,n_p=100,N_ITER=100,n_G=0.5,\
                 sigma_I_r=6,mu_I=0.3,mu_F=0.05,p_M=0.5,n_E=0.05)


from GA_Functions import Anim_COMP


# Run the Animation
X_S, X_U, X_V=Anim_COMP(Func=rosenbrock,X_Bounds=X_Bounds,n_p=200,N_ITER=20,n_G=0.5,\
                 sigma_I_r=6,mu_I=0.3,mu_F=0.05,p_M=0.5,n_E=0.05, \
                 x_1m=-2,x_1M=2,x_2m=-0.5,x_2M=3,npoints=200,\
                 Name_Video='rosen_GA.gif')












