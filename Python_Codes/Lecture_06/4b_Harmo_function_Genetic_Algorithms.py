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


#%% Import the key functions
from GA_Functions import Initialize_POP
from GA_Functions import Evaluate_POP
from GA_Functions import Update_POP
from GA_Functions import GA




#%% Run the GA optimization
# Define the input parameters
# ----------------------------------------
# Select a population of n_proc*100 elements
N_pop_PP=int(100) # Population PER Processor
N_POP=100 # Total Population
# Boundaries of the function to be optimized
X_Bounds=[(-8,8),(-8,8)] 
N_ITER=50 # Number of iterations
mu_I=0.02; mu_F=0.0001 # Initial and Final Mutation Rates    
p_M=1 # Portion of the Chromosome subject to Mutation
n_E=0.05 # Portion subjet to Elitism
Func=harmo # Function to minimize
# ----------------------------------------

# X_S, X_U, X_V=GA(Func,X_Bounds,n_p=100,N_ITER=100,n_G=0.5,\
#                  sigma_I_r=6,mu_I=0.3,mu_F=0.05,p_M=0.5,n_E=0.05)


from GA_Functions import Anim_COMP


# Run the Animation
X_S, X_U, X_V=Anim_COMP(Func=harmo,X_Bounds=X_Bounds,n_p=200,N_ITER=20,n_G=0.5,\
                 sigma_I_r=6,mu_I=0.2,mu_F=0.05,p_M=0.5,n_E=0.05, \
                 x_1m=-8,x_1M=8,x_2m=-8,x_2M=8,npoints=200,\
                 Name_Video='harmo_GA.gif')

