# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 18:07:58 2022

@author: mendez
"""

import numpy as np
import sys

#%% 0. A fancy progress bar function

# This is just a fancy waitbar
def progress(count, total, suffix=''):
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben


#%% 1. Initialize Population
def Initialize_POP(n_p,X_Bounds,n_G=0.5,sigma_I_r=6):
  """Initialize Population.

    Parameters
    ------------
    n_p : int
        Number of Elements (i.e. size of population)
    X_Bounds : list
        list of bounds for each variable (chromosome).
    n_G : float (default: 0.5, i.e. 50%)
        Portion of the population Distributed with Gaussian Pdf
    sigma_I_r : float (default: 6)
        Interval ratio for computing std of the Gaussian Pdf. 
        e.g.: if sigma_I_r=6, then sigma=X_Bounds/6

    Output
    -----------
    X_V : n_f x n_p array
        Initial Population. Every column contains an individual
    
    """
  n_f=len(X_Bounds)  
 #Generate an Initial Population  
   #%% Half distributed with Gaussian pdf
  N_Gau_pop=int(n_G*n_p) # number of entries with Gaussian pdf
  X_G=np.zeros((n_f,N_Gau_pop))
  Mean_X=np.zeros((n_f,1)) 
  Sigma_X=Mean_X 
  for j in range(n_f):
   Mean_X=(X_Bounds[j][1]+X_Bounds[j][0])/2
   Sigma_X=abs(X_Bounds[j][1]-X_Bounds[j][0])/sigma_I_r        
   X_G[j,:]=np.random.normal(Mean_X,Sigma_X, N_Gau_pop)
   #%% Half Uniformly distributed 
  n_U=n_p-N_Gau_pop
  X_U=np.zeros((n_f,n_U))
  for j in range(n_f):
   X_U[j,:]=np.random.uniform(X_Bounds[j][1],X_Bounds[j][0], n_U)
  #%% Prepare Initial Population 
  X_V=np.concatenate([X_G, X_U],axis=1) 
  return X_V



#%% 2 Evaluate Population
def Evaluate_POP(X_V,Func): 
   """Evaluate a population of candidates.
    Parameters
    ------------
    X_V : n_f x n_p array
        Input Population. Every column contains an individual
    Func : function __main__.Function(X)
        Function we seek to minimize.
    Output
    -----------
    Err_1 :  n_p x 1 array
        Cost of every individual
    
   """
   n_f,n_p=X_V.shape; # Number of features and Pop size
   Err_1=np.zeros((n_p,1)) # Cost Function
   for k in range(n_p): # To be parallelized
    Err_1[k]=Func(X_V[:,k])
   return Err_1


#%% 3. Update Population
def Update_POP(X_V,Err_1,X_Bounds,n_I,N_ITER,mu_I=0.3,mu_F=0.05,p_M=0.5,n_E=0.05): 
    """Update Population.

    Parameters
    ------------
    X_V : n_f x n_p array
        Input Population. Every column contains an individual
    Err_1 :  n_p x 1 array
        Cost of every individual
    X_Bounds : list
        list of bounds for each variable (chromosome)
    n_I : int 
        Number of current iteration
    N_ITER : int 
        Number of iterations that will run    
    mu_I : float (default: 0.3, i.e. 30%)
        Initial portion of the population subject to Mutation
    mu_F : float (default: 0.5, i.e. 50%)
        Final portion of the population subject to Mutation
    p_M : float (default: 0.5, i.e. 50%)
        Portions of the Chromosomes subject to Mutations    
    n_E : float (default: 0.05, i.e. 5%)
        Portion of the population subject to Elitism. 
        This excludes the mutations!
    Output
    -----------
    X_V_n : n_f x n_p array
        Updated Population. Every column contains an individual
    
    """
    # Optional: Introduce an update bar
    progress(n_I,N_ITER)    
    print("\n Best:  %s Mean %s" % (np.min(Err_1), np.mean(Err_1)))
    #%% Sort the Population and bring forward elitism and mutations
    n_f,n_p=X_V.shape; # Number of features and Pop size   
    index=Err_1.argsort(axis=0) # Sorted index 
    # Mutated Elements
    alpha=1/N_ITER*np.log(mu_F/mu_I) # Exp Coefficient
    Mut=mu_I*np.exp(alpha*n_I) # Number of mutate elements (float)
    N_M=int(np.round(Mut*n_p)) # Number of mutate elements (int)
    N_E=int((n_p-N_M)*n_E) # Number of Elite Elements
    N_C=int(n_p-N_M-N_E) # Number of Cross-over Elements
    print(" Elites:%s Mutated:%s Cross-over:%s" % (N_E, N_M, N_C))
    #%% Perform Genetic Operations
    # 1. Elitism---------------------------------------------------------------------
    X_V_E=X_V[:,index[0:N_E,0]]
    # 2. Mutations -----------------------------------------------------------------
    # Number of chromosomes that will mutate
    P_M=int(p_M*n_f)
    # We mutate over the best n_M Individuals 
    # Take randomly the chromosomes that will mutate
    X_V_M=np.zeros((n_f,N_M))
    for m in range(N_M):
     X_V_M[:,m]=X_V[:,index[m,0]] # Take the Best N_M
     print('Mutation ' +str(m))
     for mm in range(P_M):
      Ind_M=np.random.randint(0,n_f)
      print('Change entry ' +str(Ind_M))
      X_V_M[mm,m]=np.random.uniform(X_Bounds[Ind_M][1],X_Bounds[Ind_M][0], 1)              
    # 3. Cross-Over   ------------------------------------------------------------
    X_V_C=np.zeros((n_f,N_C))
    for k in range(0,N_C):
     SEL=np.random.triangular(0, 0, N_C, 2)
     for j in range(0,n_f):
      a=np.random.uniform(0,1,1)
      X_V_C[:,k]  = a*X_V[:,index[int(SEL[0]),0]]  +(1-a)*X_V[:,index[int(SEL[1]),0]]
    #%% Final Concatenation + cleaning    
    X_V_n=np.concatenate([X_V_C, X_V_E,X_V_M],axis=1) 
    for j in range(0,n_f):
     mask1=X_V_n[j,:] < X_Bounds[j][0]
     X_V_n[j,mask1]=X_Bounds[j][0]
     mask2=X_V_n[j,:] > X_Bounds[j][1]
     X_V_n[j,mask2]=X_Bounds[j][1]   
    return X_V_n




#%% 4. Main GA Function Definition (non parallel code)
def GA(Func,X_Bounds,n_p=100,N_ITER=100,
       n_G=0.5,sigma_I_r=6,mu_I=0.3,mu_F=0.05,
       p_M=0.5,n_E=0.05): 
    """Genetic Algorithm Optimization of a function Func.

    Parameters
    ------------
    Func : function __main__.Function(X)
        Function we seek to minimize.
    X_Bounds : list
        list of bounds for each variable (chromosome)
    n_p : int
        Number of Elements (i.e. size of population)
    N_ITER : int 
        Number of iterations that will run    
    n_G : float (default: 0.5, i.e. 50%)
        Portion of the population Distributed with Gaussian Pdf
    sigma_I_r : float (default: 6)
        Interval ratio for computing std of the Gaussian Pdf. 
        e.g.: if sigma_I_r=6, then sigma=X_Bounds/6

    mu_I : float (default: 0.3, i.e. 30%)
        Initial portion of the population subject to Mutation
    mu_F : float (default: 0.5, i.e. 50%)
        Final portion of the population subject to Mutation
    p_M : float (default: 0.5, i.e. 50%)
        Portions of the Chromosomes subject to Mutations    
    n_E : float (default: 0.05, i.e. 5%)
        Portion of the population subject to Elitism. 
        This excludes the mutations!
    parallel: Binary (Default is False)    
    Output
    -----------
    X_S :  n_f x 1 array (Best Solution entry)
        Final Solution
    X_U :  n_f x 1 array
        Solution Uncertainty (std in each entry)   
    X_V: n_f x n_p ( entire Population)    
    
    """
    # Initialize Population
    print('Initializing Population...')
    X_V=Initialize_POP(n_p,X_Bounds,n_G=0.5,sigma_I_r=6)
    # Prepare Some Stats
    Err_Best=np.zeros((N_ITER,1))
    Err_Mean=np.zeros((N_ITER,1))    
    print('Preparing the loop...')
    for k in range(N_ITER): 
      Err_1=Evaluate_POP(X_V,Func) # Evaluate Error
      X_V=Update_POP(X_V,Err_1,X_Bounds,k,N_ITER,\
                     mu_I=mu_I,mu_F=mu_F,p_M=p_M,n_E=n_E) 
      Err_Best[k]=np.min(Err_1); Err_Mean[k]=np.mean(Err_1)         
    # Finally give the answer
    Index=Err_1.argmin(); 
    X_S=X_V[:,Index]
    X_U=np.std(X_V,axis=1)
    print('Optimization finished')
    return X_S, X_U, X_V


#%% 5. Post processing (Valid for the 2D functions implemented)

import os
import matplotlib.pyplot as plt

import imageio
# This function is the same as the previous but also creates a video
# to show the population's dynamics

def Anim_COMP(Func,X_Bounds,n_p=100,N_ITER=100,n_G=0.5,\
                 sigma_I_r=6,mu_I=0.3,mu_F=0.05,p_M=0.5,n_E=0.05,
                 x_1m=-2,x_1M=2,x_2m=-0.5,x_2M=3,npoints=200,Name_Video='Gif.gif'):
   # This function makes an animation of the population search 
   # The X_k is the story of the optimizer's population .
   # func is the function optimized
   # x_1m,x_1M,x_2m,x_2M,n_p is the same input for the plot func
   # Name_Video is the name of the gif that will be exported
    
   # Temporary Folder
   FOLDER='Temp'
   if not os.path.exists(FOLDER):
    os.makedirs(FOLDER) 

   #%% Prepare the Contour
   # Create the vectors for the grid  
   x = np.linspace(x_1m, x_1M, npoints)
   y = np.linspace(x_2m, x_2M, npoints)
   X, Y = np.meshgrid(x, y) # Create grid
   COST=np.zeros((npoints,npoints)) # Initialize the cost func
   # Evaluate the cost function
   for i in range(0,len(x)):
     for j in range(0,len(x)):
      XX=np.array([X[i,j],Y[i,j]]) # Get Point Loc   
      COST[i,j]=Func(XX) # Interrogate the func  

   # Find the approximate location of the minima
   obb=COST.min()
   ID=np.where(COST == obb) 
   
   # Get number of iterations
   plt.ioff()

   # Initialize Population
   print('Initializing Population...')
   X_V=Initialize_POP(n_p,X_Bounds,n_G=0.5,sigma_I_r=6)
   # Prepare Some Stats
   Err_Best=np.zeros((N_ITER,1))
   Err_Mean=np.zeros((N_ITER,1))    
   print('Preparing the loop...')
   for k in range(N_ITER): 
      Err_1=Evaluate_POP(X_V,Func) # Evaluate Error
      X_V=Update_POP(X_V,Err_1,X_Bounds,k,N_ITER,\
                     mu_I=mu_I,mu_F=mu_F,p_M=p_M,n_E=n_E) 
      Err_Best[k]=np.min(Err_1); Err_Mean[k]=np.mean(Err_1)
      #%% Plotting
      fig= plt.figure(figsize=(10, 4)) # This creates the figure
      ax1 = fig.add_subplot(1,2,1)
      ax2 = fig.add_subplot(1,2,2)
      # First plot
      ax1.contourf(X, Y, COST,cmap='gray', extend='both', alpha=0.5)     
      ax1.plot(X_V[0,:],X_V[1,:],'ko',markersize=3)
      ax1.plot(X[ID],Y[ID],'ro',markersize=5)    
      ax1.set_xlim([x_1m, x_1M])
      ax1.set_ylim([x_2m, x_2M])   
      # Second Plot
      ax2.plot(np.linspace(0,k,k),Err_Best[0:k],'ro:',label='Best')
      ax2.plot(np.linspace(0,k,k),Err_Mean[0:k],'bo:',label='Mean')
      ax2.legend()
      # Give the iteration number      
      plt.title("Iteration "+ str(k))
      plt.savefig(FOLDER+'/'+'Step'+str(k)+'.png',dpi=200)
      plt.close('all')

         
   # Finally give the answer
   Index=Err_1.argmin(); 
   X_S=X_V[:,Index]
   X_U=np.std(X_V,axis=1)
   print('Optimization finished')
  
   # Make a Gif 1
   GIFNAME=Name_Video
   images=[]    
   for k in range(N_ITER):
     MEX= 'Preparing Im '+ str(k)+' of ' + str(N_ITER-1)
     print(MEX)
     FIG_NAME=FOLDER+'/'+'Step'+str(k)+'.png'
     images.append(imageio.imread(FIG_NAME))
   
   imageio.mimsave(GIFNAME, images,duration=0.5)
   import shutil  # nice and powerfull tool to delete a folder and its content
   shutil.rmtree(FOLDER)
   return X_S, X_U, X_V
    





















