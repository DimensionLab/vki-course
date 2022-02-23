# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 18:37:01 2022

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



def Initialize_POP(n_p,X_Bounds,n_G=0.5,sigma_I_r=6,I_V=0.1):
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
    I_V : float (default: 6)
         Initial Velocity

    Output
    -----------
    X_V : n_f x n_p array
        Initial Population. Every column contains the Position of a Particle
    V_P : n_f x n_p array
        Initial Velocities. Every column contains the Velocity of a Particle     
    """
  n_f=len(X_Bounds)  
 #Generate an Initial Population  
   #%% Half Initially distributed with Gaussian pdf
  N_Gau_pop=int(n_G*n_p) # number of entries with Gaussian pdf
  X_G=np.zeros((n_f,N_Gau_pop))
  Mean_X=np.zeros((n_f,1)) 
  Sigma_X=Mean_X 
  for j in range(n_f):
   Mean_X=(X_Bounds[j][1]+X_Bounds[j][0])/2
   Sigma_X=abs(X_Bounds[j][1]-X_Bounds[j][0])/sigma_I_r        
   X_G[j,:]=np.random.normal(Mean_X,Sigma_X, N_Gau_pop)
   #%% Half Initially Uniformly distributed 
  n_U=n_p-N_Gau_pop
  # Initialize Uniformly Distributed Positions and Velocities
  X_U=np.zeros((n_f,n_U)); V_P=np.zeros((n_f,n_p))
    
  for j in range(n_f):
   X_U[j,:]=np.random.uniform(X_Bounds[j][1],X_Bounds[j][0], n_U)
   V_MAX=np.abs(X_Bounds[j][1]-X_Bounds[j][0])*I_V
   V_P[j,:]=np.random.uniform(-V_MAX,V_MAX, n_p)
   
  #%% Prepare Initial Population 
  X_V=np.concatenate([X_G, X_U],axis=1) 
  
  
  return X_V, V_P



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



def Update_POP(X_V,V_P,X_B_V,Err_B_V,Err_1,
               X_Bounds,n_I,N_ITER,w_I=0.3,w_F=0.05,c_c=2,c_s=2): 
    """Update Population.

    Parameters
    ------------
    X_V : n_f x n_p array
        Input Particle Positions. A column has the position of Particle k
    V_P : n_f x n_p array
        Input Particle Velocities. A column has the velocity of Particle k
    X_B_V: n_f x n_p array
        Current Best Particle Location.
         A column has the best position ever visited by Particle k
    Err_B_V: n_p x 1
         Best error ever achieved by Particle k in [0,n_p-1]    
    Err_1 :  n_p x 1 array
        Cost of every particle
    X_Bounds : list
        list of bounds for each variable (chromosome)
    n_I : int 
        Number of current iteration
    N_ITER : int 
        Number of iterations that will run    
    w_I : float (default: 0.8)
        Initial Inertia Coefficient
    w_F : float (default: 0.05)
        Final Inertia Coefficient
    c_c : float (default: 2)
        Coefficient of the Cognitive Term   
    c_s : float (default: 2)
        Coefficient of the Social Term
        
    Output
    -----------
    X_V_n : n_f x n_p array
        Updated Particle Position. Every column contains a Particle
    V_P_n : n_f x n_p array
        Updated Particle Velocities. A column has the velocity of Particle k    
    X_B_V: n_f x n_p array
        Update Best Particle Location.
         A column has the best position ever visited by Particle k
    Err_B_V: n_p x 1
         Best error ever achieved by Particle k in [0,n_p-1]         
        
    
    """
    # Optional: Introduce an update bar
    progress(n_I,N_ITER)    
    print("\n Best:  %s Mean %s" % (np.min(Err_1), np.mean(Err_1)))

    n_f,n_p=X_V.shape; # Number of features and Pop size   
    # Coefficient for the Inertia reduction
    alpha=1/N_ITER*np.log(w_F/w_I) # Exp Coefficient
    w=w_I*np.exp(alpha*n_I) # Actual Inertia Coefficient
    
    #%% Find the Global BEST (for SOCIAL TERM)
    ID=np.where(Err_1 == Err_1.min())
    X_BEST=X_V[:,ID[0]] # If you have more than one, take the first
    
    #%% Store the BEST for each particle (for Cognitive TERM)
    for k in range(n_p):
     # Define the Particle BEST
     if Err_1[k]<Err_B_V[k]:
       # Assign best error and best position in items that are overwritten
       Err_P_Best=Err_1[k]; X_P_BEST=X_B_V[:,k]  
       Err_B_V[k]=Err_P_Best # Update the Best Error
       X_B_V[:,k]=X_P_BEST # Update the Best Location
       
    # Define Random numbers for Cognitive and Social Terms  
    R1=np.random.uniform(0,1, (n_f,n_p))
    R2=np.random.uniform(0,1, (n_f,n_p))
       
    X_V_n=X_V+V_P # Update of the Population
    V_P_n=w*V_P+c_c*R1*(X_B_V-X_V)+c_s*R2*(X_BEST-X_V)
             
     #%% Final Concatenation + Boundary Conditions
    for j in range(0,n_f):
     mask1=X_V_n[j,:] < X_Bounds[j][0]
     X_V_n[j,mask1]=X_Bounds[j][0]
     mask2=X_V_n[j,:] > X_Bounds[j][1]
     X_V_n[j,mask2]=X_Bounds[j][1]   
   
    return X_V_n, V_P_n, X_B_V, Err_B_V








#%% 4. Main GA Function Definition (non parallel code)
def PSO(Func,X_Bounds,n_p=100,N_ITER=100,
       n_G=0.5,sigma_I_r=6,w_I=0.0001,w_F=0.0001,c_c=0.1,c_s=0.3): 
    """Particle Swarm Optimization of a function Func.

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

    w_I : float (default: 0.8)
        Initial Inertia Coefficient
    w_F : float (default: 0.05)
        Final Inertia Coefficient
    c_c : float (default: 2)
        Coefficient of the Cognitive Term   
    c_s : float (default: 2)
        Coefficient of the Social Term
        
  
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
    X_V,V_P=Initialize_POP(n_p,X_Bounds,n_G=0.1,sigma_I_r=6,I_V=0)
    # Prepare Some Stats
    Err_Best=np.zeros((N_ITER,1)) # Best Cost Function Value
    Err_Mean=np.zeros((N_ITER,1)) # Mean of the Cost Function
    Err_1=np.zeros((n_p,1)) # Initialization of the Errors
    print('Preparing the loop...')
    for k in range(N_ITER): 
      Err_1=Evaluate_POP(X_V,Func) # Evaluate Error
      if k==0:
       Err_B_V=Err_1 # Best error obtained by a particle
       X_B_V=X_V # Best location ever visited by a particle      
      X_V_n, V_P_n, X_B_V_n, Err_B_V_n=Update_POP(X_V,V_P,X_B_V,Err_B_V,Err_1,\
               X_Bounds,k,N_ITER,w_I,w_F,c_c,c_s) 
      Err_Best[k]=np.min(Err_1); Err_Mean[k]=np.mean(Err_1)      
      
      #%% Step in the future-------------------------------
      X_V=X_V_n; V_P=V_P_n
      X_B_V=X_B_V_n; Err_B_V=Err_B_V_n
      #----------------------------------------------------------
      
      
      
    # Finally give the answer
    Index=Err_1.argmin(); 
    X_S=X_V[:,Index]
    X_U=np.std(X_V,axis=1)
    print('Optimization finished')
    return X_S, X_U, X_V


import os
import matplotlib.pyplot as plt

import imageio
# This function is the same as the previous but also creates a video
# to show the population's dynamics



def Anim_COMP(Func,X_Bounds,n_p=100,N_ITER=100,\
                 n_G=0.5,sigma_I_r=6,w_I=0.1,w_F=0.01,c_c=0.02,c_s=0.5,\
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
   X_V,V_P=Initialize_POP(n_p,X_Bounds,n_G=0.1,sigma_I_r=6,I_V=0)
   # Prepare Some Stats
   Err_Best=np.zeros((N_ITER,1)) # Best Cost Function Value
   Err_Mean=np.zeros((N_ITER,1)) # Mean of the Cost Function
   Err_1=np.zeros((n_p,1)) # Initialization of the Errors
   print('Preparing the loop...')
    
   for k in range(N_ITER): 
     Err_1=Evaluate_POP(X_V,Func) # Evaluate Error
     if k==0:
       Err_B_V=Err_1 # Best error obtained by a particle
       X_B_V=X_V # Best location ever visited by a particle
       
     X_V_n, V_P_n, X_B_V_n, Err_B_V_n=Update_POP(X_V,V_P,X_B_V,Err_B_V,Err_1,\
               X_Bounds,k,N_ITER,w_I,w_F,c_c,c_s) 
     Err_Best[k]=np.min(Err_1); Err_Mean[k]=np.mean(Err_1)  
      
      
     #%% Plotting
     fig= plt.figure(figsize=(10, 4)) # This creates the figure
     ax1 = fig.add_subplot(1,2,1)
     ax2 = fig.add_subplot(1,2,2)
     # First plot
     ax1.contourf(X, Y, COST,cmap='gray', extend='both', alpha=0.5)     
     ax1.quiver(X_V[0,:],X_V[1,:],V_P_n[0,:],V_P_n[1,:],color='r')
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
     
           
     #%% Step in the future-------------------------------
     X_V=X_V_n; V_P=V_P_n
     X_B_V=X_B_V_n; Err_B_V=Err_B_V_n
     #----------------------------------------------------------
      

         
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
   
   imageio.mimsave(GIFNAME, images,duration=0.2)
   import shutil  # nice and powerfull tool to delete a folder and its content
   shutil.rmtree(FOLDER)
   return X_S, X_U, X_V
    









