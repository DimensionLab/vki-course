# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 11:52:32 2022

@author: mendez
"""

import numpy as np
import matplotlib.pyplot as plt

# Configuration for plots
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

#%% Problem to Fit
x1 = np.linspace(0, 4.3, 200, endpoint=True)
x2 = np.linspace(4.8, 10, 200, endpoint=True)
x=np.concatenate((x1,x2))
# Create the deterministic part
y_clean= 3*x+(x/100)**3+4*np.sin(3/2*np.pi*x)
# Add (a seeded) stochastic part
np.random.seed(0)
y=y_clean+1*np.random.randn(len(x))
# Introduce some outliers in x=2 and x=8
G1=10*np.exp(-(x-2)**2/0.005)*np.random.randn(len(x))
G2=15*np.exp(-(x-8)**2/0.005)*np.random.randn(len(x))
y_final=y+G1+G2


fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
plt.scatter(x,y_final,c='black',
            marker='o',edgecolor='black',s=16)
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
ax.set_xlabel('x',fontsize=12)
ax.set_ylabel('y',fontsize=12)
Name='Exercise_1_data.png'
plt.tight_layout()
plt.savefig(Name, dpi=300) 
plt.show()
#shutil.copyfile(Name,'../../Figures/'+Name) #  copy file to latex directory


#%% Create Feature matrix for the Sigmoid
# Define the number of bases
n_b=100


# Perform the feature scaling

from sklearn.preprocessing import MinMaxScaler
# Scale the x 
scaler_X = MinMaxScaler(); 
scaler_X.fit_transform(x.reshape(-1,1))
x_prime=scaler_X.transform(x.reshape(-1,1)) # Scale

# Scale also the y 
scaler_Y = MinMaxScaler(); 
scaler_Y.fit_transform(y_final.reshape(-1,1))
y_prime=scaler_Y.transform(y_final.reshape(-1,1)) # Scale
    
# Define grid of collocation points
x_b=np.linspace(0,1,n_b)

#%% 1 Sigmoid Basis function
def sigmoid(x,x_r=0,c_r=0.1):
    z=(x-x_r)/c_r;
    phi_r=1/(1+np.exp(-z))
    return phi_r

def PHI_sigmoid_X(x_in, x_b, c_r=0.05):
 n_x=np.size(x_in)
 Phi_X=np.zeros((n_x,n_b+1)) # Initialize Basis Matrix on x
 # Add a constant and a linear term
 Phi_X[:,0]=Phi_X[:,1]=x_in 
 for j in range(0,n_b): # Loop to prepare the basis matrices (inefficient)
  Phi_X[:,j+1]=sigmoid(x_in,x_r=x_b[j],c_r=c_r)  # Prepare all the terms in the basis 
 return Phi_X

# Check the basis:
PHI_sig_X=PHI_sigmoid_X(x_prime[:,0], x_b, c_r=0.05)
plt.figure(1)
plt.plot(x_prime,PHI_sig_X)

# Compute the condition number 
H_sig=PHI_sig_X.T@PHI_sig_X
k_H_sig=np.linalg.cond(H_sig)
print('rcond for sigmoid: {:.3f}'.format(k_H_sig))    
    
    
#%% 2 Gaussian Basis function
def Gauss_RBF(x,x_r=0,c_r=0.1):
    d=x-x_r # Get distance
    phi_r=np.exp(-c_r**2*d**2)
    return phi_r

def PHI_Gauss_X(x_in, x_b, c_r=0.05):
 n_x=np.size(x_in)
 Phi_X=np.zeros((n_x,n_b+1)) # Initialize Basis Matrix on x
 # Add a constant and a linear term
 Phi_X[:,0]=x_in 
 # Loop to prepare the basis matrices (inefficient)
 for j in range(0,n_b): 
  # Prepare all the terms in the basis 
  Phi_X[:,j+1]=Gauss_RBF(x_in,x_r=x_b[j],c_r=c_r)  
 return Phi_X

# Check the basis:
Phi_Gauss_X=PHI_Gauss_X(x_prime[:,0], x_b, c_r=1/0.05)
plt.figure(1)
plt.plot(x_prime,Phi_Gauss_X)

# Compute the condition number 
H_Gauss=Phi_Gauss_X.T@Phi_Gauss_X
k_H_Gauss=np.linalg.cond(H_Gauss)
print('rcond for Gauss RBF: {:.3f}'.format(k_H_Gauss))    

#%% 3 C4 Basis
def C4_Compact_RBF(x,x_r=0,c_r=0.1):
    d=x-x_r # Get distance
    phi_r=(1+d/c_r)**5*(1-d/c_r)**5
    phi_r[np.abs(d)>c_r]=0
    return phi_r

def PHI_C4_X(x_in, x_b, c_r=0.1):
 n_x=np.size(x_in); n_b=len(x_b)
 Phi_X=np.zeros((n_x,n_b+1)) # Initialize Basis Matrix on x
 # Add a constant and a linear term
 Phi_X[:,0]=x_in 
 for j in range(0,n_b): # Loop to prepare the basis matrices (inefficient)
  Phi_X[:,j+1]=C4_Compact_RBF(x_in,x_r=x_b[j],c_r=c_r)  # Prepare all the terms in the basis 
 return Phi_X

# Check the basis:
Phi_C4_X=PHI_C4_X(x_prime[:,0], x_b, c_r=0.1)
plt.figure(1)
plt.plot(x_prime,Phi_C4_X)

# Compute the condition number 
H_C4=Phi_C4_X.T@Phi_C4_X
k_H_C4=np.linalg.cond(H_C4)
print('rcond for C4 RBF: {:.3f}'.format(k_H_C4))    
 


#%% OS prediction for the case of 
H_C4=Phi_C4_X.T@Phi_C4_X
# Train Model
w_C4=np.linalg.inv(H_C4).dot(Phi_C4_X.T).dot(y_prime)   
# Make prediction on the nex x
x_test=np.linspace(0,1,300)
# Prepare the basis on the new data
Phi_X_test=PHI_C4_X(x_test, x_b, c_r=0.1)
# Predictions on the new data
y_prime_pred=Phi_X_test.dot(w_C4)

# revert the transform:
fig, ax = plt.subplots(figsize=(5, 3)) 
plt.scatter(x_prime,y_prime,c='white',
            marker='o',edgecolor='black',
            s=10,label='Data')
plt.plot(x_test,y_prime_pred,'r--')

ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)  

#%% We can use scikitlearn:
from sklearn.linear_model import LinearRegression
reg = LinearRegression(fit_intercept=False).fit(Phi_C4_X, y_prime)
# look for the coefficients:
w_O_s=reg.coef_.T
# Regression:
y_prime_pred_s=reg.predict(Phi_X_test)


# revert the transform:
fig, ax = plt.subplots(figsize=(5, 3)) 
plt.scatter(x_prime,y_prime,c='white',
            marker='o',edgecolor='black',
            s=10,label='Data')
plt.plot(x_test,y_prime_pred,'r--',label='numpy')
plt.plot(x_test,y_prime_pred_s,'b--',label='scikit')

ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)  

plt.legend()
Name='NPvsScikit_OLS.png'
plt.tight_layout()
plt.savefig(Name, dpi=200) 


fig, ax = plt.subplots(figsize=(5, 3)) 
plt.scatter(x_prime,y_prime,c='white',
            marker='o',edgecolor='black',
            s=10,label='Data')
plt.plot(x_test,y_prime_pred,'r--',label='numpy')
plt.plot(x_test,y_prime_pred_s,'b--',label='scikit')

ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)  

plt.legend()
Name='NPvsScikit_OLS_zoom.png'

plt.xlim([0.5,0.7])

plt.ylim([0,1])
plt.tight_layout()
plt.savefig(Name, dpi=200) 






plt.plot(w_C4,w_O_s,'ko')
plt.plot(w_C4,w_C4,'r--')

#%% Ridge Regression via numpy
alpha=0.2
w_star_C4=np.linalg.inv(Phi_C4_X.T.dot(Phi_C4_X)+alpha*np.eye(n_b+1)).\
           dot(Phi_C4_X.T).dot(y_prime)

from sklearn.linear_model import Ridge
clf = Ridge(fit_intercept=False,alpha=0.2); 
clf.fit(Phi_C4_X, y_prime)

w_reg_R_s=clf.coef_

# Check that these are the same:
plt.plot(w_star_C4,w_reg_R_s.T,'ko')
plt.plot(w_star_C4,w_star_C4,'r--')

#%% Question 3 

#%% Ordinary Least Square
from sklearn.linear_model import LinearRegression
OLS = LinearRegression(fit_intercept=False); 
OLS.fit(Phi_C4_X, y_prime)
w_reg_s=OLS.coef_
# Regression:
y_prime_pred_O=OLS.predict(Phi_X_test)


#%% Lasso Regression
from sklearn.linear_model import Lasso
Las = Lasso(fit_intercept=False,alpha=0.01); 
Las.fit(Phi_C4_X, y_prime)
w_reg_L_s=Las.coef_
# Regression:
y_prime_pred_R=Las.predict(Phi_X_test)

#%% Ridge Regression
from sklearn.linear_model import Ridge
Rid = Ridge(fit_intercept=False,alpha=0.01); 
Rid.fit(Phi_C4_X, y_prime)
w_reg_R_s=Rid.coef_
# Regression:
y_prime_pred_L=Rid.predict(Phi_X_test)

#%% Plot the three
fig, ax = plt.subplots(figsize=(5, 3)) 
plt.scatter(x_prime,y_prime,c='white',
            marker='o',edgecolor='black',
            s=10,label='Data')
plt.plot(x_test,y_prime_pred_O,'r--',label='OLS')
plt.plot(x_test,y_prime_pred_R,'b--',label='Ridge')
plt.plot(x_test,y_prime_pred_L,'k--',label='Lasso')

ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)  

plt.legend()
Name='Lasso_Ridge_OLS_full.png'

plt.tight_layout()
plt.savefig(Name, dpi=200) 

#%% Plot the three
fig, ax = plt.subplots(figsize=(5, 3)) 
plt.scatter(x_prime,y_prime,c='white',
            marker='o',edgecolor='black',
            s=10,label='Data')
plt.plot(x_test,y_prime_pred_O,'r--',label='OLS')
plt.plot(x_test,y_prime_pred_R,'b--',label='Ridge')
plt.plot(x_test,y_prime_pred_L,'k--',label='Lasso')

ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)  

plt.legend()
plt.xlim([0.5,0.7])

plt.ylim([0,1])
Name='Lasso_Ridge_OLS_Zoom.png'

plt.tight_layout()
plt.savefig(Name, dpi=200) 

#%% plot the weights

fig, ax = plt.subplots(figsize=(5, 2)) 
plt.title('Weights from OLS',fontsize=16)
plt.stem(w_reg_s[0,:])
ax.set_xlabel('$r$',fontsize=18)
ax.set_ylabel('$w_r$',fontsize=18)  
Name='W_OLS.png'
plt.tight_layout()
plt.savefig(Name, dpi=200) 


fig, ax = plt.subplots(figsize=(5, 2)) 
plt.title('Weights from Ridge R',fontsize=16)
plt.stem(w_reg_R_s[0,:])
ax.set_xlabel('$r$',fontsize=16)
ax.set_ylabel('$w_r$',fontsize=16) 
Name='W_R.png'
plt.tight_layout()
plt.savefig(Name, dpi=200) 


fig, ax = plt.subplots(figsize=(5, 2)) 
plt.title('Weights from Lasso R',fontsize=16)
plt.stem(w_reg_L_s)
ax.set_xlabel('$r$',fontsize=16)
ax.set_ylabel('$w_r$',fontsize=16)  
Name='W_L.png'
plt.tight_layout()
plt.savefig(Name, dpi=200) 

#%% Question 4

# Here we perform the K-fold validation
K=5 # number of folds
alphas=np.linspace(0,0.01,500)
from sklearn.model_selection import KFold
# Create the k-fold split object
kf = KFold(n_splits=K, random_state=3, shuffle=True)

# # Perform the splitting
# kf.split(Phi_C4_X, y_prime)
count=1
# Have a look at the data spliting!
for train_index, test_index in kf.split(Phi_C4_X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    Phi_X_train, Phi_X_test = Phi_C4_X[train_index], Phi_C4_X[test_index]
    fig, ax = plt.subplots(figsize=(5, 3))  
    y_train, y_test = y_prime[train_index], y_prime[test_index]
    x_train, x_test = x_prime[train_index], x_prime[test_index]
    plt.plot(x_train,y_train,'bo',label='Train')
    plt.plot(x_test,y_test,'ro',label='Test')
    plt.legend()
    ax.set_xlabel('$x$',fontsize=16)
    ax.set_ylabel('$y$',fontsize=16)  
    Name='Fold_'+str(count)+'.png'
    count+=1
    plt.tight_layout()
    plt.savefig(Name, dpi=200) 


Animation_Name='K_Folds.gif'

# Make the animation
import imageio  # This used for the animation
images = []
    
for k in range(1,10):
  MEX = 'Preparing Im ' + str(k) 
  print(MEX)  
  FIG_NAME ='Fold_'+str(k)+'.png'  
  images.append(imageio.imread(FIG_NAME))
        
    
     # Now we can assembly the video and clean the folder of png's (optional)
imageio.mimsave(Animation_Name, images, duration=0.3)
 

#%% Loop for the regression

#%% Question 4
# Here we perform the K-fold validation
K=5 # number of folds
alphas=np.linspace(0,0.01,500)
J_out=np.zeros(len(alphas))
J_in=np.zeros(len(alphas))
from sklearn.model_selection import KFold
# Create the k-fold split object
kf = KFold(n_splits=K, random_state=3, shuffle=True)

# Loop over given alphas
for j in range(len(alphas)):
  print('alpha '+str(j)+' of '+str(len(alphas)))
  # Select one alpha
  alpha=alphas[j]  
  # Initialize the out of sample error vector
  count=0; J_out_fold=np.zeros(K);J_in_fold=np.zeros(K)
  # Loop over the folds
  for train_index, test_index in kf.split(Phi_C4_X):
  # Get the training and test sets
   Phi_X_train, Phi_X_test = Phi_C4_X[train_index], Phi_C4_X[test_index] 
   y_train, y_test = y_prime[train_index], y_prime[test_index] 
   # Fit the model on the trainig set
   Rid = Ridge(fit_intercept=False,alpha=alpha) 
   Rid.fit(Phi_X_train, y_train)
   # Test the model on the testing set
   y_prime_test=Rid.predict(Phi_X_test); y_prime_train=Rid.predict(Phi_X_train)  
   # Collect all the out of sample errors
   J_in_fold[count]=np.mean((y_prime_train-y_train)**2) 
   J_out_fold[count]=np.mean((y_prime_test-y_test)**2) 
   # Take the mean out of sample error over the folds
   J_out[j]=np.mean(J_out_fold)
   J_in[j]=np.mean(J_in_fold)    
       
fig, ax = plt.subplots(figsize=(5, 3)) 
plt.plot(alphas,J_in,label='$J_i$')
plt.plot(alphas,J_out,label='$J_o$')
plt.legend()
ax.set_xlabel('$\\alpha$',fontsize=16)
ax.set_ylabel('$J_i,J_o$',fontsize=16)  
Name='J_i_J_o.png'
plt.tight_layout()
plt.savefig(Name, dpi=200) 




#%% Support vector regression in Python
from sklearn.svm import SVR
# Create SVR object
svr=SVR(kernel='rbf',gamma=100,C=10,epsilon=0.3)
# Fit the regressor
svr.fit(x_prime,np.ravel(y_prime))
# Make predictions:
y_p_SVM=svr.predict(x_test.reshape(-1,1))    
# Look for the epsilon sensitive tube
y_p_eps=y_p_SVM+svr.epsilon
y_m_eps=y_p_SVM-svr.epsilon

# Plot the result
fig, ax = plt.subplots(figsize=(5, 3)) 
plt.scatter(x_prime,y_prime,c='white',marker='o',edgecolor='black',
            s=10,label='Data')
plt.plot(x_test,y_p_SVM,'r--')
plt.plot(x_test,y_m_eps,'b--')
plt.plot(x_test,y_p_eps,'b--')

ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)  

Name='SVR.png'

plt.tight_layout()
plt.savefig(Name, dpi=200) 

