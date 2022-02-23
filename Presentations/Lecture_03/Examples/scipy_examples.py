# ## Scipy
# - Typical function and constants (scipy.pi, scipy.sin, etc)
# - Integrators (ODE45 --> scipy.integrate.RK45, scipy.integrate.odeint)
# - Curve fitting (scipy.optimize.curve_fit
# - Interpolation (scipy.interpolate)
# - Statistics (scipy.stats)

# ### Example: Solving an ODE with Scipy
#
# y'(x) = y*ln(y)/x
#
# y(2) = e

from scipy.integrate import odeint
import numpy as np

X0 = [2.0, np.e] # initial value

t = np.linspace(2, 10, 10) # solution points
# Function to integrate
df = lambda y, X: y*np.log(y)/X
# Scipy has a very powerful ODE integrator: - BUILT-IN FUNCTION
y_P = odeint(df, np.e, t)


import matplotlib.pyplot as plt
plt.figure()
plt.plot(t, np.e**(t/2), color='r', label='True')
plt.plot(t, y_P, color='g', linestyle= '--', label= 'scipy.ODEINT', marker='o' )
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
