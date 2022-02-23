import numpy as np  # by convention alias np

# Check here the list of almost all useful functions: <br>
# https://numpy.org/devdocs/user/numpy-for-matlab-users.html

# ## Multiple ways of creating numpy arrays
# * From lists:

m = np.array([[2, 7, 9], [4, 3, 3], [8, 1, 0]]);
print(m)
type(m)

# * Using functions:

print(np.zeros((3, 2)))
print(np.ones((2, 3)))
print(np.linspace(0.0, 10.0, 3))
print(np.logspace(1.0, 3.0, 3))

print(np.random.rand(2, 4))

# ## Matrix properties

print(m)

print(m.shape)  # shape of matrix
print(np.shape(m))
print(m.size)  # number of elements
print(np.size(m))

# ## Indexing and slicing arrays as for lists

print(m)

print(m[0, 0])

print(m[:, -1])  # get the last column

print(m[0:2, 1])  # get two first elements of first column

print(m[[0, 2], 0])  # from column 0, get elements 0 and 2

# ## Linear algebra
# * Element-wise operations:

m = np.array([[0, 1, 2], [3, 4, 5]])  # our initial matrix
print(m * 2)

print(m + 2)

print(m ** 2)

# * Matrix algebra:

v = (np.array([1, 2, 3]))
print(m)
print(m.shape)
print(v)
print(v.shape)

print(m.dot(v))  # matrix multiplication
print(np.dot(m, v))
print(m @ v)

print(m.transpose())  # transpose
print(m.T)  # transpose, shorter
print(m.flatten())  # convert to 1D vector

# reshape
m_reshape = m.reshape(6, 1)
print(m_reshape.shape)

# ### About dimensions in numpy...
#
# This is used sometimes in sklearn

print(m_reshape.shape)
print(v.shape)

v = np.expand_dims(v, axis=1)
print(v.shape)

# ## Numpy does not change the original matrix inplace
# ### Assign to a new variable if needed


print(m)
print(m.T)
print('---Square matrix did not change---')
print(m)

# ### Calculate inverse and check if identity matrix is recovered


square_matrix = np.array([[2, 7, 9], [4, 2, 3], [0, 1, 9]])
inverse_matrix = np.linalg.inv(square_matrix)
identity_matrix = square_matrix.dot(inverse_matrix)
print(np.around(identity_matrix, decimals=0))  # Roundoff error

# ## Numpy matrix algebra is faster than loops!
# ### To check this, let's calculate the modulus of a vector in three ways:
# 1. Loops
# 2. Matrix multiplication
# 3. Built-in function

import time  # to compute the time neeed

# Vector of n points between [-10, 10]
n = 10000000
v = np.random.normal(loc=0, scale=20, size=n);  # use of ; avoids print of ouput in jupyter

# ## With loops: $|\textbf{v}|=\sqrt{\sum v_i^2}$


start = time.time()  # get start time

sum_vectors_squared = 0
for vi in v:
    sum_vectors_squared += vi ** 2
module = np.sqrt(sum_vectors_squared)

end = time.time()  # get end time

print("Module is:", module)
print("Execution time :", end - start, " s")

# ## With matrices:  $|\textbf{v}|=\sqrt{ v^T\cdot v}$

start = time.time()

product = v.transpose().dot(v)
module = np.sqrt(product)

end = time.time()

print("Module is:", module)
print("Execution time :", end - start, " s")

# ## With built-in function norm: $|\textbf{v}|=$`np.norm(v)`

start = time.time()

module = np.linalg.norm(v)

end = time.time()

print("Module is:", module)
print("Execution time :", end - start, " s")

# ### Conclusion: ALWAYS search for built-in functions
# ### They are faster, cleaner and easy to use
# ### You will save a lot of time

# ## Extra trick: A better way to time your functions...use a decorator!

def timer(func):
    # This function shows the execution time of
    # the function object passed
    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrapper


@timer
def compute_module(v):
    module = np.linalg.norm(v)
    return module


v = [1, 3, 4, 5, 6]
compute_module(v)

v = [1, 3, 4, 5, 6] * 2000
compute_module(v)
