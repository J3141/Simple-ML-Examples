import numpy as np
from matplotlib import pyplot as plt

""" plain simple linear regression with bases function extensiony """


""" Draw x-coorinates uniformly distributed in [a,b]"""
a, b = -4, 1
nr_points = 20

def bases_function_expansion(x):
    return np.hstack([x, x**2, x**3, x**4])

def transform_to_intervall(x, low_bound=-1, upper_bound=1):
    return (upper_bound - low_bound) * x + low_bound

x = transform_to_intervall(np.random.random(nr_points),
                           low_bound=a, upper_bound=b)

x.sort()  # for plotting

""" Draw y-coordinates normally distributed with std rho
    and mean conditional_mean"""
rho = 5

def conditional_mean(x):
    return x**3 - x**2 - 10*x + 1

y = np.random.normal(conditional_mean(x), rho, nr_points)

""" learning weights using Pseudo Inverse"""
X = x.reshape(-1, 1)
Y = y.reshape(-1, 1)

X = np.hstack([bases_function_expansion(X), np.ones_like(X)])     # add biases
w = np.linalg.inv(X.T @ X) @ X.T  @ Y

""" Prediction """
x_true = np.linspace(a, b, 100).reshape(-1, 1)
y_true = conditional_mean(x_true)
prediction = (w[:-1].T @ bases_function_expansion(x_true).T + w[-1])
 
""" plot Solution """
plt.plot(x, y, "o", label="$mu_{Data}$")
plt.plot(x_true, y_true, "-", label="$mu_{true}$")
plt.plot(x_true, prediction.T, "-", label="$mu_{pred.}$")
plt.legend()
plt.title("Linear Regression")
