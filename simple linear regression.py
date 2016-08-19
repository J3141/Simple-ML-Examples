import numpy as np
from matplotlib import pyplot as plt

""" plain simple linear regression """


""" Draw x-coorinates uniformly distributed in [a,b]"""
a, b = -1, 1
nr_points = 10

def transform_to_intervall(x, low_bound=-1, upper_bound=1):
    return (upper_bound - low_bound) * x + low_bound

x = transform_to_intervall(np.random.random(nr_points),
                           low_bound=-1, upper_bound=1)
x.sort()  # for plotting

""" Draw y-coorindates normally distributed with std rho
    and mean conditional_mean"""
rho = 0.3

def conditional_mean(x):
    return x**2 + x + 1

y = np.random.normal(conditional_mean(x), rho, nr_points)

""" learning weights using Pseudo Inverse to solve y=w_1x + w_0"""
X = x.reshape(-1, 1)
Y = y.reshape(-1, 1)

X = np.hstack([X, np.ones_like(X)])     # add biases
w = np.linalg.inv(X.T @ X) @ X.T  @ Y

""" Prediction """
x_true = np.linspace(a, b)
y_true = conditional_mean(x_true)
prediction = (w[0] @ x_true.reshape(1, -1) + w[1])

""" plot Solution """
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(x, y, "o", label="$mu_{Data}$")
plt.plot(x_true, y_true, "-", label="$mu_{true}$")
plt.plot(x_true, prediction, "-", label="$mu_{pred.}$")
plt.legend()
plt.title("Linear Regression")
