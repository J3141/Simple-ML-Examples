import numpy as np
from matplotlib import pyplot as plt

""" simple linear kernel machine """


# Specify kernel
def kernel(x, m, rho=0.1):
    return np.exp(-1/2*(x-m)**2/rho)  #SE Kernel with bandwith rho

def create_feature_vector(x, train_data):
    return np.hstack([kernel(x, m) for m in train_data])

# Draw x-coorinates uniformly distributed in [a,b]
a, b = -5, 5
nr_points = 100
regularization_constant = 1

def transform_to_interval(x, low_bound=-1, upper_bound=1):
    return (upper_bound - low_bound) * x + low_bound

x = transform_to_interval(np.random.random(nr_points),
                           low_bound=a, upper_bound=b)

x.sort()  # for plotting

# Draw y-coord. normally distributed with std rho and mean conditional_mean
rho = 1

def conditional_mean(x):
    return  10*x + 1

y = np.random.normal(conditional_mean(x), rho, nr_points)

# learning weights using Pseudo Inverse
X = x.reshape(-1, 1)
Y = y.reshape(-1, 1)

X = np.hstack([create_feature_vector(X, X), np.ones_like(X)])     # add biases
w = np.linalg.inv(X.T @ X + regularization_constant*np.eye(X.shape[1])) @ X.T  @ Y

# Prediction
x_true = np.linspace(a, b, 500).reshape(-1, 1)
y_true = conditional_mean(x_true)
prediction = (w[:-1].T
              @ create_feature_vector(x_true, x.reshape(-1, 1)).T + w[-1])

# Plot Solution
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(x, y, "o", label="$mu_{Data}$")
plt.plot(x_true, y_true, "-", label="$mu_{true}$")
plt.plot(x_true, prediction.T, "-", label="$mu_{pred.}$")
plt.legend()
#plt.axis([a, b, -10, 10])
plt.title("Kernel Machine")
plt.subplot(2, 1, 2)
plt.plot(x_true, kernel(x_true, -2), "-", label="$kernel(0)$")
plt.plot(x_true, kernel(x_true, -1), "-", label="$kernel(1)$")
plt.legend()
