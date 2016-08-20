# -*- coding: utf-8 -*-
"""
Naive Bayes Classification
"""
import numpy as np
from matplotlib import pyplot as plt

# Generate points of Class 1 
nr_points_0 = 100
mean_0 = np.array([1, 1])
Cov_0  = np.diag([0.001, 1])
x_0 = np.random.multivariate_normal(mean_0, Cov_0, (nr_points_0))

# Generate points of Class 2
nr_points_1 = 100
mean_1 = np.array([1, 1])
Cov_1  = np.diag ([1, 0.001])
x_1 = np.random.multivariate_normal(mean_1, Cov_1, (nr_points_1))

plt.plot(x_0[:, 0], x_0[:, 1], "ro", label="0")
plt.plot(x_1[:, 0], x_1[:, 1], "bo", label="1")


def gaussian_pdf_1d(x, mu=0 , cov=1):
    return 1 / (np.sqrt( 2*np.pi * cov)) * np.exp(-0.5 * (x - mu)**2 / cov)
    
    
# Naive Bayes
def prior(y):
    "assume x ~ Bernouli"
    total = nr_points_0 + nr_points_1
    likleyhood_class_1 = nr_points_1/total
    return likleyhood_class_1 * y + (1-likleyhood_class_1) * (1-y)

def gaussian_conditional_likelyhood(x, y):
    if y==0:
        #sample mean
        mean_0 = x_0[:,0].mean()
        mean_1 = x_0[:,1].mean()
        #sample Covariance 
        var_0 = x_0[:,0].var()  #is this the biased or unbiased?
        var_1 = x_0[:,1].var()
    else:
        #sample mean
        mean_0 = x_1[:,0].mean()
        mean_1 = x_1[:,1].mean()
        #sample Covariance 
        var_0 = x_1[:,0].var()  #is this the biased or unbiased?
        var_1 = x_1[:,1].var()
    return np.array([gaussian_pdf_1d(x[0], mean_0, var_0), gaussian_pdf_1d(x[1], mean_1, var_1)])
    
def predict(x=[0,0]):
    c_0 = prior(0)*gaussian_conditional_likelyhood(x, 0).prod(axis=0)   
    c_1 = prior(1)*gaussian_conditional_likelyhood(x, 1).prod(axis=0)  
    return np.argmax([c_0, c_1], axis = 0)
    
#Tests
test_0 =  np.random.uniform(x_0.min(),x_0.max(),500) 
test_1 =  np.random.uniform(x_1.min(),x_1.max(),500)

test = np.vstack([test_0, test_1])

test_p = predict(test)

index_0 = np.where(test_p == 0)
index_1 = np.where(test_p == 1)

plt.plot(test_0[index_0], test_1[index_0], "rx", label ="pred. 0")
plt.plot(test_0[index_1], test_1[index_1], "bx", label ="pred. 1")
plt.legend()

def gaussian_pdf_2d(x, mu=np.array([0,0]) , cov=np.diag([1,1])):
    return 1 / (np.sqrt( (2*np.pi)**2 * np.linalg.det(cov))) * np.exp(-0.5 * (x - mu) @ np.linalg.inv(cov) @ (x - mu).T)