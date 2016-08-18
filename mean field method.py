import numpy as np
from matplotlib import pyplot as plt

""" 
    Mean field method for Distribution on {0, 1}^2
    Approximation with two independent Bernouli rvs 
"""

#Initial values for means
mu_1 = 0.5
mu_2 = 0.5

#lists that accumulate the updated means
v_1 = [mu_1]
v_2 = [mu_2]

#number of iterations
l =10

#choose experiment: dependent or independent
experiment = "independent"

if experiment == "dependent":
    p_00, p_01, p_10, p_11  = 1/8, 1, 1/4, 5/8   #proportional to joint probability distribution on {0, 1}^2

elif experiment == "independent":
    theta_1, theta_2= 0.7, 0.1  #parameters of marginal bernouli distributions
    p_00, p_01, p_10, p_11  =  (1 - theta_1)*(1 - theta_2), (1 - theta_1)*theta_2, theta_1*(1 - theta_2), theta_1*theta_2 


for _ in range(l):
    #Normalising Constants
    z_1  = np.exp(mu_2 * np.log(p_11) + (1 - mu_2) * np.log(p_10) ) +  np.exp(mu_2 * np.log(p_01) + (1 - mu_2) * np.log(p_00))  
    z_2  = np.exp(mu_1 * np.log(p_11) + (1 - mu_1) * np.log(p_01) ) +  np.exp(mu_1 * np.log(p_10) + (1 - mu_1) * np.log(p_00)) 
    #Mean updates
    mu_1 = np.exp(mu_2 * np.log(p_11) + (1 - mu_2) * np.log(p_10)) / z_1
    mu_2 = np.exp(mu_1 * np.log(p_11) + (1 - mu_1) * np.log(p_01)) / z_2

    v_1 = v_1 + [mu_1]
    v_2 = v_2 + [mu_2]   

#Plot updated means    
x = np.arange(l+1)
plt.plot(x,v_1,"x-", label = "mu_1")
plt.plot(x,v_2,"x-", label = "mu_2")
plt.legend()

#Compare approx. and true Distributions
true_dist   = np.array([p_00, p_01, p_10, p_11])
approx_dist = np.array([(1 - v_1[-1]) * (1 - v_2[-1]), (1 - v_1[-1]) * v_2[-1], v_1[-1] * (1 - v_2[-1]), v_1[-1] * v_2[-1]])
print(np.abs(true_dist - approx_dist))

    