# STANDARD IMPORTS
import numpy as np
from scipy import optimize
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from scipy.stats import beta
from types import SimpleNamespace

# QUESTION 1 #
# Defining equations
def u(z, par):
    """ Utility function
    args:
        z (float): Input in utility function
    return:
        u (float): Utility
    """
    return (z**(1.+ par.theta))/(1.+ par.theta)

def v(q, x, par):
    """ Expected value of ensured agent
    args:
        q (float): Insurance coverage
        x (float): Monetary loss
        par (namespace): Parameters
    return:
        v (ndarray): Expected value
    """
    return par.p*u(par.y-x, par)+(1-par.p)*u(par.y, par)

# QUESTION 2 #
# Defining equations
def v_q2(q, pi, x, par):
    """ Expected value of ensured agent
    args:
        q (float): Insurance coverage
        pi (float): Insurance premium
    return:
        v_q2 (ndarray): Expected value of ensure agent
    """
    return par.p*u(par.y-x+q-pi, par)+(1-par.p)*u(par.y-pi, par)

def v_0(x, par):
    """ Expected value in case of no insurance 
    args:
        x (float): Monetary loss
        par (namespace): Parameters
    return:
        v_0 (float): expected value of no insurance
    """
    return par.p*u(par.y-x, par)+(1-par.p)*u(par.y, par)

def diff(q, pi, x, par):
    """ difference in utilty between ensured and unensured agents
    args:
        q (float): Insurance coverage
        pi (float): Insurance premium
    return:
        diff (ndarray): Difference in utilty between ensured and unensured agents
    """
    return np.absolute(v_q2(q, pi, x, par) - v_0(x, par))

# QUESTION 3 #
# Defining equations
def pi_q3(par):
    """ Helping function for optimizing insurance premium
    args:
        par (namespace): parameters
    return:
        pi_q3 (ndarray): Insurance premium
    """
    return par.p*gamma*par.x_beta

def montecarlo(gamma, pi_q3, par):
    """ The agents utility level with x drawn from beta distribution 
    args:
        gamma (float): Percentage coverage
        pi_q3 (ndarray): Help function
        par (namespace): parameters
    return:
        np.mean (float): output value
    """
    return np.mean(u(par.y-(1-gamma)*par.x_beta-pi_q3, par))

# QUESTION 4 #
# Defining equations
def montecarlo_insurance(pi, par):
    """ The agents utility given insurance
    args:
        pi (ndarray): Insurance premium
    return:
        monetecarlo_insurance (float): Montecarlo estimation
    """
    
    # The mean value of insurance for the agent
    return np.mean(u(par.y-(1-par.gamma)*par.x_beta-pi,par))
    
def montecarlo_noinsurance(par):
    """ The agents utility given no insurance
    args:
       par (namespace): parameters
    return:
        montecarlo_noinsurance (float): Montecarlo estimation
    """
    # The utility of the agent given no profit the firm and the agent has the whole burden of the loss
    return np.mean(u(par.y-par.x_beta, par))