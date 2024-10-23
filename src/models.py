import numpy as np

def create_polynomial_features(x, degree=2):
    """Creates the polynomial features
    Args:
        x: A array for the data
        degree (int, optional): A integer for the degree
        of the generated polynomial function
    """
    x_mem = [] 
    for x_sub in x.T:
        x_sub = x_sub.T 
        x_new = x_sub
        for d in range(2, degree+1):
            x_new = np.c_[x_new, np.power(x_sub, d)]
        x_mem.extend(x_new.T)
    return np.c_[x_mem].T 