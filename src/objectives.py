import numpy as np


def rastrigin_function(x):
    x = np.asarray(x)
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])


def sphere_function(x):
    x = np.asarray(x)
    return np.sum(np.square(x))


def rosenbrock_function(x):
    x = np.asarray(x)
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
