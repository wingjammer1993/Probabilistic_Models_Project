from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.cm import *
import numpy as np

def f(x, y):
    # Three camel humps function inverse
    one = 2*x**2 - 1.05*x**4
    two = x**6/6
    three = x*y
    four = y**2
    return -(one + two + three + four) + np.random.normal(0, 0.1)


def plot_func():
    x = np.linspace(2, -2, num=1000)
    y = np.linspace(2, -2, num=1000)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()



