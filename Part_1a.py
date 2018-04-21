from bayes_opt import BayesianOptimization
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("./")


def target(x, y):
    z_val = np.zeros((len(x), len(y)))
    mean_1 = np.random.uniform()
    mean_2 = np.random.uniform()
    for idx, xval in enumerate(x):
        for idy, yval in enumerate(y):
            func = 2 * xval + 4 * yval + np.random.normal(mean_1, 0.1) + np.random.normal(mean_2, 0.5)
            z_val[idx][idy] = func + np.random.normal(0, 0.1)  # add some gaussian noise
    return z_val


if __name__ == "__main__":
    # fig = plt.figure()
    # ax = Axes3D(fig)
    x_input = np.linspace(-5, 5, 100).reshape(-1, 1)
    y_input = np.linspace(-5, 5, 100).reshape(-1, 1)
    X, Y = np.meshgrid(x_input, y_input)
    z_value = target(x_input, y_input)
    plt.contourf(X, Y, z_value)
    plt.show()
