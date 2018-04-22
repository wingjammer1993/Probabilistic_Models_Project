from bayes_opt import BayesianOptimization
from mpl_toolkits.mplot3d import axes3d, Axes3D
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
sys.path.append("./")


def target(x, y):
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    sigma_1 = np.array([[1., -0.5], [-0.5, 1.5]])
    sigma_2 = np.array([[1., 0.5], [0.5, 1.5]])
    norm_1 = multivariate_normal([8, 0], sigma_1)
    norm_2 = multivariate_normal([2, 2], sigma_2)
    z_1 = norm_1.pdf(pos)
    z_2 = norm_2.pdf(pos)
    z = np.add(z_1, z_2)
    for i in range(0, z.shape[0]):
        for j in range(0, z.shape[1]):
            z[i][j] = z[i][j] + 0.01*x[i][j] - 0.01*y[i][j] + np.random.normal(0, 0.005)
    return z


if __name__ == "__main__":
    #fig = plt.figure()
    #ax = Axes3D(fig)
    x_input = np.linspace(0, 10, 1000)
    y_input = np.linspace(-5, 5, 1000)
    x_mesh, y_mesh = np.meshgrid(x_input, y_input)
    z_value = target(x_mesh, y_mesh)
    plt.contourf(x_mesh, y_mesh, z_value)
    plt.show()
