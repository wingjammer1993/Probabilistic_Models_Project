import os
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from pyGPGO.covfunc import matern52
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential
from pyGPGO.covfunc import rationalQuadratic
import Part_1a


def plot_f(x_values, y_values, f):
    z = np.zeros((len(x_values), len(y_values)))
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            z[i, j] = f(x_values[i], y_values[j])
    plt.imshow(z.T, origin='lower', extent=[np.min(x_values), np.max(x_values), np.min(y_values), np.max(y_values)])
    plt.colorbar()
    plt.show()


def plot_convergence(model, title_text):
    sampled_values = model.history
    best_iteration = []
    for i in range(1, len(sampled_values)):
        best_sofar = max(sampled_values[0:i])
        best_iteration.append(best_sofar)
    plt.xlabel("Iterations")
    plt.ylabel("Recovered Optimum")
    plt.title(title_text)
    plt.plot(best_iteration, 'r--')


def plot_acquisition(gpgo, param, index, colors, acq_titles, new=True):
    n = 100
    r_x, r_y = gpgo.parameter_range[0], gpgo.parameter_range[1]
    x_test = np.linspace(r_x[0], r_x[1], n)
    y_test = np.linspace(r_y[0], r_y[1], n)
    z_hat = np.empty((len(x_test), len(y_test)))
    z_var = np.empty((len(x_test), len(y_test)))
    ac = np.empty((len(x_test), len(y_test)))
    for i in range(len(x_test)):
        for j in range(len(y_test)):
            res = gpgo.GP.predict([x_test[i], y_test[j]])
            z_hat[i, j] = res[0]
            z_var[i, j] = res[1][0]
            ac[i, j] = -gpgo._acqWrapper(np.atleast_1d([x_test[i], y_test[j]]))
    plt.title(acq_titles)
    plt.imshow(ac.T, origin='lower', extent=[r_x[0], r_x[1], r_y[0], r_y[1]])
    plt.colorbar()
    gpgo._optimizeAcq(method='L-BFGS-B', n_start=500)
    plt.plot(gpgo.best[0], gpgo.best[1], 'gx', markersize=15)


def plot_figure():
    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-2, 2, 1000)
    plot_f(x, y, Part_1a.f)


def part_1(max_iter):
    # Plot the function
    param = OrderedDict()
    param['x'] = ('cont', [-2, 2])
    param['y'] = ('cont', [-2, 2])

    # squared exponential kernel function
    plt.suptitle("Convergence Rate, True Optimum = 0")
    np.random.seed(20)
    plt.subplot(131)
    sqexp = squaredExponential()
    gp = GaussianProcess(sqexp)
    acq = Acquisition(mode='ExpectedImprovement')
    gpgo = GPGO(gp, acq, Part_1a.f, param, n_jobs=-1)
    gpgo.run(max_iter=max_iter)
    plot_convergence(gpgo, "Squared Exponential Kernel")

    # matern52 kernel function
    np.random.seed(20)
    plt.subplot(132)
    matern = matern52()
    gp = GaussianProcess(matern)
    acq = Acquisition(mode='ExpectedImprovement')
    gpgo = GPGO(gp, acq, Part_1a.f, param, n_jobs=-1)
    gpgo.run(max_iter=max_iter)
    plot_convergence(gpgo, "Matern52 Kernel")

    # rational quadratic kernel function
    np.random.seed(20)
    plt.subplot(133)
    ratq = rationalQuadratic()
    gp = GaussianProcess(ratq)
    acq = Acquisition(mode='ExpectedImprovement')
    gpgo = GPGO(gp, acq, Part_1a.f, param, n_jobs=-1)
    gpgo.run(max_iter=max_iter)
    plot_convergence(gpgo, "Rational Quadratic Kernel")
    plt.show()


def part_2(max_iter):
    acq_1 = Acquisition(mode='ExpectedImprovement')
    acq_2 = Acquisition(mode='ProbabilityImprovement')
    acq_3 = Acquisition(mode='UCB', beta=0.5)
    acq_4 = Acquisition(mode='UCB', beta=1.5)
    acq_list = [acq_1, acq_2, acq_3, acq_4]
    sqexp = squaredExponential()
    param = OrderedDict()
    param['x'] = ('cont', [-2, 2])
    param['y'] = ('cont', [-2, 2])
    new = True
    colors = ['green', 'red', 'orange', 'black']
    acq_titles = ['Expected improvement', 'Probability of Improvement', 'GP-UCB, beta = .5',
                  'GP-UCB beta = 1.5']

    plt.suptitle('Acquisition Functions')
    idx = 0

    for index, acq in enumerate(acq_list):
        np.random.seed(200)
        gp = GaussianProcess(sqexp)
        gpgo = GPGO(gp, acq, Part_1a.f, param)
        gpgo.run(max_iter=max_iter)
        plt.subplot(4, 2, idx+1)
        plot_acquisition(gpgo, param, index + 2, colors, acq_titles[index], new=new)
        plt.subplot(4, 2, idx+2)
        plot_convergence(gpgo, acq_titles[index])
        new = False
        idx = idx + 2
    plt.show()


if __name__ == '__main__':
    max_iterations = 10
    # plot_figure()
    # part_1(max_iter=5)
    part_2(max_iterations)







