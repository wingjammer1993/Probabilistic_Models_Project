

from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO
import Part_1a
import numpy as np

cov = matern32()
gp = GaussianProcess(cov, optimize=True, usegrads=True)
acq = Acquisition(mode='ExpectedImprovement')
param = {'x': ('cont', [-2, 2]),
         'y': ('cont', [-2, 2])}

np.random.seed(20)
gpgo = GPGO(gp, acq, Part_1a.f, param)
gpgo.run(max_iter=20)
print(gpgo.getResult())