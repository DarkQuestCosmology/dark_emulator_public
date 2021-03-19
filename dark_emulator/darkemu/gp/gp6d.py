import os
import sys
import numpy as np
import george
from george.modeling import Model
from george import kernels
from george.metrics import Metric
import scipy.optimize as op
import copy


class gp6d:
    def __init__(self, x, y, e, fbase):
        # print 'Initialize a 6D gaussian process'
        self.xbounds = ((0.0211375, 0.0233625), (0.10782, 0.13178), (0.54752,
                                                                     0.82128), (2.4752, 3.7128), (0.916275, 1.012725), (-1.2, -0.8))
        self.pass_all_data(x, y, e)
        if os.path.exists(fbase+'_kerneltype.npy'):
            self.k_type = np.load(fbase+'_kerneltype.npy')
        else:
            self.k_type = np.zeros(self.nparams)
        self.init_gps()
        self.load_hyperparams(fbase)

    def pass_all_data(self, x, y, e):
        self.nparams = y.shape[1]
        self.nmodels = y.shape[0]
        self.means = np.average(y, axis=0)
        self.yerr = copy.copy(e)
        self.xall = np.zeros(x.shape)
        for i in range(self.nmodels):
            for j in range(6):
                self.xall[i, j] = (x[i, j] - self.xbounds[j][0]) / \
                    (self.xbounds[j][1] - self.xbounds[j][0])
        self.yall = copy.copy(y.reshape(self.nmodels, self.nparams))
        for i in range(self.nmodels):
            self.yall[i, ] -= self.means
        self.logyvars = np.log(np.var(y, axis=0))

    def init_gps(self):
        self.gps = []
        for i in range(self.nparams):
            if self.k_type[i] == 0:
                kernel = kernels.ConstantKernel(
                    self.logyvars[i], ndim=6)*kernels.ExpSquaredKernel(metric=np.eye(6), ndim=6)
            elif self.k_type[i] == 1:
                kernel = kernels.ConstantKernel(
                    self.logyvars[i], ndim=6)*kernels.ExpKernel(metric=np.eye(6), ndim=6)
            elif self.k_type[i] == 2:
                kernel = kernels.ConstantKernel(
                    self.logyvars[i], ndim=6)*kernels.Matern32Kernel(metric=np.eye(6), ndim=6)
            elif self.k_type[i] == 3:
                kernel = kernels.ConstantKernel(
                    self.logyvars[i], ndim=6)*kernels.Matern52Kernel(metric=np.eye(6), ndim=6)
            tmp = copy.copy(george.GP(kernel))
            tmp.compute(self.xall, self.yerr[:, i])
            self.gps.append(tmp)

    def load_hyperparams(self, fbase):
        hyperparams = np.load(fbase+'_hyperparams.npy')
        for i in range(self.nparams):
            self.gps[i].set_parameter_vector(hyperparams[i, ])

    def predict(self, x_new, return_var=False):
        nmods = np.atleast_2d(x_new).shape[0]
        x_new_norm = np.zeros((nmods, 6))
        for i in range(nmods):
            for j in range(6):
                x_new_norm[i, j] = (
                    x_new[i, j] - self.xbounds[j][0]) / (self.xbounds[j][1] - self.xbounds[j][0])

        res = []
        var = []
        if return_var:
            for i in range(self.nparams):
                val, vari = self.gps[i].predict(
                    self.yall[:, i], x_new_norm, return_var=True)
                res.append(val)
                var.append(np.sqrt(vari))
            var = np.array(var)
        else:
            for i in range(self.nparams):
                val = self.gps[i].predict(
                    self.yall[:, i], x_new_norm, return_cov=False)
                res.append(val)

        res = np.array(res)
        for i in range(nmods):
            res[:, i] += self.means
        if return_var:
            return np.transpose(res), np.transpose(var)
        else:
            return np.transpose(res)
