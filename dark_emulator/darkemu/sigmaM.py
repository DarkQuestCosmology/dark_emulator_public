import os
import numpy as np
import george
from george import kernels
from scipy import integrate

from scipy.interpolate import InterpolatedUnivariateSpline as ius


class sigmaM_gp:
    def __init__(self):
        print('Initialize sigmaM emulator')
        self.Mlist = np.logspace(10, 16, 601)
        self.load_sigma_gp()
        self.As_fid = np.exp(3.094)

    def load_sigma_gp(self):
        self.cosmos = np.loadtxt(os.path.dirname(
            os.path.abspath(__file__)) + '/../data/cparams_4d.dat')
        self.ydata = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/sigmaM/coeff_all.dat')
        self.eigdata = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/sigmaM/pca_eigvec.dat')
        self.ymean = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/sigmaM/pca_mean.dat')
        self.ystd = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/sigmaM/pca_std.dat')
        self.yavg = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/sigmaM/pca_avg.dat')
        self.gp_params = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/sigmaM/gp_params.dat')
        self.ktypes = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/sigmaM/ktypes.dat')
        self.gps = []
        for i in range(4):
            if self.ktypes[i] == 10:
                kernel = 1. * \
                    kernels.Matern52Kernel(
                        np.ones(4), ndim=4) + kernels.ConstantKernel(1e-4, ndim=4)
            elif self.ktypes[i] == 6:
                kernel = 1. * \
                    kernels.ExpSquaredKernel(
                        np.ones(4), ndim=4) + kernels.ConstantKernel(1e-4, ndim=4)
            else:
                print('kernel type 6 and 10 are the only supported types.')
            gp = george.GP(kernel)
            gp.compute(self.cosmos[:800])
            gp.set_parameter_vector(self.gp_params[i])
            self.gps.append(gp)

    def set_cosmology(self, cosmo):
        cparams = cosmo.get_cosmology()
        gp_rec = np.array([self.gps[i].predict(self.ydata[:800, i], np.atleast_2d(
            cparams[0, [0, 1, 2, 4]]), return_cov=False)[0] for i in range(4)])
        self.pred_table = np.sqrt(np.exp(cparams[0, 3]) / self.As_fid * np.exp(
            np.dot(gp_rec*self.ystd+self.yavg, self.eigdata) + self.ymean))
        if not np.isclose(cparams[0, 5], -1):
            growth_wcdm = _linearGrowth(
                cparams[0, 2], cparams[0, 5], 0.)/_linearGrowth(cparams[0, 2], cparams[0, 5], 1000.)
            growth_lcdm = _linearGrowth(
                cparams[0, 2], -1., 0.)/_linearGrowth(cparams[0, 2], -1., 1000.)
            self.pred_table *= growth_wcdm/growth_lcdm
        self.sigma_M = ius(self.Mlist, self.pred_table)

    def _set_cosmology_param(self, cparams):
        gp_rec = np.array([self.gps[i].predict(self.ydata[:800, i], np.atleast_2d(
            cparams[[0, 1, 2, 4]]), return_cov=False)[0] for i in range(4)])
        self.pred_table = np.sqrt(np.exp(cparams[3]) / self.As_fid * np.exp(
            np.dot(gp_rec*self.ystd+self.yavg, self.eigdata) + self.ymean))
        if not np.isclose(cparams[5], -1):
            growth_wcdm = _linearGrowth(
                cparams[2], cparams[5], 0.)/_linearGrowth(cparams[2], cparams[5], 1000.)
            growth_lcdm = _linearGrowth(
                cparams[2], -1., 0.)/_linearGrowth(cparams[2], -1., 1000.)
            self.pred_table *= growth_wcdm/growth_lcdm
        self.sigma_M = ius(self.Mlist, self.pred_table)

    def get_sigma(self, M):
        return self.sigma_M(M)

    def get_sigma_in_default_binning(self):
        return self.pred_table[200:]


def _linearGrowth(Ode, wde, z):
    Om = 1 - Ode
    a_scale = 1./(1.+z)
    alpha = -1./(3.*wde)
    beta = (wde-1.)/(2.*wde)
    gamma = 1.-5./(6.*wde)
    x = -Ode/Om * a_scale**(-3.*wde)
    res = integrate.quad(lambda t: t**(beta-1.)*(1.-t) **
                         (gamma-beta-1.)*(1.-t*x)**(-alpha), 0, 1.)
    return a_scale * res[0]
