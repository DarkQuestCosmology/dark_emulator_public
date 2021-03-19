import os
import numpy as np
import george
from george import kernels
from scipy import integrate


class sigmad_gp:
    def __init__(self):
        print('Initialize sigma_d emulator')
        self.cosmos = np.loadtxt(os.path.dirname(
            os.path.abspath(__file__)) + '/../data/cparams_4d.dat')
        self.ydata = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/sigmad/coeff_all.dat')
        self.yavg = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/sigmad/sigd_avg.dat')
        self.ystd = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/sigmad/sigd_std.dat')
        self.gp_params = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/sigmad/gp_params.dat')
        self.ktypes = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/sigmad/ktypes.dat')
        if self.ktypes == 10:
            kernel = 1. * \
                kernels.Matern52Kernel(np.ones(4), ndim=4) + \
                kernels.ConstantKernel(1e-4, ndim=4)
        elif self.ktypes == 6:
            kernel = 1. * \
                kernels.ExpSquaredKernel(
                    np.ones(4), ndim=4) + kernels.ConstantKernel(1e-4, ndim=4)
        else:
            print('kernel type 6 and 10 are the only supported types.')
        self.gp = george.GP(kernel)
        self.gp.compute(self.cosmos[:800])
        self.gp.set_parameter_vector(self.gp_params)
        self.As_fid = np.exp(3.094)

    def get(self, cosmo):
        cparams = cosmo.get_cosmology()[0]
        if not np.isclose(cparams[5], -1):
            growth_wcdm = _linearGrowth(
                cparams[2], cparams[5], 0.)/_linearGrowth(cparams[2], cparams[5], 1000.)
            growth_lcdm = _linearGrowth(
                cparams[2], -1., 0.)/_linearGrowth(cparams[2], -1., 1000.)
            return growth_wcdm/growth_lcdm * np.sqrt(np.exp(cparams[3]) / self.As_fid) * (self.ystd*self.gp.predict(self.ydata[:800], np.atleast_2d(cparams)[:, [0, 1, 2, 4]], return_cov=False)[0]+self.yavg)
        else:
            return np.sqrt(np.exp(cparams[3]) / self.As_fid) * (self.ystd*self.gp.predict(self.ydata[:800], np.atleast_2d(cparams)[:, [0, 1, 2, 4]], return_cov=False)[0]+self.yavg)

    def _get_params(self, cparams):
        if not np.isclose(cparams[5], -1):
            growth_wcdm = _linearGrowth(
                cparams[2], cparams[5], 0.)/_linearGrowth(cparams[2], cparams[5], 1000.)
            growth_lcdm = _linearGrowth(
                cparams[2], -1., 0.)/_linearGrowth(cparams[2], -1., 1000.)
            return growth_wcdm/growth_lcdm * np.sqrt(np.exp(cparams[3]) / self.As_fid) * (self.ystd*self.gp.predict(self.ydata[:800], np.atleast_2d(cparams)[:, [0, 1, 2, 4]], return_cov=False)[0]+self.yavg)
        else:
            return np.sqrt(np.exp(cparams[3]) / self.As_fid) * (self.ystd*self.gp.predict(self.ydata[:800], np.atleast_2d(cparams)[:, [0, 1, 2, 4]], return_cov=False)[0]+self.yavg)


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
