import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy import integrate
import george
from george import kernels


class pklin_gp:
    def __init__(self):
        print('Initialize pklin emulator')
        self.klist = np.logspace(-3, 1, 200)
        self.logklist = np.log(self.klist)
        self.cosmos = np.loadtxt(os.path.dirname(
            os.path.abspath(__file__)) + '/../data/cparams_3d.dat')
        self.ydata = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/pklin/coeff_all.dat')
        self.eigdata = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/pklin/pca_eigvec.dat')
        self.ymean = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/pklin/pca_mean.dat')
        self.ystd = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/pklin/pca_std.dat')
        self.yavg = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/pklin/pca_avg.dat')
        self.gp_params = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/pklin/gp_params.dat')
        self.ktypes = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/pklin/ktypes.dat')
        self.gps = []
        for i in range(20):
            if self.ktypes[i] == 10:
                kernel = 1. * \
                    kernels.Matern52Kernel(
                        np.ones(3), ndim=3) + kernels.ConstantKernel(1e-4, ndim=3)
            elif self.ktypes[i] == 6:
                kernel = 1. * \
                    kernels.ExpSquaredKernel(
                        np.ones(3), ndim=3) + kernels.ConstantKernel(1e-4, ndim=3)
            else:
                print('kernel type 6 and 10 are the only supported types.')
            gp = george.GP(kernel)
            gp.compute(self.cosmos[:800])
            gp.set_parameter_vector(self.gp_params[i])
            self.gps.append(gp)

    def set_cosmology(self, cosmo):
        cparams = cosmo.get_cosmology()
        self.ns = cparams[0, 4]
        As = np.exp(cparams[0, 3])/1e10
        k0 = 0.05
        h = np.sqrt((cparams[0, 0] + cparams[0, 1] +
                     0.00064)/(1-cparams[0, 2]))
        gp_rec = np.array([self.gps[i].predict(self.ydata[:800, i], np.atleast_2d(
            cparams[0, :3]), return_cov=False)[0] for i in range(20)])
        tk_rec = np.exp(np.dot(gp_rec*self.ystd+self.yavg,
                               self.eigdata) + self.ymean)
        if not np.isclose(cparams[0, 5], -1):
            growth_wcdm = _linearGrowth(
                cparams[0, 2], cparams[0, 5], 0.)/_linearGrowth(cparams[0, 2], cparams[0, 5], 1000.)
            growth_lcdm = _linearGrowth(
                cparams[0, 2], -1., 0.)/_linearGrowth(cparams[0, 2], -1., 1000.)
            tk_rec *= growth_wcdm/growth_lcdm
        pzeta = (2.*np.pi**2)/self.klist**3 * As * \
            (self.klist/(k0/h))**(self.ns-1.)
        self.pred_table = np.log(pzeta * tk_rec**2)
        self.pklin_spl = ius(self.logklist, self.pred_table, ext=3)

    def _set_cosmology_param(self, cparams):
        self.ns = cparams[4]
        As = np.exp(cparams[3])/1e10
        k0 = 0.05
        h = np.sqrt((cparams[0] + cparams[1] + 0.00064)/(1-cparams[2]))
        gp_rec = np.array([self.gps[i].predict(self.ydata[:800, i], np.atleast_2d(
            cparams[:3]), return_cov=False)[0] for i in range(20)])
        tk_rec = np.exp(np.dot(gp_rec*self.ystd+self.yavg,
                               self.eigdata) + self.ymean)
        if not np.isclose(cparams[5], -1):
            growth_wcdm = _linearGrowth(
                cparams[2], cparams[5], 0.)/_linearGrowth(cparams[2], cparams[5], 1000.)
            growth_lcdm = _linearGrowth(
                cparams[2], -1., 0.)/_linearGrowth(cparams[2], -1., 1000.)
            tk_rec *= growth_wcdm/growth_lcdm
        pzeta = (2.*np.pi**2)/self.klist**3 * As * \
            (self.klist/(k0/h))**(self.ns-1.)
        self.pred_table = np.log(pzeta * tk_rec**2)
        self.pklin_spl = ius(self.logklist, self.pred_table, ext=3)

    def get(self, ks):
        w_small = ks < 0.001
        w_large = ks > 10.
        if (sum(w_small) > 0) and (sum(w_large) == 0):
            p_small_k = np.exp(
                self.pred_table[0]) * (ks[w_small]/np.exp(self.logklist[0]))**self.ns
            ww = ks >= 0.001
            p_large_k = np.exp(self.pklin_spl(np.log(ks[ww])))
            return np.hstack((p_small_k, p_large_k))
        elif (sum(w_small) > 0) and (sum(w_large) > 0):
            p_small_k = np.exp(
                self.pred_table[0]) * (ks[w_small]/np.exp(self.logklist[0]))**self.ns
            ww = (ks >= 0.001) * (ks <= 10.)
            p_mid_k = np.exp(self.pklin_spl(np.log(ks[ww])))
            slope = np.log(p_mid_k[-1]/p_mid_k[-2]) / \
                np.log(ks[ww][-1]/ks[ww][-2])
            norm = p_mid_k[-1]/ks[ww][-1]**slope
            p_large_k = norm * ks[w_large]**slope
            return np.hstack((p_small_k, p_mid_k, p_large_k))
        elif (sum(w_small) == 0) and (sum(w_large) > 0):
            ww = (ks <= 10.)
            p_mid_k = np.exp(self.pklin_spl(np.log(ks[ww])))
            slope = np.log(p_mid_k[-1]/p_mid_k[-2]) / \
                np.log(ks[ww][-1]/ks[ww][-2])
            norm = p_mid_k[-1]/ks[ww][-1]**slope
            p_large_k = norm * ks[w_large]**slope
            return np.hstack((p_mid_k, p_large_k))
        else:
            return np.exp(self.pklin_spl(np.log(ks)))


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
