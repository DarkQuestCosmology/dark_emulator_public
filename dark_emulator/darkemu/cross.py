import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RectBivariateSpline as rbs
from scipy import ndimage
from . import gp


class cross_gp:

    def __init__(self):
        print('initialize cross-correlation emulator')
        self.logrscale = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/xicross/loge_xs.npy')
        self.rscale = np.exp(self.logrscale)
        self.xdata = np.loadtxt(os.path.dirname(
            os.path.abspath(__file__)) + '/../data/params_80models.dat')
        self.ydata = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/xicross/coeff_all.npy')
        self.ymean = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/xicross/ymeans.npy')
        self.eigdata = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/xicross/pca_eigvec.npy')
        self.yerr = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/xicross/yerr.npy')
        self.logdens_list = np.linspace(-2.5, -8.5, 13)
        self.dens_list = 10**self.logdens_list
        self.ascale_list = np.loadtxt(os.path.dirname(
            os.path.abspath(__file__)) + '/../data/scales.dat')
        self.redshift_list = 1./self.ascale_list-1.
        self.gps = gp.gp6d(self.xdata, self.ydata, self.yerr, os.path.dirname(
            os.path.abspath(__file__))+'/../learned_data/xicross/gp6d')
        self.weight = (self.dens_list.reshape(13, 1) *
                       self.radial_weight(self.rscale).reshape(1, 66))

    def set_cosmology(self, cosmo):
        self.pred_table = (np.dot(self.gps.predict(np.atleast_2d(
            cosmo.get_cosmology())), self.eigdata) + self.ymean).reshape(13, 21, 66)
        for i in range(21):
            self.pred_table[:, i, :] = np.log(
                np.fmax(self.pred_table[:, i, :]/self.weight, 1e-10))

    def get(self, rs, redshift, logdens):
        xic = np.array([rbs(-self.logdens_list, -self.redshift_list,
                            self.pred_table[:, :, i])(-logdens, -redshift) for i in range(66)])
        return np.exp(ius(self.logrscale, xic, ext=3)(np.log(rs)))

    def getNoInterpol(self, redshift, logdens):
        xic = np.array([rbs(-self.logdens_list, -self.redshift_list, self.pred_table[:,
                                                                                     :, i])(-logdens, -redshift, grid=False) for i in range(66)])
        return np.exp(xic)

    def radial_weight(self, x):
        return x * (1-np.exp(-x**2))
