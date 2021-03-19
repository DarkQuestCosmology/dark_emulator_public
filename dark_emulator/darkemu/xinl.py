import os
import numpy as np
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RectBivariateSpline as rbs
from . import gp


class xinl_gp:

    def __init__(self):
        print('initialize xinl emulator')
        # self.logxscale = np.log(np.load(os.path.dirname(os.path.abspath(__file__)) + '/../learned_data/ximnl/xs.npy'))
        self.xscale = np.logspace(-2, 2, 41)
        self.logxscale = np.log(self.xscale)
        self.xdata = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../data/params.dat')[list(range(1))+list(range(61, 101))][:, [0, 1, 2, 4, 5, 6]]
        self.ydata = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/xinl/coeff_all.npy')
        self.eigdata = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/xinl/pca_eigvec.npy')
        self.yerr = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/xinl/yerr.npy')
        self.ascale_list = np.loadtxt(os.path.dirname(
            os.path.abspath(__file__)) + '/../data/scales.dat')
        self.redshift_list = 1./self.ascale_list-1.
        self.gps = gp.gp6d(self.xdata, self.ydata, self.yerr, os.path.dirname(
            os.path.abspath(__file__))+'/../learned_data/xinl/gp6d')

    def set_cosmology(self, cosmo):
        coeff_rec = np.dot(self.gps.predict(np.atleast_2d(
            cosmo.get_cosmology())), self.eigdata).reshape(21, 41)

        #pred_table = []
        # for i in range(21):
        #	self.spl[1][:-4] = coeff_rec[i,]
        #	pred_table.append(interpolate.splev(self.logxscale, self.s1pl))
        self.xinl = rbs(-self.redshift_list, self.logxscale, coeff_rec)

    def get(self, xs, redshift):
        return self.xinl(-redshift, np.log(xs))[0]
