import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RectBivariateSpline as rbs
from scipy import ndimage
from . import gp


class auto_gp:

    def __init__(self):
        print('initialize auto-correlation emulator')
        self.logrscale = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/xiauto/loge_xs.npy')
        self.xdata = np.loadtxt(os.path.dirname(
            os.path.abspath(__file__)) + '/../data/params_80models.dat')
        self.ydata = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/xiauto/coeff_all.npy')
        self.eigdata = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/xiauto/pca_eigvec.npy')
        self.yerr = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/xiauto/yerr.npy')
        self.logdens_list = np.linspace(-2.5, -6, 8)
        self.ascale_list = np.loadtxt(os.path.dirname(
            os.path.abspath(__file__)) + '/../data/scales.dat')
        self.redshift_list = 1./self.ascale_list-1.
        self.gps = gp.gp6d(self.xdata, self.ydata, self.yerr, os.path.dirname(
            os.path.abspath(__file__))+'/../learned_data/xiauto/gp6d')

    def dist_to_index(self, r):
        return ius(self.logrscale, list(range(21)), k=1)(np.log(r))

    def logdens_to_index(self, logdens):
        return ius(-self.logdens_list, list(range(8)))(-logdens)

    def redshift_to_index(self, redshift):
        return ius(-self.redshift_list, list(range(21)))(-redshift)

    def set_cosmology(self, cosmo):
        pred_table = np.dot(self.gps.predict(np.atleast_2d(
            cosmo.get_cosmology())), self.eigdata).reshape(36, 21, 21)

        self.xih_mat = np.zeros((8, 8, 21, 21))
        n = 0
        for bini in range(8):
            for binj in range(bini, 8):
                self.xih_mat[bini, binj, :, :] = pred_table[n, :, :]
                n += 1
        for bini in range(1, 8):
            for binj in range(bini):
                self.xih_mat[bini, binj, :, :] = self.xih_mat[binj, bini, :, :]

    def get(self, rs, redshift, logdens1, logdens2):
        sindex = self.redshift_to_index(redshift)
        # dindex1 = self.logdens_to_index(logdens1)
        # dindex2 = self.logdens_to_index(logdens2)
        # ins = [dindex1*np.ones(21),dindex2*np.ones(21),sindex*np.ones(21),range(21)]
        if sindex <= 0:
            s0 = 0
            xia0 = np.array([rbs(-self.logdens_list, -self.logdens_list,
                                 self.xih_mat[:, :, s0, i])(-logdens1, -logdens2) for i in range(21)])
            return ius(self.logrscale, xia0, ext=3)(np.log(rs))
        elif sindex >= 20:
            s0 = 20
            xia0 = np.array([rbs(-self.logdens_list, -self.logdens_list,
                                 self.xih_mat[:, :, s0, i])(-logdens1, -logdens2) for i in range(21)])
            return ius(self.logrscale, xia0, ext=3)(np.log(rs))
        else:
            s0 = int(sindex)
            s1 = s0+1
            u = sindex - s0
            xia0 = np.array([rbs(-self.logdens_list, -self.logdens_list,
                                 self.xih_mat[:, :, s0, i])(-logdens1, -logdens2) for i in range(21)])
            xia1 = np.array([rbs(-self.logdens_list, -self.logdens_list,
                                 self.xih_mat[:, :, s1, i])(-logdens1, -logdens2) for i in range(21)])
            xia = (1-u) * xia0 + u * xia1
            return ius(self.logrscale, xia, ext=3)(np.log(rs))

    def getNoInterpol(self, redshift, logdens1, logdens2):
        sindex = self.redshift_to_index(redshift)
        # dindex1 = self.logdens_to_index(logdens1)
        # dindex2 = self.logdens_to_index(logdens2)
        # ins = [dindex1*np.ones(21),dindex2*np.ones(21),sindex*np.ones(21),range(21)]
        if sindex <= 0:
            s0 = 0
            xia0 = np.array([rbs(-self.logdens_list, -self.logdens_list,
                                 self.xih_mat[:, :, s0, i])(-logdens1, -logdens2) for i in range(21)])
            return xua0
        elif sindex >= 20:
            s0 = 20
            xia0 = np.array([rbs(-self.logdens_list, -self.logdens_list,
                                 self.xih_mat[:, :, s0, i])(-logdens1, -logdens2) for i in range(21)])
            return xia0
        else:
            s0 = int(sindex)
            s1 = s0+1
            u = sindex - s0
            xia0 = np.array([rbs(-self.logdens_list, -self.logdens_list, self.xih_mat[:,
                                                                                      :, s0, i])(-logdens1, -logdens2, grid=False) for i in range(21)])
            xia1 = np.array([rbs(-self.logdens_list, -self.logdens_list, self.xih_mat[:,
                                                                                      :, s1, i])(-logdens1, -logdens2, grid=False) for i in range(21)])
            xia = (1-u) * xia0 + u * xia1
            return xia
