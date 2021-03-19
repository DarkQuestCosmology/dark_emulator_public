import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RectBivariateSpline as rbs
from .sigmad import sigmad_gp
from .import gp


class gamma1_gp:

    def __init__(self):
        print('initialize propagator emulator')
        self.xdata = np.loadtxt(os.path.dirname(
            os.path.abspath(__file__)) + '/../data/params_80models.dat')
        self.ydata = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/gamma1/coeff_all.npy')
        self.eigdata = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/gamma1/pca_eigvec.npy')
        self.yerr = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/gamma1/yerr.npy')
        self.gps = gp.gp6d(self.xdata, self.ydata, self.yerr, os.path.dirname(
            os.path.abspath(__file__))+'/../learned_data/gamma1/gp6d')

        self.ydata_dm = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/gamma1_dm/coeff_all.npy')
        self.eigdata_dm = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/gamma1_dm/pca_eigvec.npy')
        self.yerr_dm = np.load(os.path.dirname(os.path.abspath(
            __file__)) + '/../learned_data/gamma1_dm/yerr.npy')
        self.gps_dm = gp.gp6d(self.xdata, self.ydata_dm, self.yerr_dm, os.path.dirname(
            os.path.abspath(__file__))+'/../learned_data/gamma1_dm/gp6d')

        self.logdens_list = np.linspace(-2.5, -8.5, 13)
        self.ascale_list = np.loadtxt(os.path.dirname(
            os.path.abspath(__file__)) + '/../data/scales.dat')
        self.redshift_list = 1./self.ascale_list-1.

        self.sd = sigmad_gp()

    # def reload(self):
    # 	print 'reloading data files for gamma1_gp object...'
    # 	self.dout = np.load(os.path.dirname(__file__) + '/../learned_data/gamma1/gamma1_dm_dout.npy')
    # 	self.gp = joblib.load(os.path.dirname(__file__) + '/../learned_data/gamma1/gamma1_dm_fit_gps.pkl')
    # 	self.dout_h = np.load(os.path.dirname(__file__) + '/../learned_data/gamma1/gamma1_dout.npy')
    # 	self.gp_h = joblib.load(os.path.dirname(__file__) + '/../learned_data/gamma1/gamma1_fit_gps.pkl')

    def model_curve(x, a, b, c):
        return (a+b*x**2+c*x**4)*np.exp(-x**2/2.)

    def set_cosmology(self, cosmo):
        self.cosmo_now = cosmo
        cpara = cosmo.get_cosmology()
        self.coeff_rec = np.dot(self.gps.predict(
            np.atleast_2d(cpara)), self.eigdata).reshape(21, 13, 3)
        self.coeff_rec_dm = np.dot(self.gps_dm.predict(
            np.atleast_2d(cpara)), self.eigdata_dm).reshape(21, 2)
        self.sd0 = self.sd.get(cosmo)
        # self.D0s = np.array([cosmo.Dgrowth_from_z(z) for z in self.redshift_list])
        self.coeff1_spline = rbs(-self.redshift_list, -
                                 self.logdens_list, self.coeff_rec[:, :, 0])
        self.coeff2_spline = rbs(-self.redshift_list, -
                                 self.logdens_list, self.coeff_rec[:, :, 1])
        self.coeff3_spline = rbs(-self.redshift_list, -
                                 self.logdens_list, self.coeff_rec[:, :, 2])
        self.coeff2_spline_dm = ius(-self.redshift_list,
                                    self.coeff_rec_dm[:, 0])
        self.coeff3_spline_dm = ius(-self.redshift_list,
                                    self.coeff_rec_dm[:, 1])

    def get_bd(self, redshift, logdens):
        return self.coeff1_spline(-redshift, -logdens)

    def get_bias(self, redshift, logdens):
        D0 = self.cosmo_now.Dgrowth_from_z(redshift)
        return self.coeff1_spline(-redshift, -logdens)/D0

    def get(self, k, redshift, logdens):
        D0 = self.cosmo_now.Dgrowth_from_z(redshift)
        sig = D0 * self.sd0
        if isinstance(logdens, np.ndarray) or isinstance(logdens, list):
            g = np.zeros((len(logdens), len(k)))
            for i in range(len(logdens)):
                coeff = np.ravel(np.array([self.coeff1_spline(-redshift, -logdens[i]),
                                           self.coeff2_spline(-redshift, -logdens[i]), self.coeff3_spline(-redshift, -logdens[i])]))
                g[i] = self.model_curve(sig*k, *coeff)
        else:
            coeff = np.ravel(np.array([self.coeff1_spline(-redshift, -logdens),
                                       self.coeff2_spline(-redshift, -logdens), self.coeff3_spline(-redshift, -logdens)]))
            g = self.model_curve(sig*k, *coeff)
        return g

    def get_dm(self, k, redshift):
        D0 = self.cosmo_now.Dgrowth_from_z(redshift)
        sig = D0 * self.sd0
        coeff = np.ravel(np.array(
            [D0, self.coeff2_spline_dm(-redshift), self.coeff3_spline_dm(-redshift)]))
        return self.model_curve(sig*k, *coeff)

    def model_curve(self, x, a, b, c):
        return (a+b*x**2+c*x**4)*np.exp(-x**2/2.)

    def bias_ratio(self, redshift, logdens):
        return self.coeff1_spline(-redshift, -logdens)[0] / self.coeff1_spline(-redshift, 5.75)[0]

    def bias_ratio_arr(self, redshift, logdens):
        return self.coeff1_spline(-redshift, -logdens, grid=False) / self.coeff1_spline(-redshift, 5.75)[0]
