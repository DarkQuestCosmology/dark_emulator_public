import os
import numpy as np
# import sklearn.decomposition
# from sklearn.gaussian_process import GaussianProcess
# import george
# from sklearn.externals import joblib
from scipy import integrate
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from .sigmaM import sigmaM_gp
from . import cosmo_util
from . import gp


class hmf_gp:

    rho_cr = 2.77536627e11
    zprev = -1
    sigs0 = np.zeros((401))

    def __init__(self):
        self.mbin_min = np.logspace(12, 16, 81)[0:80]
        self.mbin_max = np.logspace(12, 16, 81)[1:81]
        self.mbin_cen = np.sqrt(self.mbin_min*self.mbin_max)
        self.Mlist = np.logspace(12, 16, 401)
        self.ascale_list = np.loadtxt(os.path.dirname(
            os.path.abspath(__file__)) + '/../data/scales.dat')
        self.redshift_list = 1./self.ascale_list-1.
        self.sM = sigmaM_gp()
        self.xdata = np.loadtxt(os.path.dirname(
            os.path.abspath(__file__)) + '/../data/params_80models.dat')
        self.ydata = np.load(os.path.dirname(os.path.abspath(
            __file__))+'/../learned_data/hmf/coeff_all.npy')
        self.eigdata = np.load(os.path.dirname(os.path.abspath(
            __file__))+'/../learned_data/hmf/pca_eigvec.npy')
        self.ymean = np.load(os.path.dirname(os.path.abspath(
            __file__))+'/../learned_data/hmf/ymeans.npy')
        self.yerr = np.load(os.path.dirname(os.path.abspath(
            __file__))+'/../learned_data/hmf/yerr.npy')
        self.gps = gp.gp6d(self.xdata, self.ydata, self.yerr, os.path.dirname(
            os.path.abspath(__file__))+'/../learned_data/hmf/gp6d')

    def f_HMF_func(self, x, z):
        alpha = 10**(-(0.75/(np.log10(200/75.)))**1.2)
        b = 2.57 * (1+z)**(-alpha)
        c = 1.19
        A = self.coeff_Anorm_spl(-z)
        a = self.coeff_a_spl(-z)
        return A * ((x/b)**(-a) + 1) * np.exp(-c/x**2)

    def f_TINK(self, x, z):
        alpha = 10**(-(0.75/(np.log10(200/75.)))**1.2)
        A = 0.186 * (1+z)**(-0.14)
        a = 1.47 * (1+z)**(-0.06)
        b = 2.57 * (1+z)**(-alpha)
        c = 1.19
        return A * ((x/b)**(-a) + 1) * np.exp(-c/x**2)

    def set_cosmology(self, cosmo):
        self.cosmo_now = cosmo
        params = cosmo.get_cosmology()
        self.sM.set_cosmology(cosmo)
        self.sigs0 = self.sM.get_sigma_in_default_binning()

        Om = 1-params[0, 2]
        dlnsdlnm = np.gradient(np.log(self.sigs0)) / \
            np.gradient(np.log(self.Mlist))
        self.f_to_num = - (Om*hmf_gp.rho_cr) * dlnsdlnm / self.Mlist**2

        flag, params_out = cosmo_util.test_cosm_range(
            params, return_edges=True)
        if flag:
            self.coeff_table = (np.dot(self.gps.predict(np.atleast_2d(
                params_out)), self.eigdata) + self.ymean).reshape(21, 2)
        else:
            self.coeff_table = (np.dot(self.gps.predict(np.atleast_2d(
                params)), self.eigdata) + self.ymean).reshape(21, 2)

        self.coeff_table[:, 0] /= 10
        self.coeff_Anorm_spl = ius(-self.redshift_list, self.coeff_table[:, 0])
        self.coeff_a_spl = ius(-self.redshift_list, self.coeff_table[:, 1])

    def get_dndM(self, redshift):
        D0 = self.cosmo_now.Dgrowth_from_z(redshift)
        return self.f_to_num * self.f_HMF_func(D0*self.sigs0, redshift)

    def get_dndM_tinker(self, redshift):
        D0 = self.cosmo_now.Dgrowth_from_z(redshift)
        alpha = 10**(-(0.75/(np.log10(200/75.)))**1.2)
        A = 0.186 * (1+redshift)**(-0.14)
        a = 1.47 * (1+redshift)**(-0.06)
        b = 2.57 * (1+redshift)**(-alpha)
        c = 1.19
        return self.f_to_num * self.f_TINK(D0*self.sigs0, redshift)

    def mass_to_dens(self, mass_thre, redshift, integration="quad"):
        dndM = self.get_dndM(redshift)
        dndM_interp = ius(np.log(self.Mlist), np.log(dndM))
        if integration == "quad":
            dens = integrate.quad(lambda t: np.exp(
                dndM_interp(np.log(t))), mass_thre, 1e16, epsabs=1e-5)[0]
        elif integration == "trapz":
            t = np.linspace(mass_thre, 1e16, 4096)
            dt = t[1]-t[0]
            dens = integrate.trapz(np.exp(dndM_interp(np.log(t))), dx=dt)
        else:
            raise RuntimeError(
                "You should specify valid integration algorithm: quad or trapz")
        return dens

    def dens_to_mass(self, dens, redshift, nint=20, integration="quad"):
        mlist = np.linspace(12., 15.95, nint)
        dlist = np.log(np.array([self.mass_to_dens(
            10**mlist[i], redshift, integration=integration) for i in range(nint)]))
        d_to_m_interp = ius(-dlist, mlist)
        return 10**d_to_m_interp(-np.log(dens))

    def get_nhalo(self, Mmin, Mmax, vol, redshift):
        if Mmax > 1e16:
            Mmax = 1e16
        dndM = self.get_dndM(redshift)
        dndM_interp = ius(np.log(self.Mlist), np.log(dndM))
        return vol * integrate.quad(lambda t: np.exp(dndM_interp(np.log(t))), Mmin, Mmax, epsabs=1e-5)[0]

    def get_nhalo_tinker(self, Mmin, Mmax, vol, redshift):
        if Mmax > 1e16:
            Mmax = 1e16
        dndM = self.get_dndM_tinker(redshift)
        dndM_interp = ius(np.log(self.Mlist), np.log(dndM))
        return vol * integrate.quad(lambda t: np.exp(dndM_interp(np.log(t))), Mmin, Mmax, epsabs=1e-5)[0]

    def get_nhalos_tinker(self, Mmins, Mmaxs, vol):
        dd = self.get_dndM_tinker()
        dndM_interp_tink = interpolate.interp1d(
            np.log10(dd[0, ]), np.log10(dd[1, ]), kind='cubic')
        nbin = Mmins.size
        nhs = vol * np.array([integrate.quad(lambda t: 10**dndM_interp_tink(
            np.log10(t)), Mmins[i], Mmaxs[i], epsabs=1e-4)[0] for i in range(nbin)])
        return nhs
