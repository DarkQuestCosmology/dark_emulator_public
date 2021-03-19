import os
import numpy as np
from scipy import integrate
from scipy.misc import derivative
from collections import OrderedDict
import logging

cosm_range = OrderedDict((["omegab", [0.0211375, 0.0233625]],
                          ["omegam", [0.10782, 0.13178]],
                          ["Omagel", [0.54752, 0.82128]],
                          ["ln(10^10As)", [2.4752, 3.7128]],
                          ["ns", [0.916275, 1.012725]],
                          ["w", [-1.2, -0.8]]))

cosm_range_linear = OrderedDict((["omegab", [0.02025, 0.02425]],
                                 ["omegam", [0.0998, 0.1398]],
                                 ["Omagel", [0.4594, 0.9094]],
                                 ["ln(10^10As)", None],
                                 ["ns", [0.8645, 1.0645]],
                                 ["w", None]))

class constants:
    c_m_s = 299792458. # speed of light [m/s]
    c_km_s = c_m_s*10**-3 # speed of light [km/s]
    pc_m = 3.0857e16 # parsec [m]


def test_cosm_range_base(cparam_in, cparam_range, return_edges=False):
    flag = False
    cparam_in = cparam_in.reshape(1, 6)
    if return_edges:
        cparam_out = np.copy(cparam_in)

    for i, (key, edges) in enumerate(cparam_range.items()):
        if edges is None:
            continue
        if cparam_in[0, i] < edges[0]:
            flag = True
            logging.warning(('Warning: %s=%f is out of the supported range [%f:%f]' % (
                key, cparam_in[0, i], edges[0], edges[1])))
            if return_edges:
                cparam_out[0, i] = edges[0]
        if cparam_in[0, i] > edges[1]:
            flag = True
            logging.warning(('Warning: %s=%f is out of the supported range [%f:%f]' % (
                key, cparam_in[0, i], edges[0], edges[1])))
            if return_edges:
                cparam_out[0, i] = edges[1]

    if return_edges:
        return flag, cparam_out
    else:
        return flag


def test_cosm_range(cparam_in, return_edges=False):
    return test_cosm_range_base(cparam_in, cosm_range, return_edges=return_edges)


def test_cosm_range_linear(cparam_in, return_edges=False):
    return test_cosm_range_base(cparam_in, cosm_range_linear, return_edges=return_edges)


class cosmo_class:
    cparam = np.array([0.02225, 0.1198, 0.6844, 3.094,
                       0.9645, -1.]).reshape(1, 6)
    rho_cr = 2.77536627e11  # [M_sun/h] / [Mpc/h]^3\n"

    def __init__(self):
        print('initialize cosmo_class')
        self.cpara_list = np.loadtxt(os.path.dirname(os.path.abspath(
            __file__)) + '/../data/params.dat')[:, [0, 1, 2, 4, 5, 6]]
        self.ascale_list = np.loadtxt(os.path.dirname(
            os.path.abspath(__file__)) + '/../data/scales.dat')

    def set_cosmology_predefined(self, i):
        self.set_cosmology(self.cpara_list[i, ])

    def get_cosmology_predefined(self, i):
        return self.cpara_list[i, ]

    def get_Omega0(self):
        return 1.-self.cparam[0, 2]

    def get_scalefactor_predefined(self, i):
        return self.ascale_list[i]

    def get_redshift_predefined(self, i):
        return 1./self.ascale_list[i] - 1.

    def set_cosmology(self, cparam_in):
        try:
            test_cosm_range(cparam_in)
            self.cparam = cparam_in.reshape(1, 6)
        except:
            print('cosmological parameter out of supported range!')

    def get_cosmology(self):
        return self.cparam

    def get_comoving_distance(self, z):
        if isinstance(z, np.ndarray) or isinstance(z, list):
            chi = np.zeros(len(z))
            for i, _z in enumerate(z):
                res = integrate.quad(lambda t: 1./self.get_Ez(t), 0., _z)
                chi[i] = constants.c_km_s/100.*res[0]
        else:
            res = integrate.quad(lambda t: 1./self.get_Ez(t), 0., z)
            chi = constants.c_km_s/100.*res[0]
        return chi # Mpc/h

    def get_Ez(self, z):
        Omegade0 = self.cparam[0, 2]
        w = self.cparam[0,5]
        return np.sqrt(self.get_Omega0()*(1.+z)**3.+Omegade0*np.exp(3.*(1.+w)*np.log(1.+z)))

    def linearGrowth(self, z):
        Ode = self.cparam[0, 2]
        Om = 1 - Ode
        wde = self.cparam[0, 5]
        a_scale = 1./(1.+z)
        alpha = -1./(3.*wde)
        beta = (wde-1.)/(2.*wde)
        gamma = 1.-5./(6.*wde)
        x = -Ode/Om * a_scale**(-3.*wde)
        res = integrate.quad(lambda t: t**(beta-1.)*(1.-t)
                             ** (gamma-beta-1.)*(1.-t*x)**(-alpha), 0, 1.)
        return a_scale * res[0]

    def Dgrowth_from_z(self, z):
        return self.linearGrowth(z)/self.linearGrowth(0)

    def Dgrowth_from_a(self, a):
        z = 1./a - 1.
        return self.Dgrowth_from_z(z)

    def Dgrowth_from_snapnum(self, i):
        z = self.get_redshift_predefined(i)
        return self.Dgrowth_from_z(z)

    def f_from_z(self, z):
        a = 1./(1. + z)
        return self.f_from_a(a)

    def f_from_a(self, a):
        return derivative(lambda t: a*np.log(self.Dgrowth_from_a(t)), a, dx=1e-6)

    def f_from_spapnum(self, i):
        z = self.get_redshift_predefined(i)
        a = 1./(1. + z)
        return self.f_from_a(a)

    def get_hubble(self):
        return np.sqrt((self.cparam[0, 0] + self.cparam[0, 1] + 0.00064) / (1 - self.cparam[0, 2]))

    def get_H0(self):
        return 100.*self.get_hubble()

    def get_BAO_approx(self):
        h = self.get_hubble()
        return 147. * h * (self.cparam[0, 1]/0.13)**(-0.25) * (self.cparam[0, 0]/0.024)**(-0.08)
