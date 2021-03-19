import os
from .cosmo_util import cosmo_class
from .cosmo_util import constants
from .pklin import pklin_gp
from .xinl import xinl_gp
from .gamma1 import gamma1_gp
from .cross import cross_gp
from .auto import auto_gp
from .hmf import hmf_gp
from .. import pyfftlog_interface
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from scipy import integrate


class base_class(object):
    """base_class

        The base class of dark emulator.
        This holds all the individual emulator class objects for different statistical quantities.
        By passing to the base class object, the cosmological paramters in all the lower-level objects are updated.

        Args:
            cparam (numpy array): Cosmological parameters :math:`(\omega_b, \omega_{m}, \Omega_{de}, \ln(10^{10}A_s), n_s, w)`

        Attributes:
            cosmo (class cosmo_class): A class object dealing with the cosmological parameters and some basic cosmological quantities such as expansion and linear growth.
            pkL (class pklin_gp): A class object that takes care of the linear matter power spectrum
            g1 (class gamma1_gp): A class object that takes care of the large-scale bias as well as the BAO damping
            xi_cross (class cross_gp): A class object that takes care of the halo-matter cross correlation function
            xi_auto (class auto_gp): A class object that takes care of the halo-halo correlation function
            massfunc (class hmf_gp): A class object that takes care of the halo mass function
            xiNL (class xinl_gp): A class object that takes care of the nonlinear matter correlation function (experimental)
    """
    def __init__(self):
        self.cosmo = cosmo_class()
        self.pkL = pklin_gp()
        self.g1 = gamma1_gp()
        self.xi_cross = cross_gp()
        self.xi_auto = auto_gp()
        self.massfunc = hmf_gp()
        self.xiNL = xinl_gp()
        # initialize emulators with the fiducial model and at z=0
        self.set_cosmology(self.cosmo.get_cosmology())

    def set_cosmology(self, cparam):
        """set_cosmology

        Let the emulator know the cosmological parameters.
        This interface passes the 6 parameters to all the class objects
        used for the emulation of various halo statistics.

        The current version supports wCDM cosmologies specified by the 6
        parameters as described below. Other parameters are automatically computed:

        :math:`\Omega_{m}=1-\Omega_{de},`

        :math:`h=\sqrt{(\omega_b+\omega_c+\omega_{\\nu})/\Omega_m},`

        where the neutrino density is fixed by :math:`\omega_{\\nu} = 0.00064` corresponding to the mass sum of 0.06 eV.

        Args:
            cparam (numpy array): Cosmological parameters
                :math:`(\omega_b, \omega_{m}, \Omega_{de}, \ln(10^{10}A_s), n_s, w)`
        """
        self.cosmo.set_cosmology(cparam)
        self.pkL.set_cosmology(self.cosmo)
        # self.xiL.set_cosmology(self.cosmo)
        self.xiNL.set_cosmology(self.cosmo)
        self.xi_auto.set_cosmology(self.cosmo)
        self.xi_cross.set_cosmology(self.cosmo)
        self.massfunc.set_cosmology(self.cosmo)
        self.g1.set_cosmology(self.cosmo)

    def _set_cosmology_predefined(self, i):
        self.set_cosmology(self.cosmo.get_cosmology_predefined(i))

    def get_sd(self, z):
        """get_sd

        Compute the root mean square of the linear displacement, :math:`\sigma_d`,
        for the current cosmological model at redshift z.

        Args:
            z (float): redshift

        Returns:
            float: :math:`\sigma_d`
        """
        return self.Dgrowth_from_z(z)*self.g1.sd0

    def mass_to_dens(self, mass_thre, redshift, integration="quad"):
        """mass_to_dens

        Convert the halo mass threshold to the cumulative number density for the current
        cosmological model at redshift z.

        Args:
            mass_thre (float): mass threshold in :math:`h^{-1}M_{\odot}`
            redshift (float): redshift
            integration (str, optional): type of integration (default: "quad", "trapz" is also supported)

        Returns:
            float: halo number density in :math:`[(h^{-1}\mathrm{Mpc})^{-3}]`
        """
        return self.massfunc.mass_to_dens(mass_thre, redshift, integration=integration)

    def dens_to_mass(self, dens, redshift, nint=20, integration="quad"):
        """dens_to_mass

        Convert the cumulative number density to the halo mass threshold for the current
        cosmological model at redshift z.

        Args:
            dens (float): halo number density in :math:`(h^{-1}\mathrm{Mpc})^{-3}`
            redshift (float): redshift
            nint (int, optional): number of sampling points in log(M) used for interpolation
            integration (str, optional): type of integration (default: "quad", "trapz" is also supported)

        Returns:
            float: mass threshold in :math:`[h^{-1}M_{\odot}]`
        """
        return self.massfunc.dens_to_mass(dens, redshift, nint, integration=integration)

    def get_f_HMF(self, redshift):
        """get_f_HMF

        Compute the multiplicity function :math:`f(\sigma)`, defined through :math:`dn/dM = f(\sigma)\\bar{\\rho}_m/M d \ln \sigma^{-1}/dM`.

        Args:
            redshift (float): redshift

        Returns:
            (tuple): tuple containing:

                mass(numpy array): :math:`M_{200b}`

                mass variance(numpy array): :math:`\sigma(M_{200b)`

                multiplicity function(numpy array): :math:`f(\sigma)`
        """
        D0 = self.Dgrowth_from_z(redshift)
        return self.massfunc.Mlist, D0*self.massfunc.sigs0, self.massfunc.f_HMF_func(D0*self.massfunc.sigs0, redshift)

    def get_nhalo(self, Mmin, Mmax, vol, redshift):
        """get_nhalo

            Compute the mean number of halos in a given mass range and volume.

            Args:
                Mmin (float): Minimum halo mass in :math:`[h^{-1}M_\odot]`
                Mmax (float): Maximum halo mass in :math:`[h^{-1}M_\odot]`
                vol (float): Volume in :math:`[(h^{-1}\mathrm{Mpc})^3]`

            Returns:
                float: Number of halos
        """
        return self.massfunc.get_nhalo(Mmin, Mmax, vol, redshift)

    def get_nhalo_tinker(self, Mmin, Mmax, vol, redshift):
        """get_nhalo_tinker

            Compute the mean number of halos in a given mass range and volume based on the fitting formula by Tinker et al. (ApJ 688 (2008) 709).

            Args:
                Mmin (float): Minimum halo mass in :math:`[h^{-1}M_\odot]`
                Mmax (float): Maximum halo mass in :math:`[h^{-1}M_\odot]`
                vol (float): Volume in :math:`[(h^{-1}\mathrm{Mpc})^3]`

            Returns:
                float: Number of halos
        """
        return self.massfunc.get_nhalo_tinker(Mmin, Mmax, vol, redshift)

    def get_xilin(self, xs):
        """get_xilin

            Compute the linear matter correlation function at z=0.

            Args:
                xs (numpy array): Separations in :math:`[h^{-1}\mathrm{Mpc}]`

            Returns:
                numpy array: Correlation function at separations given in the argument xs.
        """
        ks = np.logspace(-3, 3, 300)
        return pyfftlog_interface.pk2xi_pyfftlog(iuspline(ks, self.pkL.get(ks)))(xs)

    def _get_xinl_tree(self, xs, redshift):
        return pyfftlog_interface.pk2xi_pyfftlog(self._get_pkmatter_tree_spline(redshift))(xs)

    def _get_xinl_direct(self, xs, z):
        return self.xiNL.get(xs, z)

    def get_xinl(self, xs, redshift):
        """get_xinl

            Compute the nonlinear matter correlation function.  Note that this is still in a development phase, and the accuracy has not yet been fully evaluated.

            Args:
                xs (numpy array): Separations in :math:`[h^{-1}\mathrm{Mpc}]`

            Returns:
                numpy array: Correlation function at separations given in the argument xs.
        """
        xi_dir = self._get_xinl_direct(xs, redshift)
        xi_tree = self._get_xinl_tree(xs, redshift)
        rswitch = min(60., 0.5 * self.cosmo.get_BAO_approx())
        return xi_dir * np.exp(-(xs/rswitch)**4) + xi_tree * (1-np.exp(-(xs/rswitch)**4))

    def get_pknl(self, k, z):
        """get_pknl

            Compute the nonlinear matter power spectrum. Note that this is still in a development phase, and the accuracy has not yet been fully evaluated.

            Args:
                k (numpy array): Wavenumbers in :math:`[h\mathrm{Mpc}^{-1}]`
                z (float): redshift

            Returns:
                numpy array: Nonlinear matter power spectrum at wavenumbers given in the argument k.
        """
        xs = np.logspace(-3, 3, 2000)
        xinl = self.get_xinl(xs, z)
        return pyfftlog_interface.xi2pk_pyfftlog(iuspline(xs, xinl))(k)

    def get_pklin(self, k):
        """get_pklin

            Compute the linear matter power spectrum at z=0.

            Args:
                k (numpy array): Wavenumbers in :math:`[h\mathrm{Mpc}^{-1}]`

            Returns:
                numpy array: Linear power spectrum at wavenumbers given in the argument k.
        """
        return self.pkL.get(k)

    def _get_pklin_from_snap(self, x, i):
        Dp = self._Dgrowth_from_snapnum(i)
        return Dp**2 * self.pkL.get(x)

    def get_pklin_from_z(self, k, z):
        """get_pklin_z

            Compute the linear matter power spectrum.

            Args:
                k (numpy array): Wavenumbers in :math:`[h\mathrm{Mpc}^{-1}]`
                z (float): redshift

            Returns:
                numpy array: Linear power spectrum at wavenumbers given in the argument k.
        """
        Dp = self.Dgrowth_from_z(z)
        return Dp**2 * self.pkL.get(k)

    def _get_xiauto_tree(self, xs, logdens1, logdens2, redshift):
        ks = np.logspace(-3, 3, 2000)
        g1 = self.g1.get(ks, redshift, logdens1)
        g2 = self.g1.get(ks, redshift, logdens2)
        pm_lin = self.get_pklin(ks)
        ph_tree = g1 * g2 * pm_lin
        return pyfftlog_interface.pk2xi_pyfftlog(iuspline(ks, ph_tree))(xs)

    def _get_xiauto_direct(self, xs, logdens1, logdens2, redshift):
        return self.xi_auto.get(xs, redshift, logdens1, logdens2)

    def get_xiauto(self, xs, logdens1, logdens2, redshift):
        """get_xiauto

            Compute the halo-halo correlation function, :math:`\\xi_\mathrm{hh}(x;n_1,n_2)`, bwtween 2 mass threshold halo samples specified by the corresponding cumulative number densities.

            Args:
                xs (numpy array): Separations in :math:`[h^{-1}\mathrm{Mpc}]`
                logdens1 (float): Logarithm of the cumulative halo number density of the first halo sample taken from the most massive, :math:`\log_{10}[n_1/(h^{-1}\mathrm{Mpc})^3]`
                logdens2 (float): Logarithm of the cumulative halo number density of the second halo sample taken from the most massive, :math:`\log_{10}[n_2/(h^{-1}\mathrm{Mpc})^3]`
                redshift (float): Redshift at which the correlation function is evaluated

            Returns:
                numpy array: Halo correlation function
        """
        xi_tree = self._get_xiauto_tree(xs, logdens1, logdens2, redshift)
        if logdens1 >= -5.75 and logdens2 >= -5.75:
            xi_dir = self._get_xiauto_direct(xs, logdens1, logdens2, redshift)
            rswitch = min(60., 0.5 * self.cosmo.get_BAO_approx())
            xi_tot = xi_dir * np.exp(-(xs/rswitch)**4) + \
                xi_tree * (1-np.exp(-(xs/rswitch)**4))
        elif logdens1 >= -5.75 and logdens2 < -5.75:
            xi_dir = self._get_xiauto_direct(
                xs, logdens1, -5.75, redshift) * self.g1.bias_ratio(redshift, logdens2)
            rswitch = min(60., 0.5 * self.cosmo.get_BAO_approx())
            xi_tot = xi_dir * np.exp(-(xs/rswitch)**4) + \
                xi_tree * (1-np.exp(-(xs/rswitch)**4))
        elif logdens1 < -5.75 and logdens2 >= -5.75:
            xi_dir = self._get_xiauto_direct(
                xs, -5.75, logdens2, redshift) * self.g1.bias_ratio(redshift, logdens1)
            rswitch = min(60., 0.5 * self.cosmo.get_BAO_approx())
            xi_tot = xi_dir * np.exp(-(xs/rswitch)**4) + \
                xi_tree * (1-np.exp(-(xs/rswitch)**4))
        else:
            xi_dir = self._get_xiauto_direct(xs, -5.75, -5.75, redshift) * self.g1.bias_ratio(
                redshift, logdens1)*self.g1.bias_ratio(redshift, logdens2)
            rswitch = min(60., 0.5 * self.cosmo.get_BAO_approx())
            xi_tot = xi_dir * np.exp(-(xs/rswitch)**4) + \
                xi_tree * (1-np.exp(-(xs/rswitch)**4))
        return xi_tot

    def _get_xiauto_spl(self, logdens1, logdens2, redshift):
        xs = np.logspace(-1, 3., 2000)
        xi_tree = self._get_xiauto_tree(xs, logdens1, logdens2, redshift)
        if logdens1 >= -5.75 and logdens2 >= -5.75:
            xi_dir = self._get_xiauto_direct(xs, logdens1, logdens2, redshift)
            rswitch = min(60., 0.5 * self.cosmo.get_BAO_approx())
            xi_tot = xi_dir * np.exp(-(xs/rswitch)**4) + \
                xi_tree * (1-np.exp(-(xs/rswitch)**4))
        elif logdens1 >= -5.75 and logdens2 < -5.75:
            xi_dir = self._get_xiauto_direct(
                xs, logdens1, -5.75, redshift) * self.g1.bias_ratio(redshift, logdens2)
            rswitch = min(60., 0.5 * self.cosmo.get_BAO_approx())
            xi_tot = xi_dir * np.exp(-(xs/rswitch)**4) + \
                xi_tree * (1-np.exp(-(xs/rswitch)**4))
        elif logdens1 < -5.75 and logdens2 >= -5.75:
            xi_dir = self._get_xiauto_direct(
                xs, -5.75, logdens2, redshift) * self.g1.bias_ratio(redshift, logdens1)
            rswitch = min(60., 0.5 * self.cosmo.get_BAO_approx())
            xi_tot = xi_dir * np.exp(-(xs/rswitch)**4) + \
                xi_tree * (1-np.exp(-(xs/rswitch)**4))
        else:
            xi_dir = self._get_xiauto_direct(xs, -5.75, -5.75, redshift) * self.g1.bias_ratio(
                redshift, logdens1)*self.g1.bias_ratio(redshift, logdens2)
            rswitch = min(60., 0.5 * self.cosmo.get_BAO_approx())
            xi_tot = xi_dir * np.exp(-(xs/rswitch)**4) + \
                xi_tree * (1-np.exp(-(xs/rswitch)**4))
        return iuspline(xs, xi_tot)

    def get_xiauto_massthreshold(self, xs, Mthre, redshift):
        """get_xiauto_massthreshold

            Compute the halo-halo correlation function, :math:`\\xi_\mathrm{hh}(x;>M_\mathrm{th})`,  for a mass threshold halo sample.

            Args:
                xs (numpy array): Separations in :math:`[h^{-1}\mathrm{Mpc}]`
                Mthre (float): Minimum halo mass threshold in :math:`[h^{-1}M_\odot]`
                redshift (float): Redshift at which the correlation function is evaluated

            Returns:
                numpy array: Halo correlation function
        """
        logdens = np.log10(self.mass_to_dens(Mthre, redshift))
        return self.get_xiauto(xs, logdens, logdens, redshift)

    def get_xiauto_mass(self, xs, M1, M2, redshift):
        """get_xiauto_mass

            Compute the halo-halo correlation function, :math:`\\xi_\mathrm{hh}(x;M_1,M_2)`, between 2 halo samples with mass :math:`M_1` and :math:`M_2`.
            Args:
                xs (numpy array): Separations in :math:`[h^{-1}\mathrm{Mpc}]`
                M1 (float): Halo mass of the first sample in :math:`[h^{-1}M_\odot]`
                M2 (float): Halo mass of the second sample in :math:`[h^{-1}M_\odot]`
                redshift (float): Redshift at which the correlation function is evaluated

            Returns:
                numpy array: Halo correlation function
        """
        M1p = M1 * 1.01
        M1m = M1 * 0.99
        M2p = M2 * 1.01
        M2m = M2 * 0.99
        dens1p = self.mass_to_dens(M1p, redshift)
        dens1m = self.mass_to_dens(M1m, redshift)
        dens2p = self.mass_to_dens(M2p, redshift)
        dens2m = self.mass_to_dens(M2m, redshift)
        logdens1p, logdens1m, logdens2p, logdens2m = np.log10(
            dens1p), np.log10(dens1m), np.log10(dens2p), np.log10(dens2m)

        ximm = self.get_xiauto(xs, logdens1m, logdens2m, redshift)
        ximp = self.get_xiauto(xs, logdens1m, logdens2p, redshift)
        xipm = self.get_xiauto(xs, logdens1p, logdens2m, redshift)
        xipp = self.get_xiauto(xs, logdens1p, logdens2p, redshift)

        numer = ximm * dens1m * dens2m - ximp * dens1m * dens2p - \
            xipm * dens1p * dens2m + xipp * dens1p * dens2p
        denom = dens1m * dens2m - dens1m * dens2p - dens1p * dens2m + dens1p * dens2p
        return numer / denom

    def _get_phh_tree(self,ks,logdens1,logdens2,redshift):
        g1 = self.g1.get(ks,redshift,logdens1)
        g2 = self.g1.get(ks,redshift,logdens2)
        pm_lin = self.get_pklin(ks)
        ph_tree = g1 * g2 * pm_lin
        return ph_tree

    def _get_phh_direct(self,ks,logdens1,logdens2,redshift):
        xs = np.logspace(-3,3,4000)
        xihh = self.xi_auto.get(xs,redshift,logdens1,logdens2)
        return pyfftlog_interface.xi2pk_pyfftlog(iuspline(xs,xihh))(ks)

    def get_phh(self,ks,logdens1,logdens2,redshift):
        """get_phh

            Compute the halo-halo power spectrum :math:`P_{hh}(k;n_1,n_2)` between 2 mass threshold halo samples specified by the corresponding cumulative number densities.

            Args:
                ks (numpy array): Wavenumbers in :math:`[h\mathrm{Mpc}^{-1}]`
                logdens1 (float): Logarithm of the cumulative halo number density of the first halo sample taken from the most massive, :math:`\log_{10}[n_1/(h^{-1}\mathrm{Mpc})^3]`
                logdens2 (float): Logarithm of the cumulative halo number density of the second halo sample taken from the most massive, :math:`\log_{10}[n_2/(h^{-1}\mathrm{Mpc})^3]`
                redshift (float): redshift at which the power spectrum is evaluated

            Returns:
                numpy array: halo power spectrum in :math:`[(h^{-1}\mathrm{Mpc})^{3}]`
        """
        xs = np.logspace(-3,3,4000)
        xi_tree = self._get_xiauto_tree(xs,logdens1,logdens2,redshift)
        rswitch = min(60.,0.5 * self.cosmo.get_BAO_approx())
        if logdens1 >= -5.75 and logdens2 >= -5.75:
                xi_dir = self._get_xiauto_direct(xs,logdens1,logdens2,redshift)
                xi_tot = xi_dir * np.exp(-(xs/rswitch)**4) + xi_tree * (1-np.exp(-(xs/rswitch)**4))
        elif logdens1 >= -5.75 and logdens2 < -5.75:
                xi_dir = self._get_xiauto_direct(xs,logdens1,-5.75,redshift) * self.g1.bias_ratio(redshift,logdens2)
                xi_tot = xi_dir * np.exp(-(xs/rswitch)**4) + xi_tree * (1-np.exp(-(xs/rswitch)**4))
        elif logdens1 < -5.75 and logdens2 >= -5.75:
                xi_dir = self._get_xiauto_direct(xs,-5.75,logdens2,redshift) * self.g1.bias_ratio(redshift,logdens1)
                xi_tot = xi_dir * np.exp(-(xs/rswitch)**4) + xi_tree * (1-np.exp(-(xs/rswitch)**4))
        else:
                xi_dir = self._get_xiauto_direct(xs,-5.75,-5.75,redshift) * self.g1.bias_ratio(redshift,logdens1)*self.g1.bias_ratio(redshift,logdens2)
                xi_tot = xi_dir * np.exp(-(xs/rswitch)**4) + xi_tree * (1-np.exp(-(xs/rswitch)**4))
        return pyfftlog_interface.xi2pk_pyfftlog(iuspline(xs,xi_tot))(ks)

    def _get_phh_tree_cut(self,ks,logdens1,logdens2,redshift):
        xs = np.logspace(-3,3,4000)
        xi_tree = self._get_xiauto_tree(xs,logdens1,logdens2,redshift)
        rswitch = min(60.,0.5 * self.cosmo.get_BAO_approx())
        xi_tot = xi_tree * (1-np.exp(-(xs/rswitch)**4))
        return pyfftlog_interface.xi2pk_pyfftlog(iuspline(xs,xi_tot))(ks)

    def _get_phh_direct_cut(self,ks,logdens1,logdens2,redshift):
        xs = np.logspace(-3,3,4000)
        rswitch = min(60.,0.5 * self.cosmo.get_BAO_approx())
        if logdens1 >= -5.75 and logdens2 >= -5.75:
                xi_dir = self._get_xiauto_direct(xs,logdens1,logdens2,redshift)
                xi_tot = xi_dir * np.exp(-(xs/rswitch)**4)
        elif logdens1 >= -5.75 and logdens2 < -5.75:
                xi_dir = self._get_xiauto_direct(xs,logdens1,-5.75,redshift) * self.g1.bias_ratio(redshift,logdens2)
                xi_tot = xi_dir * np.exp(-(xs/rswitch)**4)
        elif logdens1 < -5.75 and logdens2 >= -5.75:
                xi_dir = self._get_xiauto_direct(xs,-5.75,logdens2,redshift) * self.g1.bias_ratio(redshift,logdens1)
                xi_tot = xi_dir * np.exp(-(xs/rswitch)**4)
        else:
                xi_dir = self._get_xiauto_direct(xs,-5.75,-5.75,redshift) * self.g1.bias_ratio(redshift,logdens1)*self.g1.bias_ratio(redshift,logdens2)
                xi_tot = xi_dir * np.exp(-(xs/rswitch)**4)
        return pyfftlog_interface.xi2pk_pyfftlog(iuspline(xs,xi_tot))(ks)


    def get_phh_massthreshold(self,ks,Mthre,redshift):
        """get_phh_massthreshold

            Compute the halo-halo auto power spectrum :math:`P_{hh}(k;>M_\mathrm{th})` for a mass threshold halo sample.

            Args:
                ks (numpy array): Wavenumbers in :math:`[h\mathrm{Mpc}^{-1}]`
                Mthre (float): Minimum halo mass threshold in :math:`[h^{-1}M_\odot]`
                redshift (float): redshift at which the power spectrum is evaluated

            Returns:
                numpy array: halo power spectrum in :math:`[(h^{-1}\mathrm{Mpc})^{3}]`
        """
        logdens = np.log10(self.mass_to_dens(Mthre,redshift))
        return self.get_phh(ks,logdens,logdens,redshift)

    def get_phh_mass(self,ks,M1,M2,redshift):
        """get_phh_mass

            Compute the halo-halo power spectrum :math:`P_{hh}(k;M_1,M_2)` between 2 halo samples with mass :math:`M_1` and :math:`M_2`.

            Args:
                ks (numpy array): Wavenumbers in :math:`[h\mathrm{Mpc}^{-1}]`
                M1 (float): Halo mass of the first sample in :math:`[h^{-1}M_\odot]`
                M2 (float): Halo mass of the second sample in :math:`[h^{-1}M_\odot]`
                redshift (float): redshift at which the power spectrum is evaluated

            Returns:
                numpy array: halo power spectrum in :math:`[(h^{-1}\mathrm{Mpc})^{3}]`
        """
        M1p = M1 * 1.01
        M1m = M1 * 0.99
        M2p = M2 * 1.01
        M2m = M2 * 0.99
        dens1p = self.mass_to_dens(M1p,redshift)
        dens1m = self.mass_to_dens(M1m,redshift)
        dens2p = self.mass_to_dens(M2p,redshift)
        dens2m = self.mass_to_dens(M2m,redshift)
        logdens1p, logdens1m, logdens2p, logdens2m = np.log10(dens1p), np.log10(dens1m), np.log10(dens2p), np.log10(dens2m)

        pmm = self.get_phh(ks,logdens1m,logdens2m,redshift)
        pmp = self.get_phh(ks,logdens1m,logdens2p,redshift)
        ppm = self.get_phh(ks,logdens1p,logdens2m,redshift)
        ppp = self.get_phh(ks,logdens1p,logdens2p,redshift)

        numer = pmm * dens1m * dens2m - pmp * dens1m * dens2p - ppm * dens1p * dens2m + ppp * dens1p * dens2p
        denom = dens1m * dens2m - dens1m * dens2p - dens1p * dens2m + dens1p * dens2p
        return numer / denom


    def get_wauto(self, R2d, logdens1, logdens2, redshift):
        """get_wauto

            Compute the projected halo-halo correlation function :math:`w_{hh}(R;n_1,n_2)` for 2 mass threshold halo samples specified by the corresponding cumulative number densities.

            Args:
                R2d (numpy array): 2 dimensional projected separation in :math:`[h^{-1}\mathrm{Mpc}]`
                logdens1 (float): Logarithm of the cumulative halo number density of the first halo sample taken from the most massive, :math:`\log_{10}[n_1/(h^{-1}\mathrm{Mpc})^3]`
                logdens2 (float): Logarithm of the cumulative halo number density of the second halo sample taken from the most massive, :math:`\log_{10}[n_2/(h^{-1}\mathrm{Mpc})^3]`
                redshift (float): redshift at which the power spectrum is evaluated

            Returns:
                numpy array: projected halo correlation function in :math:`[h^{-1}\mathrm{Mpc}]`
        """
        xs = np.logspace(-3, 3, 1000)
        xi_auto = self.get_xiauto(xs, logdens1, logdens2, redshift)
        pk_spl = pyfftlog_interface.xi2pk_pyfftlog(iuspline(xs, xi_auto))
        return pyfftlog_interface.pk2xiproj_J0_pyfftlog(pk_spl, logkmin=-3.0, logkmax=3.0)(R2d)

    def get_wauto_cut(self, R2d, logdens1, logdens2, redshift, pimax, integration="quad"):
        """get_wauto_cut

            Compute the projected halo-halo correlation function :math:`w_{hh}(R;n_1,n_2)` for 2 mass threshold halo samples specified by the corresponding cumulative number densities.
            Unlike get_wauto, this function considers a finite width for the radial integration, from :math:`-\pi_\mathrm{max}` to :math:`\pi_\mathrm{max}`.

            Args:
                R2d (numpy array): 2 dimensional projected separation in :math:`[h^{-1}\mathrm{Mpc}]`
                logdens1 (float): Logarithm of the cumulative halo number density of the first halo sample taken from the most massive, :math:`\log_{10}[n_1/(h^{-1}\mathrm{Mpc})^3]`
                logdens2 (float): Logarithm of the cumulative halo number density of the second halo sample taken from the most massive, :math:`\log_{10}[n_2/(h^{-1}\mathrm{Mpc})^3]`
                redshift (float): redshift at which the power spectrum is evaluated
                pimax (float): :math:`\pi_\mathrm{max}` for the upper limit of the integral

            Returns:
                numpy array: projected halo correlation function in :math:`[h^{-1}\mathrm{Mpc}]`
        """
        xi3d = self._get_xiauto_spl(logdens1, logdens2, redshift)
        wauto = []
        if integration == "quad":
            for R2dnow in R2d:
                wauto.append(
                    2*integrate.quad(lambda t: xi3d(np.sqrt(t**2+R2dnow**2)), 0, pimax, epsabs=1e-4)[0])
        elif integration == "trapz":
            t = np.linspace(0, pimax, 1024)
            dt = t[1]-t[0]
            for R2dnow in R2d:
                wauto.append(
                    2*integrate.trapz(xi3d(np.sqrt(t**2+R2dnow**2)), dx=dt))
        else:
            raise RuntimeError(
                "You should specify valid integration algorithm: quad or trapz")
        return np.array(wauto)

    def get_wauto_massthreshold(self, R2d, Mthre, redshift):
        """get_wauto_massthreshold

            Compute the projected halo-halo correlation function :math:`w_{hh}(R;>M_\mathrm{th})` for a mass threshold halo sample.

            Args:
                R2d (numpy array): 2 dimensional projected separation in :math:`[h^{-1}\mathrm{Mpc}]`
                Mthre (float): Minimum halo mass threshold in :math:`[h^{-1}M_\odot]`
                redshift (float): redshift at which the power spectrum is evaluated

            Returns:
                numpy array: projected halo correlation function in :math:`[h^{-1}\mathrm{Mpc}]`
        """
        logdens = np.log10(self.mass_to_dens(Mthre, redshift))
        return self.get_wauto(R2d, logdens, logdens, redshift)

    def get_wauto_masthreshold_cut(self, R2d, Mthre, redshift, pimax, integration="quad"):
        """get_wauto_massthreshold_cut

            Compute the projected halo-halo correlation function :math:`w_{hh}(R;>M_\mathrm{th})` for a mass threshold halo sample.
            Unlike get_wauto_massthreshold, this function considers a finite width for the radial integration, from :math:`-\pi_\mathrm{max}` to :math:`\pi_\mathrm{max}`.

            Args:
                R2d (numpy array): 2 dimensional projected separation in :math:`[h^{-1}\mathrm{Mpc}]`
                Mthre (float): Minimum halo mass threshold in :math:`[h^{-1}M_\odot]`
                redshift (float): redshift at which the power spectrum is evaluated
                pimax (float): :math:`\pi_\mathrm{max}` for the upper limit of the integral

            Returns:
                numpy array: projected halo correlation function in :math:`[h^{-1}\mathrm{Mpc}]`
        """
        logdens = np.log10(self.mass_to_dens(Mthre, redshift))
        return self.get_wauto_cut(R2d, logdens, logdens, redshift, pimax, integration)


    def get_wauto_mass(self, R2d, M1, M2, redshift):
        """get_wauto_mass

            Compute the projected halo-halo correlation function :math:`w_{hh}(R;M_1,M_2)` for 2 mass threshold halo samples.

            Args:
                R2d (numpy array): 2 dimensional projected separation in :math:`[h^{-1}\mathrm{Mpc}]`
                M1 (float): Halo mass of the first sample in :math:`[h^{-1}M_\odot]`
                M2 (float): Halo mass of the second sample in :math:`[h^{-1}M_\odot]`
                redshift (float): redshift at which the power spectrum is evaluated

            Returns:
                numpy array: projected halo correlation function in :math:`[h^{-1}\mathrm{Mpc}]`
        """
        M1p = M1 * 1.01
        M1m = M1 * 0.99
        M2p = M2 * 1.01
        M2m = M2 * 0.99
        dens1p = self.mass_to_dens(M1p, redshift)
        dens1m = self.mass_to_dens(M1m, redshift)
        dens2p = self.mass_to_dens(M2p, redshift)
        dens2m = self.mass_to_dens(M2m, redshift)
        logdens1p, logdens1m, logdens2p, logdens2m = np.log10(
            dens1p), np.log10(dens1m), np.log10(dens2p), np.log10(dens2m)

        wmm = self.get_wauto(R2d, logdens1m, logdens2m, redshift)
        wmp = self.get_wauto(R2d, logdens1m, logdens2p, redshift)
        wpm = self.get_wauto(R2d, logdens1p, logdens2m, redshift)
        wpp = self.get_wauto(R2d, logdens1p, logdens2p, redshift)

        numer = wmm * dens1m * dens2m - wmp * dens1m * dens2p - \
            wpm * dens1p * dens2m + wpp * dens1p * dens2p
        denom = dens1m * dens2m - dens1m * dens2p - dens1p * dens2m + dens1p * dens2p
        return numer / denom

    def get_wauto_mass_cut(self, R2d, M1, M2, redshift, pimax):
        """get_wauto_mass_cut

            Compute the projected halo-halo correlation function :math:`w_{hh}(R;M_1,M_2)` for 2 mass threshold halo samples.
            Unlike get_wauto_mass, this function considers a finite width for the radial integration, from :math:`-\pi_\mathrm{max}` to :math:`\pi_\mathrm{max}`.

            Args:
                R2d (numpy array): 2 dimensional projected separation in :math:`[h^{-1}\mathrm{Mpc}]`
                M1 (float): Halo mass of the first sample in :math:`[h^{-1}M_\odot]`
                M2 (float): Halo mass of the second sample in :math:`[h^{-1}M_\odot]`
                redshift (float): redshift at which the power spectrum is evaluated
                pimax (float): :math:`\pi_\mathrm{max}` for the upper limit of the integral

            Returns:
                numpy array: projected halo correlation function in :math:`[h^{-1}\mathrm{Mpc}]`
        """
        M1p = M1 * 1.01
        M1m = M1 * 0.99
        M2p = M2 * 1.01
        M2m = M2 * 0.99
        dens1p = self.mass_to_dens(M1p, redshift)
        dens1m = self.mass_to_dens(M1m, redshift)
        dens2p = self.mass_to_dens(M2p, redshift)
        dens2m = self.mass_to_dens(M2m, redshift)
        logdens1p, logdens1m, logdens2p, logdens2m = np.log10(
            dens1p), np.log10(dens1m), np.log10(dens2p), np.log10(dens2m)

        wmm = self.get_wauto_cut(R2d, logdens1m, logdens2m, redshift, pimax)
        wmp = self.get_wauto_cut(R2d, logdens1m, logdens2p, redshift, pimax)
        wpm = self.get_wauto_cut(R2d, logdens1p, logdens2m, redshift, pimax)
        wpp = self.get_wauto_cut(R2d, logdens1p, logdens2p, redshift, pimax)

        numer = wmm * dens1m * dens2m - wmp * dens1m * dens2p - \
            wpm * dens1p * dens2m + wpp * dens1p * dens2p
        denom = dens1m * dens2m - dens1m * dens2p - dens1p * dens2m + dens1p * dens2p
        return numer / denom


    def _get_pkmatter_tree(self, redshift):
        ks = np.logspace(-3, 3, 2000)
        g1_dm = self.g1.get_dm(ks, redshift)
        pm_lin = self.get_pklin(ks)
        return g1_dm**2 * pm_lin

    # TN suppressed this because it is a duplication of get_pknl
    # def get_pmnl(self,ks,redshift):
    #     xs = np.logspace(-3,3,2000)
    #     xi = self.get_xinl(xs,redshift)
    #     return pyfftlog_interface.xi2pk_pyfftlog(iuspline(xs,xi))(ks)

    def _get_pkmatter_tree_spline(self, redshift):
        ks = np.logspace(-3, 3, 2000)
        g1_dm = self.g1.get_dm(ks, redshift)
        pm_lin = self.get_pklin(ks)
        return iuspline(ks, g1_dm**2 * pm_lin)

    def _get_pkcross_tree(self, logdens, redshift):
        ks = np.logspace(-3, 3, 2000)
        g1 = self.g1.get(ks, redshift, logdens)
        g1_dm = self.g1.get_dm(ks, redshift)
        pm_lin = self.get_pklin(ks)
        return g1*g1_dm * pm_lin

    def _get_pkcross_tree_spline(self, logdens, redshift):
        ks = np.logspace(-3, 3, 2000)
        g1 = self.g1.get(ks, redshift, logdens)
        g1_dm = self.g1.get_dm(ks, redshift)
        pm_lin = self.get_pklin(ks)
        return iuspline(ks, g1*g1_dm * pm_lin)

    def _get_xicross_tree(self, xs, logdens, redshift):
        return pyfftlog_interface.pk2xi_pyfftlog(self._get_pkcross_tree_spline(logdens, redshift))(xs)

    def _get_xicross_direct(self, xs, logdens, redshift):
        return self.xi_cross.get(xs, redshift, logdens)

    def get_xicross(self, xs, logdens, redshift):
        """get_xicross

            Compute the halo-matter cross correlation function :math:`\\xi_{hm}(x;n_h)` for a mass threshold halo sample specified by the corresponding cumulative number density.

            Args:
                xs (numpy array): Separations in :math:`[h^{-1}\mathrm{Mpc}]`
                logdens (float): Logarithm of the cumulative halo number density of the halo sample taken from the most massive, :math:`\log_{10}[n_h/(h^{-1}\mathrm{Mpc})^3]`
                redshift (float): redshift at which the power spectrum is evaluated

            Returns:
                numpy array: Halo-matter cross correlation function
        """
        xi_dir = self._get_xicross_direct(xs, logdens, redshift)
        xi_tree = self._get_xicross_tree(xs, logdens, redshift)
        rswitch = min(60., 0.5 * self.cosmo.get_BAO_approx())
        return xi_dir * np.exp(-(xs/rswitch)**4) + xi_tree * (1-np.exp(-(xs/rswitch)**4))

    def get_xicross_massthreshold(self, xs, Mthre, redshift):
        """get_xicross_massthreshold

            Compute the halo-matter cross correlation function :math:`\\xi_{hm}(x;>M_\mathrm{th})` for a mass threshold halo sample.

            Args:
                xs (numpy array): Separations in :math:`[h^{-1}\mathrm{Mpc}]`
                Mthre (float): Minimum mass threshold of a halo sample in :math:`[h^{-1}M_\odot]`
                redshift (float): redshift at which the power spectrum is evaluated

            Returns:
                numpy array: Halo-matter cross correlation function
        """
        logdens = np.log10(self.mass_to_dens(Mthre, redshift))
        return self.get_xicross(xs, logdens, redshift)

    def get_xicross_mass(self, xs, M, redshift):
        """get_xicross_mass

            Compute the halo-matter cross correlation function :math:`\\xi_{hm}(x;M)` for halos with mass :math:`M`.

            Args:
                xs (numpy array): Separations in :math:`[h^{-1}\mathrm{Mpc}]`
                M (float): Halo mass in :math:`[h^{-1}M_\odot]`
                redshift (float): redshift at which the power spectrum is evaluated

            Returns:
                numpy array: Halo-matter cross correlation function
        """
        Mp = M * 1.01
        Mm = M * 0.99
        logdensp = np.log10(self.mass_to_dens(Mp, redshift))
        logdensm = np.log10(self.mass_to_dens(Mm, redshift))
        xip = self.get_xicross(xs, logdensp, redshift)
        xim = self.get_xicross(xs, logdensm, redshift)
        return (xim * 10**logdensm - xip * 10**logdensp) / (10**logdensm - 10**logdensp)

    def _get_phm_tree(self,ks,logdens,redshift):
        g1 = self.g1.get(ks,redshift,logdens)
        g1_dm = self.g1.get_dm(ks,redshift)
        pm_lin = self.get_pklin(ks)
        return g1*g1_dm * pm_lin

    def _get_phm_direct(self,ks,logdens,redshift):
        xs = np.logspace(-3,3,2000)
        return pyfftlog_interface.xi2pk_pyfftlog(iuspline(xs,self.xi_cross.get(xs,redshift,logdens)))(ks)

    def get_phm(self,ks,logdens,redshift):
        """get_phm

        Compute the halo-matter cross power spectrum :math:`P_{hm}(k;n_h)` for a mass threshold halo sample specified by the corresponding cumulative number density.

        Args:
            ks (numpy array): Wavenumbers in :math:`[h\mathrm{Mpc}^{-1}]`
            logdens (float): Logarithm of the cumulative halo number density of the halo sample taken from the most massive, :math:`\log_{10}[n_h/(h^{-1}\mathrm{Mpc})^3]`
            redshift (float): redshift at which the power spectrum is evaluated

        Returns:
            numpy array: Halo-matter cross power spectrum in :math:`[(h^{-1}\mathrm{Mpc})^{3}]`
        """
        xs = np.logspace(-4,3,4000)
        xi_dir = self._get_xicross_direct(xs,logdens,redshift)
        xi_tree = self._get_xicross_tree(xs,logdens,redshift)
        rswitch = min(60.,0.5 * self.cosmo.get_BAO_approx())
        xi = xi_dir * np.exp(-(xs/rswitch)**4) + xi_tree * (1-np.exp(-(xs/rswitch)**4))
        return pyfftlog_interface.xi2pk_pyfftlog(iuspline(xs,xi),logrmin = -4.0, logrmax = 3.0)(ks)

    def _get_phm_tree_cut(self,ks,logdens,redshift):
        xs = np.logspace(-4,3,4000)
        xi_tree = self._get_xicross_tree(xs,logdens,redshift)
        rswitch = min(60.,0.5 * self.cosmo.get_BAO_approx())
        xi = xi_tree * (1-np.exp(-(xs/rswitch)**4))
        return pyfftlog_interface.xi2pk_pyfftlog(iuspline(xs,xi),logrmin = -4.0, logrmax = 3.0)(ks)

    def _get_phm_direct_cut(self,ks,logdens,redshift):
        xs = np.logspace(-4,3,4000)
        xi_dir = self._get_xicross_direct(xs,logdens,redshift)
        rswitch = min(60.,0.5 * self.cosmo.get_BAO_approx())
        xi = xi_dir * np.exp(-(xs/rswitch)**4)
        return pyfftlog_interface.xi2pk_pyfftlog(iuspline(xs,xi),logrmin = -4.0, logrmax = 3.0)(ks)

    def get_phm_massthreshold(self,ks,Mthre,redshift):
        """get_phm_massthreshold

        Compute the halo-matter cross power spectrum :math:`P_{hm}(k;>M_\mathrm{th})` for a mass threshold halo sample.

        Args:
            ks (numpy array): Wavenumbers in :math:`[h\mathrm{Mpc}^{-1}]`
            Mthre (float): Minimum halo mass threshold in :math:`[h^{-1}M_\odot]`
            redshift (float): redshift at which the power spectrum is evaluated

        Returns:
            numpy array: Halo-matter cross power spectrum in :math:`[(h^{-1}\mathrm{Mpc})^{3}]`
        """
        logdens = np.log10(self.mass_to_dens(Mthre,redshift))
        return self.get_phm(ks,logdens,redshift)

    def get_phm_mass(self,ks,M,redshift):
        """get_phm_mass

        Compute the halo-matter cross power spectrum :math:`P_{hm}(k;M)` for halos with mass :math:`M`.

        Args:
            ks (numpy array): Wavenumbers in :math:`[h\mathrm{Mpc}^{-1}]`
            M (float): Halo mass in :math:`[h^{-1}M_\odot]`
            redshift (float): redshift at which the power spectrum is evaluated

        Returns:
            numpy array: Halo-matter cross power spectrum in :math:`[(h^{-1}\mathrm{Mpc})^{3}]`
        """
        Mp = M * 1.01
        Mm = M * 0.99
        logdensp = np.log10(self.mass_to_dens(Mp,redshift))
        logdensm = np.log10(self.mass_to_dens(Mm,redshift))
        pip = self.get_phm(ks,logdensp,redshift)
        pim = self.get_phm(ks,logdensm,redshift)
        return (pim * 10**logdensm - pip * 10**logdensp) / (10**logdensm - 10**logdensp)

    def _get_DeltaSigma_tree(self, R2d, logdens, redshift):
        return self.cosmo.get_Omega0() * self.cosmo.rho_cr / 1e12 * pyfftlog_interface.pk2xiproj_J2_pyfftlog(self._get_pkcross_tree_spline(logdens, redshift))(R2d)

    def _get_DeltaSigma_direct(self, R2d, logdens, redshift):
        xs = np.logspace(-3, 3, 2000)
        xi = self._get_xicross_direct(xs, logdens, redshift)
        pk_spl = pyfftlog_interface.xi2pk_pyfftlog(iuspline(xs, xi))
        return self.cosmo.get_Omega0() * self.cosmo.rho_cr / 1e12 * pyfftlog_interface.pk2xiproj_J2_pyfftlog(pk_spl)(R2d)

    def get_DeltaSigma(self, R2d, logdens, redshift):
        """get_DeltaSigma

        Compute the halo-galaxy lensing signal, the excess surface mass density, :math:`\Delta\Sigma(R;n_h)`, for a mass threshold halo sample specified by the corresponding cumulative number density.


        Args:
            R2d (numpy array): 2 dimensional projected separation in :math:`h^{-1}\mathrm{Mpc}`
            logdens (float): Logarithm of the cumulative halo number density taken from the most massive, :math:`\log_{10}[n_h/(h^{-1}\mathrm{Mpc})^3]`
            redshift (float): redshift at which the lens halos are located

        Returns:
            numpy array: excess surface mass density in :math:`[h M_\odot \mathrm{pc}^{-2}]`
        """
        xs = np.logspace(-3, 3, 2000)
        xi_tot = self.get_xicross(xs, logdens, redshift)
        pk_spl = pyfftlog_interface.xi2pk_pyfftlog(iuspline(xs, xi_tot))
        return self.cosmo.get_Omega0() * self.cosmo.rho_cr / 1e12 * pyfftlog_interface.pk2xiproj_J2_pyfftlog(pk_spl)(R2d)

    def get_DeltaSigma_massthreshold(self, R2d, Mthre, redshift):
        """get_DeltaSigma_massthreshold

        Compute the halo-galaxy lensing signal, the excess surface mass density, :math:`\Delta\Sigma(R;>M_\mathrm{th})`, for a mass threshold halo sample.

        Args:
            R2d (numpy array): 2 dimensional projected separation in :math:`h^{-1}\mathrm{Mpc}`
            Mthre (float): Minimum halo mass threshold in :math:`[h^{-1}M_\odot]`
            redshift (float): redshift at which the lens halos are located

        Returns:
            numpy array: excess surface mass density in :math:`[h M_\odot \mathrm{pc}^{-2}]`
        """
        logdens = np.log10(self.mass_to_dens(Mthre, redshift))
        return self.get_DeltaSigma(R2d, logdens, redshift)

    def get_DeltaSigma_mass(self, R2d, M, redshift):
        """get_DeltaSigma_mass

        Compute the halo-galaxy lensing signal, the excess surface mass density, :math:`\Delta\Sigma(R;M)`, for halos with mass :math:`M`.

        Args:
            R2d (numpy array): 2 dimensional projected separation in :math:`h^{-1}\mathrm{Mpc}`
            M (float): Halo mass in :math:`[h^{-1}M_\odot]`
            redshift (float): redshift at which the lens halos are located

        Returns:
            numpy array: excess surface mass density in :math:`[h M_\odot \mathrm{pc}^{-2}]`
        """
        Mp = M * 1.01
        Mm = M * 0.99
        logdensp = np.log10(self.mass_to_dens(Mp, redshift))
        logdensm = np.log10(self.mass_to_dens(Mm, redshift))
        DSp = self.get_DeltaSigma(R2d, logdensp, redshift)
        DSm = self.get_DeltaSigma(R2d, logdensm, redshift)
        return (DSm * 10**logdensm - DSp * 10**logdensp) / (10**logdensm - 10**logdensp)

    def _get_Sigma_tree(self, R2d, logdens, redshift):
        return self.cosmo.get_Omega0() * self.cosmo.rho_cr / 1e12 * pyfftlog_interface.pk2xiproj_J0_pyfftlog(self._get_pkcross_tree_spline(logdens, redshift))(R2d)

    def _get_Sigma_direct(self, R2d, logdens, redshift):
        xs = np.logspace(-3, 3, 2000)
        xi = self._get_xicross_direct(xs, logdens, redshift)
        pk_spl = pyfftlog_interface.xi2pk_pyfftlog(iuspline(xs, xi))
        return self.cosmo.get_Omega0() * self.cosmo.rho_cr / 1e12 * pyfftlog_interface.pk2xiproj_J0_pyfftlog(pk_spl)(R2d)

    def get_Sigma(self, R2d, logdens, redshift):
        """get_Sigma

        Compute the surface mass density, :math:`\Sigma(R;n_h)`, for a mass threshold halo sample specified by the corresponding cumulative number density.

        Args:
            R2d (numpy array): 2 dimensional projected separation in :math:`h^{-1}\mathrm{Mpc}`
            logdens (float): Logarithm of the cumulative halo number density taken from the most massive, :math:`\log_{10}[n_h/(h^{-1}\mathrm{Mpc})^3]`
            redshift (float): redshift at which the lens halos are located

        Returns:
            numpy array: surface mass density in :math:`[h M_\odot \mathrm{pc}^{-2}]`
        """
        xs = np.logspace(-3, 3, 2000)
        xi_tot = self._get_xicross(xs, logdens, redshift)
        pk_spl = pyfftlog_interface.xi2pk_pyfftlog(iuspline(xs, xi_tot))
        return self.cosmo.get_Omega0() * self.cosmo.rho_cr / 1e12 * pyfftlog_interface.pk2xiproj_J0_pyfftlog(pk_spl)(R2d)

    def get_Sigma_massthreshold(self, R2d, Mthre, redshift):
        """get_Sigma_massthreshold

        Compute the surface mass density, :math:`\Sigma(R;>M_\mathrm{th})`, for a mass threshold halo sample.

        Args:
            R2d (numpy array): 2 dimensional projected separation in :math:`h^{-1}\mathrm{Mpc}`
            Mthre (float): Minimum halo mass threshold in :math:`[h^{-1}M_\odot]`
            redshift (float): redshift at which the lens halos are located

        Returns:
            numpy array: surface mass density in :math:`[h M_\odot \mathrm{pc}^{-2}]`
        """
        logdens = np.log10(self.mass_to_dens(Mthre, redshift))
        return self.get_Sigma(R2d, logdens, redshift)

    def get_Sigma_mass(self, R2d, M, redshift):
        """get_Sigma_mass

        Compute the surface mass density, :math:`\Sigma(R;M)`, for halos with mass :math:`M`.

        Args:
            R2d (numpy array): 2 dimensional projected separation in :math:`h^{-1}\mathrm{Mpc}`
            M (float): Halo mass in :math:`[h^{-1}M_\odot]`
            redshift (float): redshift at which the lens halos are located

        Returns:
            numpy array: surface mass density in :math:`[h M_\odot \mathrm{pc}^{-2}]`
        """
        Mp = M * 1.01
        Mm = M * 0.99
        logdensp = np.log10(self.mass_to_dens(Mp, redshift))
        logdensm = np.log10(self.mass_to_dens(Mm, redshift))
        Sp = self.get_Sigma(R2d, logdensp, redshift)
        Sm = self.get_Sigma(R2d, logdensm, redshift)
        return (Sm * 10**logdensm - Sp * 10**logdensp) / (10**logdensm - 10**logdensp)

    def _get_gamma1_dm(self, k, redshift):
        return self.g1.get_dm(k, redshift)

    def _get_bd(self, logdens, redshift):
        return self.g1.get_bd(redshift, logdens)

    def get_bias(self, logdens, redshift):
        """get_bias

        Compute the linear bias for a mass threshold halo sample specified by the corresponding cumulative number density.

        Args:
            logdens (float): Logarithm of the cumulative halo number density taken from the most massive, :math:`\log_{10}[n_h/(h^{-1}\mathrm{Mpc})^3]`
            redshift (float): redshift at which the lens halos are located

        Returns:
            float: linear bias factor
        """
        return self.g1.get_bias(redshift, logdens)

    def _get_gamma1_h(self, k, logdens, redshift):
        return self.g1.get(k, redshift, logdens)

    def Dgrowth_from_z(self, z):
        """Dgrowth_from_z

        Compute the linear growth factor, D_+, at redshift z.
        Normalized to unity at z=0.

        Args:
            z: redshift
        Returns:
            float: linear growth factor
        """
        return self.cosmo.Dgrowth_from_z(z)

    def Dgrowth_from_a(self, a):
        """Dgrowth_from_a

        Compute the linear growth factor, D_+, at scale factor a.
        Normalized to unity at z=0.

        Args:
            a: scale factor normalized to unity at present.
        Returns:
            float: linear growth factor
        """
        return self.cosmo.Dgrowth_from_a(a)

    def _Dgrowth_from_snapnum(self, i):
        return self.cosmo.Dgrowth_from_snapnum(i)

    def f_from_z(self, z):
        """f_from_z

        Compute the linear growth rate, :math:`f = \mathrm{d}\ln D_+/\mathrm{d}\ln a`, at redshift z.

        Args:
            z: redshift
        Returns:
            float: linear growth rate
        """
        return self.cosmo.f_from_z(z)

    def f_from_a(self, a):
        """f_from_a

        Compute the linear growth rate, :math:`f = \mathrm{d}\ln D_+/\mathrm{d}\ln a`, at scale factor a.

        Args:
            a: scale factor normalized to unity at present.
        Returns:
            float: linear growth rate
        """
        return self.cosmo.f_from_a(a)

    def _f_from_snapnum(self, i):
        return self.cosmo.f_from_snampum(i)

    def get_cosmology(self):
        """get_cosmology

        Obtain the cosmological parameters currently set to the emulator.

        Returns:
            numpy array: Cosmological parameters :math:`(\omega_b, \omega_{m}, \Omega_{de}, \ln(10^{10}A_s), n_s, w)`
        """
        return self.cosmo.get_cosmology()

    def get_sigma8(self, logkmin=-4, logkmax=1, nint=100):
        """get_sigma8

            Compute :math:`\sigma_8` for the current cosmology.

            Args:
                logkmin (float, optional): log10 of the minimum wavenumber for the integral (default=-4)
                logkmin (float, optional): log10 of the maximum wavenumber for the integral (default=1)
                nint (int, optional): Number of samples taken for the trapz integration (default=100)

            Returns:
                float: :math:`\sigma_8`
        """
        R = 8.
        ks = np.logspace(logkmin, logkmax, nint)
        logks = np.log(ks)
        kR = ks * R
        integrant = ks**3*self.get_pklin(ks)*_window_tophat(kR)**2
        return np.sqrt(integrate.trapz(integrant, logks)/(2.*np.pi**2))

def _window_tophat(kR):
    return 3.*(np.sin(kR)-kR*np.cos(kR))/kR**3
