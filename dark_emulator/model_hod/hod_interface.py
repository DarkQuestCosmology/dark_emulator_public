import sys
import os
import copy
import logging
import numpy as np
#sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "../")
from ..darkemu import cosmo_util
from ..darkemu.de_interface import base_class
from ..darkemu.hmf import hmf_gp
from ..darkemu.auto import auto_gp
from ..darkemu.cross import cross_gp
from ..darkemu.gamma1 import gamma1_gp
from ..darkemu.xinl import xinl_gp
from ..darkemu.pklin import pklin_gp
from ..darkemu.cosmo_util import cosmo_class
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RectBivariateSpline as rbs
from scipy import integrate
from scipy import special
from scipy import ndimage
from scipy import optimize
from .. import pyfftlog_interface
try:
    from colossus.cosmology import cosmology as colcosmology
    from colossus.halo import concentration
except:
    print('colossus is not installed.')


path_baryon = os.path.dirname(__file__)
path_baryon = os.path.join(path_baryon, "..", "baryon")
sys.path.append(path_baryon)
try:
    from wk_baryon import return_wk
except ImportError:
    pass

rho_cr = 2.77536627e11 # [M_sun/h] / [Mpc/h]^3\n"

def _get_uk(k, cvir, rvir):
    f = 1./(np.log(1.+cvir) - cvir/(1.+cvir))
    eta = k*rvir/cvir
    si_eta_1pc, ci_eta_1pc = special.sici(eta*(1.+cvir))
    si_eta, ci_eta = special.sici(eta)
    return f*(np.sin(eta)*(si_eta_1pc - si_eta) + np.cos(eta)*(ci_eta_1pc - ci_eta) - np.sin(eta*cvir)/eta/(1.+cvir))

class darkemu_x_hod(base_class):
    """darkemu_x_hod

       This class holds cosmological parameters (see ``set_cosmology()``), HOD parameters, and other galaxy parameters (see ``set_galaxy()``), and computes galaxy-galaxy lensing, galaxy-galaxy clustering signal, and related correlation functions  based on these parameters. This class can be initialized through a dictionary that specifies the following configurations. With the default values, one can get :math:`\Delta\Sigma` and :math:`w_p` with an enough accuracy for `the HSC S16A analysis <https://ui.adsabs.harvard.edu/abs/2021arXiv210100113M/abstract>`_.

       - **fft_num** (*int*): Sampling in fftlog in unit of 2048 (default: 8).
       - **fft_logrmin_1h** (*float*): Minimum :math:`\log_{10}(r/[h^{-1}\mathrm{Mpc}])` used in internal 1-halo term calculation by fftlog (default: -5.0).
       - **fft_logrmax_1h** (*float*): Maximum :math:`\log_{10}(r/[h^{-1}\mathrm{Mpc}])` used in internal 1-halo term calculation by fftlog (default: 3.0).
       - **fft_logrmin_2h** (*float*): Minimum :math:`\log_{10}(r/[h^{-1}\mathrm{Mpc}])` used in internal 2-halo term calculation by fftlog (default: -3.0).
       - **fft_logrmax_2h** (*float*): Maximum :math:`\log_{10}(r/[h^{-1}\mathrm{Mpc}])` used in internal 2-halo term calculation by fftlog (default: 3.0).
       - **M_int_logMmin** (*float*): Minimum :math:`\log_{10}(M_\mathrm{halo}/[h^{-1}\mathrm{M}_{\odot}])` used in the integration across halo mass (default: 12.0).
       - **M_int_logMax** (*float*): Maximum :math:`\log_{10}(M_\mathrm{halo}/[h^{-1}\mathrm{M}_{\odot}])` used in the integration across halo mass (default: 15.9).
       - **M_int_k** (*int*): Sampling in the integration across halo mass which sets :math:`2^{\mathrm{M\_int\_k}}` (default: 5).
       - **c-M_relation** (*str*): Concentration-mass relation used for satellite distribution when NFW is used (see ``set_galaxy()``; default: 'diemer15'). The concentration is internally computed using `colossus <https://bdiemer.bitbucket.io/colossus/>`_, and a user can use a model listed in ``concentration models`` in `this webpage <https://bdiemer.bitbucket.io/colossus/halo_concentration.html>`_.

       Args:
           config (dict): a dictionary to specify configurations
    """
    def __init__(self, config=None):
        logging.basicConfig(level=logging.DEBUG)
        self.redshift = None

        # set up default config
        self.config = dict()
        self.config["fft_num"] = 8 # Sunao : changed from 5 to 8 for fftlog in wp_2hcs, wp_2hss, xi_2hcs and xi_2hss to converge at -1.0 < log10(r) < 1.0.
        self.config["fft_logrmin_1h"] = -5.0
        self.config["fft_logrmax_1h"] = 3.0
        self.config["fft_logrmin_2h"] = -3.0
        self.config["fft_logrmax_2h"] = 3.0
        self.config["M_int_logMmin"] = 12.
        self.config["M_int_logMmax"] = 15.9
        self.config["M_int_k"] = 5
        self.config["M_int_algorithm"] = "romberg"
        #self.config["los_int_pi_max"] = None
        #self.config["los_int_algorithm"] = "trapz"
        self.config["c-M_relation"] = "diemer15"
        self.config["p_hm_apodization"] = None # apodization with the scale sigma_k=self.config["p_hm_apodization"]/R200 is applied for satelite distribution computed from emulator. See def _compute_p_hm_satdist_emu() for details
        # override if specified in input
        if config is not None:
            for key in list(config.keys()):
                self.config[key] = config[key]

        # internal config parameters
        self.config["hmf_int_algorithm"] = "trapz"

        # set up variables shared with multiple methods
        self.Mh = np.logspace(
            self.config["M_int_logMmin"], self.config["M_int_logMmax"], 2**self.config["M_int_k"]+1)
        self.dlogMh = np.log(self.Mh[1]) - np.log(self.Mh[0])

        self.fftlog_1h = pyfftlog_interface.fftlog(self.config['fft_num'], logrmin=self.config['fft_logrmin_1h'], logrmax=self.config['fft_logrmax_1h'], kr=1)
        self.fftlog_2h = pyfftlog_interface.fftlog(self.config['fft_num'], logrmin=self.config['fft_logrmin_2h'], logrmax=self.config['fft_logrmax_2h'], kr=1)

        if self.config["M_int_algorithm"] == "romberg":
            self.do_integration = integrate.romb
        elif self.config["M_int_algorithm"] == "simpson":
            self.do_integration = integrate.simps

        self.k_1h_mat = np.tile(self.fftlog_1h.k, (len(self.Mh), 1))
        self.k_2h_mat = np.tile(self.fftlog_2h.k, (len(self.Mh), 1))
        self.Mh_mat = np.tile(self.Mh, (len(self.fftlog_2h.k), 1)).transpose()

        self.gparams = None
        self.initialized = False
        self.cparams_orig = np.zeros((1, 6))
        super(darkemu_x_hod, self).__init__()

    # The following flags tell if dndM and power spectrum should be recomputed when cosmology or redshift is varied.
    def _initialize_cosmology_computation_flags(self):
        self.dndM_spl_computed = False
        self.d_to_m_interp_computed = False
        self.dndM_computed = False
        self.p_hh_computed = False
        self.xi_hm_computed = False
        self.p_hm_computed = False
        self.p_hm_satdist_computed = False
        self.ng_computed = False
        self.ng_cen_computed = False
        self.logdens_computed = False

    def set_cosmology(self, cparams):
        if np.any(self.cosmo.get_cosmology() != cparams.reshape(1, 6)) or np.any(self.cparams_orig != cparams.reshape(1, 6)) or (self.initialized == False):
            self.do_linear_correction, cparams_tmp = cosmo_util.test_cosm_range(
                cparams, return_edges=True)
            if cosmo_util.test_cosm_range_linear(cparams):
                raise RuntimeError(
                    "Cosmological parameters are out of supported range.")

            self.cparams_orig = np.copy(cparams)
            if self.do_linear_correction:
                cparams = cparams_tmp
                logging.info("%s is out of the supported range. Instaead use %s and apply linear correction later." % (
                    self.cparams_orig, cparams))
                # compute pklin for cparams_orig here
                super(darkemu_x_hod, self).set_cosmology(self.cparams_orig)
                self.pm_lin_k_1h_out_of_range = self.get_pklin(self.fftlog_1h.k)
                self.pm_lin_k_2h_out_of_range = self.get_pklin(self.fftlog_2h.k)
                self.cosmo_orig = copy.deepcopy(self.cosmo)
                self.massfunc_cosmo_edge = hmf_gp()
                self.massfunc_cosmo_edge.set_cosmology(self.cosmo)
            super(darkemu_x_hod, self).set_cosmology(cparams)
            if self.do_linear_correction:
                self.massfunc.set_cosmology(self.cosmo_orig)
            self.rho_m = (1. - cparams[0][2])*rho_cr
            self.R200 = (3*self.Mh/(4.*np.pi*self.rho_m)/200.)**(1./3.)
            self.R200_mat = np.tile(self.R200, (len(self.fftlog_2h.k), 1)).transpose()
            self._initialize_cosmology_computation_flags()
            self.initialized = True
        else:
            logging.info(
                "Got same cosmology. Keep quantities already computed.")

    def set_galaxy(self, gparams):
        """set_galaxy

           This class sets galaxy parameter through a dictionary. See `Miyatake et al (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210100113M/abstract>`_ for the definition of galaxy parameters. Here is the list of keys.

           - HOD parameters:

                - **logMmin** (*float*): Central HOD parameter, :math:`\log M_\mathrm{min}`
                - **sigma_sq** (*float*): Central HOD parameter, :math:`\sigma^2`
                - **logM1** (*float*): Satellite HOD parameter, :math:`\log M_1`
                - **alpha** (*float*): Satellite HOD parameter, :math:`\\alpha`
                - **kappa** (*float*): Satellite HOD parameter, :math:`\kappa`

           - off-centering parameters:

                - **poff** (*float*): Fraction of off-centered galaxies, :math:`p_\mathrm{off}`
                - **Roff** (*float*): Characteristic scale of off-centered galaxies with respect to :math:`R_\mathrm{200m}`, :math:`R_\mathrm{off}`

           - satellite distribution

                - **sat_dist_type** (*float*): Profile of satellite distribution. Valid values are 'emulator' or 'NFW'. When 'NFW', concentration is specified in **config** parameter (see ``dark_emulator.model_hod.hod_interface.darkemu_x_hod``)

           - incompleteness parameters

                - **alpha_inc** (*float*): Incompleteness parameter, :math:`\\alpha_\mathrm{inc}`
                - **logM_inc** (*float*): Incompleteness parameter, :math:`\log M_\mathrm{inc}`

           Args:
              gparams (dict): a dictionary to specify galaxy parameters
        """
        # gparams includes galaxy-related parameters such as HOD, offcentering, incompleteness, and pi_max.
        if self.gparams != gparams:
            self.gparams = gparams
            self.HOD_computed = False
            self.ng_computed = False
            self.ng_cen_computed = False
            self.wp_1hcs_computed = False
            self.wp_1hss_computed = False
            self.wp_2hcc_computed = False
            self.wp_2hcs_computed = False
            self.wp_2hss_computed = False
            self.ds_cen_computed = False
            self.ds_cen_off_computed = False
            self.ds_sat_computed = False
        else:
            logging.info(
                "Got same galaxy parameters. Keep quantities already computed.")

    def _get_xiauto_direct_noint(self, logdens1, logdens2, redshift):
        sel1 = (logdens1 < -5.75)
        sel2 = (logdens2 < -5.75)
        logdens1_mod = np.copy(logdens1)
        logdens1_mod[sel1] = -5.75
        logdens2_mod = np.copy(logdens2)
        logdens2_mod[sel2] = -5.75
        xi_dir = self.xi_auto.getNoInterpol(
            redshift, logdens1_mod, logdens2_mod).transpose()
        if sel1.sum() != 0:
            xi_dir[sel1] = xi_dir[sel1] * np.tile(self.g1.bias_ratio_arr(
                redshift, logdens1[sel1]), (xi_dir.shape[1], 1)).transpose()
        if sel2.sum() != 0:
            xi_dir[sel2] = xi_dir[sel2] * np.tile(self.g1.bias_ratio_arr(
                redshift, logdens2[sel2]), (xi_dir.shape[1], 1)).transpose()
        return xi_dir

    def _get_xicross_direct_noint(self, logdens, redshift):
        return self.xi_cross.getNoInterpol(redshift, logdens).transpose()

    def _compute_dndM_spl(self, redshift):
        self.dndM_spl = ius(np.log(self.massfunc.Mlist),
                            np.log(self.massfunc.get_dndM(redshift)))
        self.dndM_spl_computed = True

    def _compute_dndM(self, redshift):
        if self.dndM_spl_computed == False:
            self._compute_dndM_spl(redshift)
        self.dndM = np.exp(self.dndM_spl(np.log(self.Mh)))
        self.dndM_mat = np.tile(self.dndM, (len(self.fftlog_2h.k), 1)).transpose()
        self.dndM_computed = True

    def _convert_mass_to_dens(self, mass_thre, redshift, integration="quad"):
        if self.dndM_spl_computed == False:
            self._compute_dndM_spl(redshift)
        dndM_interp = self.dndM_spl
        if integration == "quad":
            dens = integrate.quad(lambda t: np.exp(
                dndM_interp(np.log(t))), mass_thre, 1e16, epsabs=1e-5)[0]
        elif integration == "trapz":
            t = np.logspace(np.log10(mass_thre), 16, 512)
            dlogt = np.log(t[1]) - np.log(t[0])
            dens = integrate.trapz(np.exp(dndM_interp(np.log(t)))*t, dx=dlogt)
        else:
            raise RuntimeError(
                "You should specify valid integration algorithm: quad or trapz")
        return dens

    def _convert_dens_to_mass(self, dens, redshift, nint=20, integration="quad"):
        if self.dndM_spl_computed == False:
            self._compute_dndM_spl(redshift)
        if self.d_to_m_interp_computed == False:
            dndM_interp = self.dndM_spl
            mlist = np.linspace(12., 15.95, nint)
            dlist = np.log(np.array([self._convert_mass_to_dens(
                10**mlist[i], redshift, integration=integration) for i in range(nint)]))
            self.d_to_m_interp = ius(-dlist, mlist)
            self.d_to_m_interp_computed = True
        return 10**self.d_to_m_interp(-np.log(dens))

    def _compute_logdens(self, redshift):
        self.logdens = np.log10([self._convert_mass_to_dens(
            self.Mh[i], redshift, integration=self.config["hmf_int_algorithm"]) for i in range(len(self.Mh))])
        self.logdens_computed = True

    def _compute_ng(self):  # galaxy number density
        self.ng = self.do_integration(
            self.dndM * (self.Ncen + self.Nsat) * self.Mh, dx=self.dlogMh)
        self.ng_computed = True

    def _compute_ng_cen(self):  # central galaxy number density
        self.ng_cen = self.do_integration(
            self.dndM * self.Ncen * self.Mh, dx=dlogMh)
        self.ng_cen_computed = True

    def _compute_p_hh_spl_experiment(self, redshift):
        # first compute xi_dir with minimum resolution of mass and radial bins, i.e., the resolution used in the emulator.

        logdens_xi_auto = self.xi_auto.logdens_list
        Mh_xi_auto = [self._convert_dens_to_mass(
            10**logdens_xi_auto[i], redshift, integration=self.config["hmf_int_algorithm"]) for i in range(len(logdens_xi_auto))]
        logdens_p = np.log10([self._convert_mass_to_dens(1.02*Mh_xi_auto[i], redshift,
                                                integration=self.config["hmf_int_algorithm"]) for i in range(len(logdens_xi_auto))])
        logdens_m = np.log10([self._convert_mass_to_dens(0.98*Mh_xi_auto[i], redshift,
                                                integration=self.config["hmf_int_algorithm"]) for i in range(len(logdens_xi_auto))])

        logdens1 = list()
        logdens2 = list()
        for i in range(len(logdens_xi_auto)):
            for j in range(len(logdens_xi_auto)):
                if logdens_xi_auto[i] < logdens_xi_auto[j]:
                    continue
                logdens1.append([logdens_m[i], logdens_m[i],
                                 logdens_p[i], logdens_p[i]])
                logdens2.append([logdens_m[j], logdens_p[j],
                                 logdens_m[j], logdens_p[j]])
        xi_dir_all = self._get_xiauto_direct_noint(np.concatenate(
            logdens1), np.concatenate(logdens2), redshift)

        k = 0
        xi_dir = np.zeros((len(logdens_xi_auto), len(
            logdens_xi_auto), len(self.xi_auto.logrscale)))
        for i in range(len(logdens_xi_auto)):
            for j in range(len(logdens_xi_auto)):
                if logdens_xi_auto[i] < logdens_xi_auto[j]:
                    continue
                logdens_1p = logdens_p[i]
                logdens_1m = logdens_m[i]
                logdens_2p = logdens_p[j]
                logdens_2m = logdens_m[j]
                dens_1p, dens_1m, dens_2p, dens_2m = 10**logdens_1p, 10**logdens_1m, 10**logdens_2p, 10**logdens_2m
                denom = dens_1m * dens_2m - dens_1m * dens_2p - \
                    dens_1p * dens_2m + dens_1p * dens_2p

                # calculate xi_dir
                xi_dir_mm = xi_dir_all[4*k]
                xi_dir_mp = xi_dir_all[4*k+1]
                xi_dir_pm = xi_dir_all[4*k+2]
                xi_dir_pp = xi_dir_all[4*k+3]
                numer = xi_dir_mm * dens_1m * dens_2m - xi_dir_mp * dens_1m * dens_2p - \
                    xi_dir_pm * dens_1p * dens_2m + xi_dir_pp * dens_1p * dens_2p
                xi_dir[i, j] = numer/denom
                if i != j:
                    xi_dir[j, i] = xi_dir[i, j]
                k += 1

        # next compute xi_tree.
        pm_lin = self.get_pklin(self.fftlog_2h.k)

        logdens_g1 = self.g1.logdens_list
        Mh_g1 = [self._convert_dens_to_mass(
            10**logdens_g1[i], redshift, integration=self.config["hmf_int_algorithm"]) for i in range(len(logdens_g1))]
        logdens_p = np.log10([self._convert_mass_to_dens(
            1.02*Mh_g1[i], redshift, integration=self.config["hmf_int_algorithm"]) for i in range(len(logdens_g1))])
        logdens_m = np.log10([self._convert_mass_to_dens(
            0.98*Mh_g1[i], redshift, integration=self.config["hmf_int_algorithm"]) for i in range(len(logdens_g1))])

        g = list()
        gp = self.g1.get(self.fftlog_2h.k, redshift, logdens_p)
        gm = self.g1.get(self.fftlog_2h.k, redshift, logdens_m)
        for i in range(len(gp)):
            g.append([gm[i], gp[i]])

        xi_tree = np.zeros((len(logdens_g1), len(logdens_g1), len(self.fftlog_2h.k)))
        for i in range(len(logdens_g1)):
            for j in range(len(logdens_g1)):
                if logdens_g1[i] < logdens_g1[j]:
                    continue
                logdens_1p = logdens_p[i]
                logdens_1m = logdens_m[i]
                logdens_2p = logdens_p[j]
                logdens_2m = logdens_m[j]
                dens_1p, dens_1m, dens_2p, dens_2m = 10**logdens_1p, 10**logdens_1m, 10**logdens_2p, 10**logdens_2m
                denom = dens_1m * dens_2m - dens_1m * dens_2p - \
                    dens_1p * dens_2m + dens_1p * dens_2p

                # calculate_xi_tree
                g_1p = g[i][1]
                g_1m = g[i][0]
                g_2p = g[j][1]
                g_2m = g[j][0]
                ph_tree_mm = g_1m * g_2m * pm_lin
                ph_tree_mp = g_1m * g_2p * pm_lin
                ph_tree_pm = g_1p * g_2m * pm_lin
                ph_tree_pp = g_1p * g_2p * pm_lin
                numer = ph_tree_mm * dens_1m * dens_2m - ph_tree_mp * dens_1m * dens_2p - \
                    ph_tree_pm * dens_1p * dens_2m + ph_tree_pp * dens_1p * dens_2p
                ph_tree = numer/denom
                xi_tree[i, j] = self.fftlog_2h.pk2xi(ph_tree)
                #xi_tree[i, j] = fftLog.pk2xi_fftlog_array(
                #    self.fftlog_2h.k, self.fftlog_2h.r, ph_tree, self.fftlog_2h.kr, self.fftlog_2h.dlnk)
                if i != j:
                    xi_tree[j, i] = xi_tree[i, j]

        # combine xi_dir and xi_tree to compute xi_hh
        xi_dir_mass_resampled = list()
        for i in range(len(self.xi_auto.logrscale)):
            xi_dir_mass_resampled.append(
                rbs(-logdens_xi_auto, -logdens_xi_auto, xi_dir[:, :, i])(-logdens_g1, -logdens_g1))
        xi_dir_mass_resampled = np.array(xi_dir_mass_resampled)

        xi_dir_mass_r_resampled = np.zeros(
            (len(logdens_g1), len(logdens_g1), len(self.fftlog_2h.r)))
        for i in range(len(logdens_g1)):
            for j in range(len(logdens_g1)):
                if logdens_g1[i] < logdens_g1[j]:
                    continue
                xi_dir_mass_r_resampled[i, j] = ius(
                    self.xi_auto.logrscale, xi_dir_mass_resampled[:, i, j], ext=3)(np.log(self.fftlog_2h.r))
                if i != j:
                    xi_dir_mass_r_resampled[j,
                                            i] = xi_dir_mass_r_resampled[i, j]

        rswitch = min(60., 0.5 * self.cosmo.get_BAO_approx())
        connection_factor_1 = np.exp(-(self.fftlog_2h.r/rswitch)**4)
        connection_factor_2 = 1-np.exp(-(self.fftlog_2h.r/rswitch)**4)
        xi_hh = xi_dir_mass_r_resampled * \
            connection_factor_1 + xi_tree * connection_factor_2
        p_hh = np.zeros((len(logdens_g1), len(logdens_g1), len(self.fftlog_2h.k)))
        for i in range(len(logdens_g1)):
            for j in range(len(logdens_g1)):
                if logdens_g1[i] < logdens_g1[j]:
                    continue
                p_hh[i, j] = self.fftlog_2h.xi2pk(xi_hh[i, j])
                #p_hh[i, j] = fftLog.xi2pk_fftlog_array(
                #    self.fftlog_2h.r, self.fftlog_2h.k, xi_hh[i, j], self.fftlog_2h.kr, self.fftlog_2h.dlnr)
                if i != j:
                    p_hh[j, i] = p_hh[i, j]

        # interpolate phh along halo mass
        self.p_hh_spl = list()
        for i in range(len(self.fftlog_2h.k)):
            self.p_hh_spl.append(rbs(-logdens_g1, -logdens_g1, p_hh[:, :, i]))

    def _compute_p_hh_spl(self, redshift):
        # first generate xi_dir with minimum resolution of mass and radian bins, i.e., the resolution used in the emulator.
        logdens_de = self.g1.logdens_list
        Mh_de = [self._convert_dens_to_mass(
            10**logdens_de[i], redshift, integration=self.config["hmf_int_algorithm"]) for i in range(len(logdens_de))]
        logdens_p = np.log10([self._convert_mass_to_dens(
            1.02*Mh_de[i], redshift, integration=self.config["hmf_int_algorithm"]) for i in range(len(logdens_de))])
        logdens_m = np.log10([self._convert_mass_to_dens(
            0.98*Mh_de[i], redshift, integration=self.config["hmf_int_algorithm"]) for i in range(len(logdens_de))])

        logdens1 = list()
        logdens2 = list()
        for i in range(len(logdens_de)):
            for j in range(len(logdens_de)):
                if logdens_de[i] < logdens_de[j]:
                    continue
                logdens1.append([logdens_m[i], logdens_m[i],
                                 logdens_p[i], logdens_p[i]])
                logdens2.append([logdens_m[j], logdens_p[j],
                                 logdens_m[j], logdens_p[j]])
        xi_dir_all = self._get_xiauto_direct_noint(np.concatenate(
            logdens1), np.concatenate(logdens2), redshift)

        pm_lin = self.get_pklin(self.fftlog_2h.k)

        g = list()
        gp = self.g1.get(self.fftlog_2h.k, redshift, logdens_p)
        gm = self.g1.get(self.fftlog_2h.k, redshift, logdens_m)

        if self.do_linear_correction:
            bias_correction = np.zeros(len(Mh_de))
            for i in range(len(Mh_de)):
                bias_correction[i] = _compute_tinker10_bias(redshift, Mh_de[i], self.massfunc)/_compute_tinker10_bias(redshift, Mh_de[i], self.massfunc_cosmo_edge)

        for i in range(len(gp)):
            g.append([gm[i], gp[i]])

        rswitch = min(60., 0.5 * self.cosmo.get_BAO_approx())
        connection_factor_1 = np.exp(-(self.fftlog_2h.r/rswitch)**4)
        connection_factor_2 = 1-np.exp(-(self.fftlog_2h.r/rswitch)**4)
        p_hh = np.zeros((len(logdens_de), len(logdens_de), len(self.fftlog_2h.k)))

        k = 0
        for i in range(len(logdens_de)):
            for j in range(len(logdens_de)):
                if logdens_de[i] < logdens_de[j]:
                    continue
                logdens_1p = logdens_p[i]
                logdens_1m = logdens_m[i]
                logdens_2p = logdens_p[j]
                logdens_2m = logdens_m[j]
                dens_1p, dens_1m, dens_2p, dens_2m = 10**logdens_1p, 10**logdens_1m, 10**logdens_2p, 10**logdens_2m
                denom = dens_1m * dens_2m - dens_1m * dens_2p - \
                    dens_1p * dens_2m + dens_1p * dens_2p

                # calculate xi_dir
                xi_dir_mm = xi_dir_all[4*k]
                xi_dir_mp = xi_dir_all[4*k+1]
                xi_dir_pm = xi_dir_all[4*k+2]
                xi_dir_pp = xi_dir_all[4*k+3]
                numer = xi_dir_mm * dens_1m * dens_2m - xi_dir_mp * dens_1m * dens_2p - \
                    xi_dir_pm * dens_1p * dens_2m + xi_dir_pp * dens_1p * dens_2p
                xi_dir_tmp = numer/denom
                xi_dir_spl = ius(self.xi_auto.logrscale, xi_dir_tmp, ext=3)
                xi_dir = xi_dir_spl(np.log(self.fftlog_2h.r))
                #p_hh_dir[i,j] = fftLog.xi2pk_fftlog_array(self.r_2h, self.k_2h, xi_dir, self.kr, self.dlnr_2h)

                # calculate_xi_tree
                g_1p = g[i][1]
                g_1m = g[i][0]
                g_2p = g[j][1]
                g_2m = g[j][0]
                ph_tree_mm = g_1m * g_2m * pm_lin
                ph_tree_mp = g_1m * g_2p * pm_lin
                ph_tree_pm = g_1p * g_2m * pm_lin
                ph_tree_pp = g_1p * g_2p * pm_lin
                numer = ph_tree_mm * dens_1m * dens_2m - ph_tree_mp * dens_1m * dens_2p - \
                    ph_tree_pm * dens_1p * dens_2m + ph_tree_pp * dens_1p * dens_2p
                ph_tree = numer/denom
                #p_hh_tree[i,j] = ph_tree

                xi_tree = self.fftlog_2h.pk2xi(ph_tree)
                #xi_tree = fftLog.pk2xi_fftlog_array(
                #    self.k_2h, self.r_2h, ph_tree, self.kr, self.dlnk_2h)
                xi_hh = xi_dir * connection_factor_1 + xi_tree * connection_factor_2

                if self.do_linear_correction:
                    xi_hh *= bias_correction[i]*bias_correction[j]

                p_hh[i, j] = self.fftlog_2h.xi2pk(xi_hh)
                #p_hh[i, j] = fftLog.xi2pk_fftlog_array(
                #    self.r_2h, self.k_2h, xi_hh, self.kr, self.dlnr_2h)
                if i != j:
                    p_hh[j, i] = p_hh[i, j]
                k += 1

                if self.do_linear_correction:
                    p_hh[i, j] *= (self.pm_lin_k_2h_out_of_range/pm_lin)

        self.p_hh_base = p_hh

    def _compute_p_hh(self, redshift):
        self._compute_p_hh_spl(redshift)

        logdens_de = self.g1.logdens_list
        logdens = self.logdens

        if True:  # fastest so far
            self.p_hh_tmp = np.zeros(
                (len(logdens_de), len(self.Mh), len(self.fftlog_2h.k)))
            for i in range(len(logdens_de)):
                self.p_hh_tmp[i] = rbs(-logdens_de, self.fftlog_2h.k,
                                       self.p_hh_base[i])(-logdens, self.fftlog_2h.k)

            self.p_hh = np.zeros((len(self.Mh), len(self.Mh), len(self.fftlog_2h.k)))
            for i in range(len(self.Mh)):
                self.p_hh[:, i] = rbs(-logdens_de, self.fftlog_2h.k,
                                      self.p_hh_tmp[:, i, :])(-logdens, self.fftlog_2h.k)

        if False:  # slow!
            self.p_hh_tmp = np.zeros(
                (len(logdens_de), len(self.Mh), len(self.fftlog_2h.k)))
            for i in range(len(logdens_de)):
                for j in range(len(self.fftlog_2h.k)):
                    self.p_hh_tmp[i, :, j] = ius(-logdens_de,
                                                 self.p_hh_base[i, :, j])(-logdens)

            self.p_hh = np.zeros((len(self.Mh), len(self.Mh), len(self.fftlog_2h.k)))
            for i in range(len(self.Mh)):
                for j in range(len(self.fftlog_2h.k)):
                    self.p_hh[:, i, j] = ius(-logdens_de,
                                             self.p_hh_tmp[:, i, j])(-logdens)

        if False:  # slow!
            p_hh_new = np.zeros((len(logdens), len(logdens), len(self.fftlog_2h.k)))
            for i in range(len(self.fftlog_2h.k)):
                p_hh_new[:, :, i] = rbs(-logdens_de, -logdens_de,
                                        self.p_hh_base[:, :, i])(-logdens, -logdens)
            self.p_hh = p_hh_new

        if False:  # slow! vectorize does not really help
            p_hh_new = np.zeros((len(logdens), len(logdens), len(self.fftlog_2h.k)))

            def _myrbs(zin, xin, yin, xout, yout):
                return rbs(xin, yin, zin)(xout, yout)
            vmyrbs = np.vectorize(_myrbs, excluded=[
                                  "xin", "yin", "xout", "yout"], signature='(n,n),(n),(n),(m),(m)->(m,m)')
            p_hh_new = vmyrbs(self.p_hh_base.transpose(
                2, 0, 1), -logdens_de, -logdens_de, -logdens, -logdens)

            self.p_hh = p_hh_new.transpose(1, 2, 0)

        if False:  # slow and does not work yet!
            scaled_logdens = -logdens / \
                float(len(logdens))*float(len(logdens_de))
            scaled_k_2h = list(range(len(self.fftlog_2h.k)))
            x, y, z = np.meshgrid(scaled_logdens, scaled_logdens, scaled_k_2h)

            p_hh_new = ndimage.map_coordinates(self.p_hh_base, np.array(
                [x, y, z]), order=3, mode="constant", cval=0.0, prefilter=True)

        self.p_hh_computed = True

    def _compute_xi_hm(self, redshift):
        logdens_de = self.xi_cross.logdens_list
        Mh_de = [self._convert_dens_to_mass(
            10**logdens_de[i], redshift, integration=self.config["hmf_int_algorithm"]) for i in range(len(logdens_de))]
        # logdens_de = np.log10([self._convert_mass_to_dens(self.Mh[i], redshift, integration = self.config["hmf_int_algorithm"]) for i in range(len(self.Mh))])#self.xi_cross.logdens_list
        #Mh_de = self.Mh
        logdens_p = np.log10([self._convert_mass_to_dens(
            1.01*Mh_de[i], redshift, integration=self.config["hmf_int_algorithm"]) for i in range(len(logdens_de))])
        logdens_m = np.log10([self._convert_mass_to_dens(
            0.99*Mh_de[i], redshift, integration=self.config["hmf_int_algorithm"]) for i in range(len(logdens_de))])

        pm_lin = self.get_pklin(self.fftlog_1h.k)
        g1_dm = self.g1.get_dm(self.fftlog_1h.k, redshift)

        logdens = list()
        for i in range(len(logdens_p)):
            logdens.append([logdens_m[i], logdens_p[i]])

        g = list()
        gp = self.g1.get(self.fftlog_2h.k, redshift, logdens_p)
        gm = self.g1.get(self.fftlog_2h.k, redshift, logdens_m)

        if self.do_linear_correction:
            bias_correction = np.zeros(len(Mh_de))
            for i in range(len(Mh_de)):
                bias_correction[i] = _compute_tinker10_bias(redshift, Mh_de[i], self.massfunc)/_compute_tinker10_bias(redshift, Mh_de[i], self.massfunc_cosmo_edge)

        for i in range(len(gp)):
            g.append([gm[i], gp[i]])

        xi_hm_dir_arr_all = self._get_xicross_direct_noint(
            np.concatenate(logdens), redshift)

        rswitch = min(60., 0.5 * self.cosmo.get_BAO_approx())
        xi_hm_de = list()
        for i in range(len(logdens_de)):
            logdensp = logdens[i][1]
            logdensm = logdens[i][0]
            g1p = g[i][1]
            g1m = g[i][0]

            denom = (10**logdensm - 10**logdensp)

            # calculate xicross_direct
            xi_hm_dir_arr = xi_hm_dir_arr_all[2*i: 2*(i+1)]
            xi_hm_dir_m = xi_hm_dir_arr[0]
            xi_hm_dir_p = xi_hm_dir_arr[1]
            xi_hm_dir_tmp = (xi_hm_dir_m * 10**logdensm -
                             xi_hm_dir_p * 10**logdensp) / denom
            xi_hm_dir_spl = ius(self.xi_cross.logrscale, xi_hm_dir_tmp, ext=3)
            xi_hm_dir = xi_hm_dir_spl(np.log(self.fftlog_1h.r))

            # calculate xicross_tree
            phm_tree_m = g1m * g1_dm * pm_lin
            phm_tree_p = g1p * g1_dm * pm_lin
            phm_tree = (phm_tree_m * 10**logdensm -
                        phm_tree_p * 10**logdensp) / denom
            #import matplotlib.pyplot as plt
            #plt.semilogx(self.k_1h, g1p)
            xi_hm_tree = self.fftlog_1h.pk2xi(phm_tree)
            #xi_hm_tree = fftLog.pk2xi_fftlog_array(
            #    self.k_1h, self.r_1h, phm_tree, self.kr, self.dlnk_1h)
            xi_hm_tmp = xi_hm_dir * np.exp(-(self.fftlog_1h.r/rswitch)**4) + xi_hm_tree * (1-np.exp(-(self.fftlog_1h.r/rswitch)**4))
            if self.do_linear_correction:
                xi_hm_tmp *= bias_correction[i]
            xi_hm_de.append(xi_hm_tmp)

        xi_hm_de = np.array(xi_hm_de)

        logdens = self.logdens
        self.xi_hm = rbs(-logdens_de, self.fftlog_1h.r, xi_hm_de)(-logdens, self.fftlog_1h.r)
        self.xi_hm_computed = True

    def _compute_p_hm(self, redshift):
        if self.xi_hm_computed == False:
            self._compute_xi_hm(redshift)

        logdens = self.logdens

        if self.do_linear_correction:
            pm_lin = self.get_pklin(self.fftlog_1h.k)

        p_hm = np.zeros((len(logdens), len(self.fftlog_1h.k)))
        for i in range(len(logdens)):
            p_hm[i] = self.fftlog_1h.xi2pk(self.xi_hm[i])
            #p_hm[i] = fftLog.xi2pk_fftlog_array(
            #    self.r_1h, self.k_1h, self.xi_hm[i], self.kr, self.dlnr_1h)
            if "baryon" in list(self.gparams.keys()):
                baryonic_params = self.gparams["baryon"]
                Mc = 10**baryonic_params[0]
                beta = baryonic_params[1]
                eta = baryonic_params[2]
                M200 = self.Mh[i]
                c200 = self._concentration(M200, redshift)
                #M200 = 1e14
                #c200 = 4
                k_1h_tmp = np.logspace(
                    np.log10(self.fftlog_1h.k[0]), np.log10(self.fftlog_1h.k[-1]), 300)
                wk = return_wk(k_1h_tmp, Mc, beta, eta, M200, c200, redshift)
                wk = (ius(k_1h_tmp, wk))(self.fftlog_1h.k)
                p_hm[i] *= wk
            if self.do_linear_correction:
                p_hm[i] *= (self.pm_lin_k_1h_out_of_range/pm_lin)

        self.p_hm = p_hm
        self.p_hm_computed = True

    def _compute_p_hm_satdist_emu(self, redshift):
        if self.xi_hm_computed == False:
            self._compute_xi_hm(redshift)

        logdens = self.logdens

        if self.do_linear_correction:
            pm_lin = self.get_pklin(self.fftlog_1h.k)

        p_hm_dist = list()
        for i, _R200 in enumerate(self.R200):
            sel = (self.fftlog_1h.r > _R200)
            xi_hm_dist = np.copy(self.xi_hm[i])
            xi_hm_dist[sel] = 0.
            p_hm_dist_tmp = self.fftlog_1h.xi2pk(xi_hm_dist)
            #p_hm_dist_tmp = fftLog.xi2pk_fftlog_array(
            #    self.r_1h, self.k_1h, xi_hm_dist, self.kr, self.dlnr_1h)
            if self.do_linear_correction:
                p_hm_dist_tmp *= (self.pm_lin_k_1h_out_of_range/pm_lin)

            # apodization
            if self.config["p_hm_apodization"] is not None:
                sigma = _R200/self.config["p_hm_apodization"]
                p_hm_dist_tmp *= np.exp(-0.5*sigma**2*self.fftlog_1h.k**2)

            norm_int_k = 10
            norm_int_r = np.logspace(self.config["fft_logrmin_2h"], np.log10(
                _R200), 2**norm_int_k+1)  # check if logmin_2h is okay!
            dlog_norm_int_r = np.log(norm_int_r[1])-np.log(norm_int_r[0])
            xi_hm_dist_spline = ius(self.fftlog_1h.r, xi_hm_dist)
            norm = self.do_integration(
                4.*np.pi*norm_int_r**3*self.rho_m*(1.+xi_hm_dist_spline(norm_int_r)), dx=dlog_norm_int_r)
            p_hm_dist.append(p_hm_dist_tmp * self.rho_m/norm)

        self.p_hm_dist_1h = p_hm_dist
        self.p_hm_dist_2h = np.array(
            [ius(self.fftlog_1h.k, p_hm_dist[i], ext=3)(self.fftlog_2h.k) for i in range(len(self.Mh))])

    def _concentration(self, M, z, model='diemer15'):
        if self.do_linear_correction:
            cparam = self.cparams_orig[0]
        else:
            cparam = self.cosmo.get_cosmology()[0]
        Om0 = 1. - cparam[2]
        h = np.sqrt((cparam[0] + cparam[1] + 0.00064)/Om0)
        H0 = h*100
        Ob0 = cparam[0]/h**2
        rho_m = (1. - cparam[2])*rho_cr
        M_8mpc = 4.*np.pi/3.*8.**3*rho_m
        sigma8 = self.massfunc.sM.get_sigma(M_8mpc)
        ns = cparam[4]
        params = {'flat': True, 'H0': H0, 'Om0': Om0,
                  'Ob0': Ob0, 'sigma8': sigma8, 'ns': ns}
        cosmo = colcosmology.setCosmology('myCosmo', params)
        c200m = concentration.concentration(M, '200m', z, model=model)
        return c200m

    def _compute_p_hm_satdist_NFW(self, redshift):
        R200 = (3*self._convert_dens_to_mass(10**np.array(self.xi_cross.logdens_list), redshift,
                                    integration=self.config["hmf_int_algorithm"])/(4.*np.pi*self.rho_m)/200.)**(1./3.)

        p_hm_dist = list()
        if "sat_dist_Rc" in list(self.gparams.keys()):
            Rc = self.gparams["sat_dist_Rc"]
        else:
            Rc = 1.0
        for i, (_M, _R200) in enumerate(zip(self.Mh, self.R200)):
            c200 = self._concentration(_M, redshift, self.config["c-M_relation"])  # , model = 'duffy08')
            if c200 == -1:
                #raise RuntimeError("concentration cannot be computed at (M,z) = (%e,%e)" % (_M, redshift))
                if self.do_linear_correction:
                    cparam = self.cparams_orig[0]
                else:
                    cparam = self.cosmo.get_cosmology()[0]
                logging.info("Colossus failed in computing concentration when cparams = %s, M=%s, and z=%s. Compute by diemer15 concentration implemented in Dark Emulator.", cparam, _M, redshift)
                c200 = self._get_concentration_Diemer15(_M, redshift)
            c200 = Rc*c200
            p_hm_dist.append(_get_uk(self.fftlog_1h.k, c200, _R200))

        self.p_hm_dist_1h = p_hm_dist
        self.p_hm_dist_2h = np.array(
            [ius(self.fftlog_1h.k, p_hm_dist[i], ext=3)(self.fftlog_2h.k) for i in range(len(self.Mh))])

    def _compute_p_hm_satdist(self, redshift):
        if self.p_hm_satdist_computed == False:
            if self.gparams["sat_dist_type"] == "emulator":
                self._compute_p_hm_satdist_emu(redshift)
            elif self.gparams["sat_dist_type"] == "NFW":
                self._compute_p_hm_satdist_NFW(redshift)
        self.p_hm_satdist_computed = True

    def _compute_HOD(self):
        Mmin = 10**self.gparams["logMmin"]
        sigma = np.sqrt(self.gparams["sigma_sq"])
        M1 = 10**self.gparams["logM1"]
        alpha = self.gparams["alpha"]
        kappa = self.gparams["kappa"]

        Ncen = 0.5*special.erfc(np.log10(Mmin/self.Mh)/sigma)
        lambda_sat = np.zeros(self.Mh.shape)
        sel = (self.Mh > kappa*Mmin)
        lambda_sat[sel] = ((self.Mh[sel]-kappa*Mmin)/M1)**alpha
        Nsat = Ncen*lambda_sat

        alpha_inc = self.gparams["alpha_inc"]
        logM_inc = self.gparams["logM_inc"]
        f_inc = np.maximum(0., np.minimum(
            1., 1.+alpha_inc*(np.log10(self.Mh) - logM_inc)))
        self.f_inc = f_inc

        self.Ncen = Ncen*f_inc
        self.Nsat = Nsat*f_inc
        self.lambda_sat = lambda_sat

        self.Ncen_mat = np.tile(self.Ncen, (len(self.fftlog_2h.k), 1)).transpose()

        self.HOD_computed = True

    def _compute_p_1hcs(self, redshift):
        if self.logdens_computed == False:
            self._compute_logdens(redshift)
        if self.dndM_computed == False:
            self._compute_dndM(redshift)
        if self.HOD_computed == False:
            self._compute_HOD()
        if self.ng_computed == False:
            self._compute_ng()
        if self.p_hm_satdist_computed == False:
            self._compute_p_hm_satdist(redshift)

        poff = self.gparams["poff"]
        Roff = self.gparams["Roff"]

        Hc_1h_over_Ncen = 1./self.ng * \
            (1. - poff + poff*np.exp(-0.5*self.k_1h_mat**2*(Roff*self.R200_mat)**2))
        Nsat_mat = np.tile(self.Nsat, (len(self.fftlog_1h.k), 1)).transpose()
        Hs_1h = Nsat_mat/self.ng*self.p_hm_dist_1h

        self.p_1hcs = self.do_integration(
            Hc_1h_over_Ncen*Hs_1h*self.dndM_mat*self.Mh_mat, axis=0, dx=self.dlogMh)

    def _compute_p_1hss(self, redshift):
        if self.logdens_computed == False:
            self._compute_logdens(redshift)
        if self.dndM_computed == False:
            self._compute_dndM(redshift)
        if self.HOD_computed == False:
            self._compute_HOD()
        if self.ng_computed == False:
            self._compute_ng()
        if self.p_hm_satdist_computed == False:
            self._compute_p_hm_satdist(redshift)

        Ncen_mat = np.tile(self.Ncen, (len(self.fftlog_1h.k), 1)).transpose()
        lambda_sat_mat = np.tile(
            self.lambda_sat, (len(self.fftlog_1h.k), 1)).transpose()
        lambda_1h_mat = lambda_sat_mat/self.ng*self.p_hm_dist_1h

        self.p_1hss = self.do_integration(
            lambda_1h_mat*lambda_1h_mat*Ncen_mat*self.dndM_mat*self.Mh_mat, axis=0, dx=self.dlogMh)

    def _compute_p_2hcc(self, redshift):
        if self.logdens_computed == False:
            self._compute_logdens(redshift)
        if self.dndM_computed == False:
            self._compute_dndM(redshift)
        if self.HOD_computed == False:
            self._compute_HOD()
        if self.ng_computed == False:
            self._compute_ng()
        if self.p_hh_computed == False:
            self._compute_p_hh(redshift)

        poff = self.gparams["poff"]
        Roff = self.gparams["Roff"]

        if False:
            Hc_2h = self.Ncen_mat/self.ng * \
                (1. - poff + poff * np.exp(-0.5 * self.k_2h_mat**2*(Roff*self.R200_mat)**2))
            Hc_2hHc_2h_mat = np.zeros(
                (len(self.Mh), len(self.Mh), len(self.fftlog_2h.k)))
            for i in range(Hc_2hHc_2h_mat.shape[0]):
                for j in range(Hc_2hHc_2h_mat.shape[1]):
                    Hc_2hHc_2h_mat[i, j] = Hc_2h[i]*Hc_2h[j]

            xx, yy = np.meshgrid(self.dndM, self.dndM)
            dndMdndM_mat = np.tile(xx*yy, (len(self.fftlog_2h.k), 1, 1)).transpose()

            xx, yy = np.meshgrid(self.Mh, self.Mh)
            MhMh_mat = np.tile(xx*yy, (len(self.fftlog_2h.k), 1, 1)).transpose()

            integrant = self.p_hh*Hc_2hHc_2h_mat*dndMdndM_mat*MhMh_mat
            self.p_2hcc = self.do_integration(self.do_integration(
                integrant, axis=0, dx=self.dlogMh), axis=0, dx=self.dlogMh)

        if True:
            Hc_2h = self.Ncen_mat/self.ng * \
                (1. - poff + poff * np.exp(-0.5 * self.k_2h_mat**2*(Roff*self.R200_mat)**2))

            p_2hcc_M2_int = list()
            for i, M in enumerate(self.Mh):
                p_2hcc_M2_int.append(self.do_integration(
                    self.p_hh[i] * Hc_2h * self.dndM_mat * self.Mh_mat, axis=0, dx=self.dlogMh))
            p_2hcc_M2_int = np.array(p_2hcc_M2_int)
            self.p_2hcc = self.do_integration(
                p_2hcc_M2_int * Hc_2h * self.dndM_mat * self.Mh_mat, axis=0, dx=self.dlogMh)

    def _compute_p_2hcs(self, redshift):
        if self.logdens_computed == False:
            self._compute_logdens(redshift)
        if self.dndM_computed == False:
            self._compute_dndM(redshift)
        if self.HOD_computed == False:
            self._compute_HOD()
        if self.ng_computed == False:
            self._compute_ng()
        if self.p_hh_computed == False:
            self._compute_p_hh(redshift)

        poff = self.gparams["poff"]
        Roff = self.gparams["Roff"]

        Hc_2h = self.Ncen_mat/self.ng * \
            (1. - poff + poff * np.exp(-0.5 * self.k_2h_mat**2*(Roff*self.R200_mat)**2))
        Hs_2h_mat = np.tile(self.Nsat, (len(self.fftlog_2h.k), 1)
                            ).transpose()/self.ng*self.p_hm_dist_2h

        p_2hcs_M2_int = list()
        for i, M in enumerate(self.Mh):
            p_2hcs_M2_int.append(self.do_integration(
                self.p_hh[i] * Hs_2h_mat * self.dndM_mat * self.Mh_mat, axis=0, dx=self.dlogMh))
        p_2hcs_M2_int = np.array(p_2hcs_M2_int)
        self.p_2hcs = self.do_integration(
            p_2hcs_M2_int * Hc_2h * self.dndM_mat * self.Mh_mat, axis=0, dx=self.dlogMh)

    def _compute_p_2hss(self, redshift):
        if self.logdens_computed == False:
            self._compute_logdens(redshift)
        if self.dndM_computed == False:
            self._compute_dndM(redshift)
        if self.HOD_computed == False:
            self._compute_HOD()
        if self.ng_computed == False:
            self._compute_ng()
        if self.p_hh_computed == False:
            self._compute_p_hh(redshift)
        if self.p_hm_satdist_computed == False:
            self._compute_p_hm_satdist(redshift)

        Hs_2h_mat = np.tile(self.Nsat, (len(self.fftlog_2h.k), 1)
                            ).transpose()/self.ng*self.p_hm_dist_2h
        p_2hss_M2_int = list()
        for i, M in enumerate(self.Mh):
            p_2hss_M2_int.append(self.do_integration(
                self.p_hh[i] * Hs_2h_mat * self.dndM_mat * self.Mh_mat, axis=0, dx=self.dlogMh))
        p_2hss_M2_int = np.array(p_2hss_M2_int)
        self.p_2hss = self.do_integration(
            p_2hss_M2_int * Hs_2h_mat * self.dndM_mat * self.Mh_mat, axis=0, dx=self.dlogMh)

    def _compute_p_cen(self, redshift):
        if self.logdens_computed == False:
            self._compute_logdens(redshift)
        if self.dndM_computed == False:
            self._compute_dndM(redshift)
        if self.HOD_computed == False:
            self._compute_HOD()
        if self.ng_computed == False:
            self._compute_ng()
        if self.p_hm_computed == False:
            self._compute_p_hm(redshift)

        poff = self.gparams["poff"]
        Roff = self.gparams["Roff"]

        Hc = self.Ncen_mat/self.ng*(1. - poff)
        self.p_cen = self.do_integration(
            self.p_hm*Hc*self.dndM_mat*self.Mh_mat, axis=0, dx=self.dlogMh)

    def _compute_p_cen_off(self, redshift):
        if self.logdens_computed == False:
            self._compute_logdens(redshift)
        if self.dndM_computed == False:
            self._compute_dndM(redshift)
        if self.HOD_computed == False:
            self._compute_HOD()
        if self.ng_computed == False:
            self._compute_ng()
        if self.p_hm_computed == False:
            self._compute_p_hm(redshift)

        poff = self.gparams["poff"]
        Roff = self.gparams["Roff"]

        Hc_off = self.Ncen_mat/self.ng*poff * \
            np.exp(-0.5*self.k_1h_mat**2*(Roff*self.R200_mat)**2)
        self.p_cen_off = self.do_integration(
            self.p_hm*Hc_off*self.dndM_mat*self.Mh_mat, axis=0, dx=self.dlogMh)

    def _compute_p_sat(self, redshift):
        if self.logdens_computed == False:
            self._compute_logdens(redshift)
        if self.dndM_computed == False:
            self._compute_dndM(redshift)
        if self.HOD_computed == False:
            self._compute_HOD()
        if self.ng_computed == False:
            self._compute_ng()
        if self.p_hm_computed == False:
            self._compute_p_hm(redshift)
        if self.p_hm_satdist_computed == False:
            self._compute_p_hm_satdist(redshift)

        Nsat_mat = np.tile(self.Nsat, (len(self.fftlog_1h.k), 1)).transpose()
        Hs = Nsat_mat/self.ng*self.p_hm_dist_1h
        self.p_sat = self.do_integration(
            self.p_hm*Hs*self.dndM_mat*self.Mh_mat, axis=0, dx=self.dlogMh)

    def _compute_effective_bias(self, redshift):
        if self.logdens_computed == False:
            self._compute_logdens(redshift)
        if self.dndM_computed == False:
            self._compute_dndM(redshift)
        if self.HOD_computed == False:
            self._compute_HOD()
        if self.ng_computed == False:
            self._compute_ng()
        if self.p_hh_computed == False:
            self._compute_p_hh(redshift)


        # original
        #logdens_p = self.logdens_p
        #logdens_m = self.logdens_m
        #g = self.g

        # The following block can be calculated in a function and then the result can be stored. This can be recycled in lensing calculation
        logdens_de = self.xi_cross.logdens_list
        Mh_de = [self._convert_dens_to_mass(
            10**logdens_de[i], redshift, integration=self.config["hmf_int_algorithm"]) for i in range(len(logdens_de))]
        logdens_p = np.log10([self._convert_mass_to_dens(
            1.01*Mh_de[i], redshift, integration=self.config["hmf_int_algorithm"]) for i in range(len(logdens_de))])
        logdens_m = np.log10([self._convert_mass_to_dens(
            0.99*Mh_de[i], redshift, integration=self.config["hmf_int_algorithm"]) for i in range(len(logdens_de))])
        g = list()
        gp = self.g1.get(self.fftlog_2h.k, redshift, logdens_p)
        gm = self.g1.get(self.fftlog_2h.k, redshift, logdens_m)
        for i in range(len(gp)):
            g.append([gm[i], gp[i]])

        g1_dm = np.average(self.g1.get_dm(
            self.fftlog_2h.k, redshift)[self.fftlog_2h.k < 0.05])
        bias = list()
        for i in range(len(logdens_de)):
            dens_p, dens_m = 10**logdens_p[i], 10**logdens_m[i]
            denom = dens_m - dens_p
            g1p = np.average(g[i][1][self.fftlog_2h.k < 0.05])
            g1m = np.average(g[i][0][self.fftlog_2h.k < 0.05])
            bias.append((g1m * dens_m - g1p * dens_p) / denom / g1_dm)

        if self.do_linear_correction:
            for i in range(len(Mh_de)):
                bias[i] *= _compute_tinker10_bias(redshift, Mh_de[i], self.massfunc)/_compute_tinker10_bias(redshift, Mh_de[i], self.massfunc_cosmo_edge)

        bias = ius(-logdens_de, bias)(-self.logdens)

        bias_eff = self.do_integration(
            self.dndM * (self.Ncen + self.Nsat) * self.Mh * bias, dx=self.dlogMh)/self.ng
        return bias_eff

    def _check_update_redshift(self, redshift):
        if self.redshift != redshift:
            self.redshift = redshift
            self._initialize_cosmology_computation_flags()

    def _get_effective_bias(self, redshift):
        return self._compute_effective_bias(redshift)

    def _get_wp_rsd(self, rp, redshift, pimax):
        # calculate xi
        r_ref = np.logspace(-3, 3, 512)
        xi = self.xi_gg(r_ref, redshift)
        xi_spl = ius(r_ref, xi)

        # calculate beta
        f = self.f_from_z(redshift)
        #b = 2.118
        b = self._get_effective_bias(redshift)
        beta = f/b

        n = 3
        J_n = list()
        for _r in r_ref:
            t = np.linspace(1e-10, _r, 1024)
            dt = t[1]-t[0]
            J_n.append(1./_r**n*integrate.trapz(t**(n-1.)*xi_spl(t), dx=dt))
        J_3 = np.array(J_n)

        n = 5
        J_n = list()
        for _r in r_ref:
            t = np.linspace(1e-10, _r, 1024)
            dt = t[1]-t[0]
            J_n.append(1./_r**n*integrate.trapz(t**(n-1.)*xi_spl(t), dx=dt))
        J_5 = np.array(J_n)

        xi_0 = (1.+2./3.*beta+1./5.*beta**2)*xi
        xi_2 = (4./3.*beta+4./7.*beta**2)*(xi-3.*J_3)
        xi_4 = 8./35.*beta**2*(xi+15./2.*J_3-35./2.*J_5)

        r_pi = np.logspace(-3, np.log10(pimax), 512)
        rp, r_pi = np.meshgrid(rp, r_pi, indexing='ij')

        s = np.sqrt(rp**2+r_pi**2)

        mu = r_pi/s

        l0 = special.eval_legendre(0, mu)
        l2 = special.eval_legendre(2, mu)
        l4 = special.eval_legendre(4, mu)

        xi_s = ius(r_ref, xi_0)(s)*l0 + ius(r_ref, xi_2)(s) * \
            l2 + ius(r_ref, xi_4)(s)*l4

        xi_s_spl = rbs(rp[:, 0], r_pi[0], xi_s)

        wp = list()
        for _r in rp:
            wp.append(2*integrate.quad(lambda t: xi_s_spl(_r, t)
                                       [0][0], 0, pimax, epsabs=1e-4)[0])
        wp = np.array(wp)
        return wp

    def get_wp(self, rp, redshift, pimax=None, rsd=False):
        """get_wp

        Compute projected galaxy auto-correlation function :math:`w_\mathrm{p}(r_\mathrm{p})`.

        Args:
            r_p (numpy array): 2 dimensional separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the galaxies are located
            pi_max (float): The range of line of sight integral :math:`\pi_{\mathrm{max}}` in :math:`h^{-1}\mathrm{Mpc}`. If None, the projection is performed using the zeroth order Bessel function, i.e., :math:`\pi_{\mathrm{max}}=\infty` (default=None).
            rsd (bool): if True, redshift space distortion is incorporated into the model (default=False).

        Returns:
            numpy array: projected galaxy auto-correlation function in :math:`h^{-1}\mathrm{Mpc}`
        """

        # Projected correlation functions should be the same if or not there is redshift space distortions when integrated to infinity.
        if (pimax == None and rsd == False) or (pimax == None and rsd == True):
            self._check_update_redshift(redshift)
            self._compute_p_1hcs(redshift)
            self._compute_p_1hss(redshift)
            self._compute_p_2hcc(redshift)
            self._compute_p_2hcs(redshift)
            self._compute_p_2hss(redshift)

            p_tot_1h = 2.*self.p_1hcs + self.p_1hss
            p_tot_2h = self.p_2hcc + 2.*self.p_2hcs + self.p_2hss
            wp = ius( self.fftlog_1h.r, self.fftlog_1h.pk2wp(p_tot_1h) )(rp) + ius( self.fftlog_2h.r, self.fftlog_2h.pk2wp(p_tot_2h) )(rp)
            #wp = ius(self.fftlog_1h.r, fftLog.pk2xiproj_J0_fftlog_array(self.fftlog_1h.k, self.fftlog_1h.r, p_tot_1h, self.fftlog_1h.kr, self.fftlog_2h.dlnk))(
            #    rp)+ius(self.fftlog_2h.r, fftLog.pk2xiproj_J0_fftlog_array(self.k_2h, self.r_2h, p_tot_2h, self.kr, self.dlnk_2h))(rp)
        else:
            if not isinstance(pimax, float):
                raise RuntimeError("pi_max should be None or float")
            if rsd == True:
                wp = self._get_wp_rsd(rp, redshift, pimax)
            else:
                r_ref = np.logspace(np.min([self.config["fft_logrmin_1h"], self.config["fft_logrmin_2h"]]), np.max(
                    [self.config["fft_logrmax_1h"], self.config["fft_logrmax_2h"]]), 1024)
                xi_gg_spl = ius(r_ref, self.xi_gg(r_ref, redshift))
                t = np.linspace(0, pimax, 1024)
                dt = t[1]-t[0]
                wp = list()
                for rpnow in rp:
                    wp.append(
                        2*integrate.trapz(xi_gg_spl(np.sqrt(t**2+rpnow**2)), dx=dt))
                wp = np.array(wp)
        return wp

    def get_wp_1hcs(self, rp, redshift):
        """get_wp_1hcs

        Compute projected 1-halo correlation function between central and satellite galaxies :math:`w_\mathrm{p, cen-sat}^\mathrm{1h}(r_\mathrm{p})`. Note that the line-of-sight integration is performed using the zeroth order Bessel function, i.e., , :math:`\pi_{\mathrm{max}}=\infty`.

        Args:
            r_p (numpy array): 2 dimensional separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the galaxies are located

        Returns:
            numpy array: projected 1-halo correlation function between central and satellite galaxies in :math:`h^{-1}\mathrm{Mpc}`
        """

        self._check_update_redshift(redshift)

        self._compute_p_1hcs(redshift)
        wp1hcs = ius( self.fftlog_1h.r, self.fftlog_1h.pk2wp(self.p_1hcs) )(rp)
        return wp1hcs

    def get_wp_1hss(self, rp, redshift):
        """get_wp_1hss

        Compute projected 1-halo correlation function between satellite galaxies :math:`w_\mathrm{p, sat-sat}^\mathrm{1h}(r_\mathrm{p})`. Note that the line-of-sight integration is performed using the zeroth order Bessel function, i.e., , :math:`\pi_{\mathrm{max}}=\infty`.

        Args:
            r_p (numpy array): 2 dimensional separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the galaxies are located

        Returns:
            numpy array: projected 1-halo correlation function between satellite galaxies in :math:`h^{-1}\mathrm{Mpc}`
        """

        self._check_update_redshift(redshift)

        self._compute_p_1hss(redshift)
        wp1hss = ius( self.fftlog_1h.r, self.fftlog_1h.pk2wp(self.p_1hss) )(rp)
        return wp1hss

    def get_wp_2hcc(self, rp, redshift):
        """get_wp_2hcc

        Compute projected 2-halo correlation function between central galaxies :math:`w_\mathrm{p, cen-cen}^\mathrm{2h}(r_\mathrm{p})`. Note that the line-of-sight integration is performed using the zeroth order Bessel function, i.e., , :math:`\pi_{\mathrm{max}}=\infty`.

        Args:
            r_p (numpy array): 2 dimensional separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the galaxies are located

        Returns:
            numpy array: projected 2-halo correlation function between central galaxies in :math:`h^{-1}\mathrm{Mpc}`
        """

        self._check_update_redshift(redshift)

        self._compute_p_2hcc(redshift)
        wp2hcc = ius( self.fftlog_2h.r, self.fftlog_2h.pk2wp(self.p_2hcc) )(rp)
        return wp2hcc

    def get_wp_2hcs(self, rp, redshift):
        """get_wp_2hcs

        Compute projected 2-halo correlation function between central and satellite galaxies :math:`w_\mathrm{p, cen-sat}^\mathrm{2h}(r_\mathrm{p})`. Note that the line-of-sight integration is performed using the zeroth order Bessel function, i.e., , :math:`\pi_{\mathrm{max}}=\infty`.

        Args:
            r_p (numpy array): 2 dimensional separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the galaxies are located

        Returns:
            numpy array: projected 2-halo correlation function between central and satellite galaxies in :math:`h^{-1}\mathrm{Mpc}`
        """

        self._check_update_redshift(redshift)

        self._compute_p_2hcs(redshift)
        wp2hcs = ius( self.fftlog_2h.r, self.fftlog_2h.pk2wp(self.p_2hcs) )(rp)
        return wp2hcs

    def get_wp_2hss(self, rp, redshift):
        """get_wp_2hss

        Compute projected 2-halo correlation function between satellite galaxies :math:`w_\mathrm{p, sat-sat}^\mathrm{2h}(r_\mathrm{p})`. Note that the line-of-sight integration is performed using the zeroth order Bessel function, i.e., , :math:`\pi_{\mathrm{max}}=\infty`.

        Args:
            r_p (numpy array): 2 dimensional separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the galaxies are located

        Returns:
            numpy array: projected 2-halo correlation function between satellite galaxies in :math:`h^{-1}\mathrm{Mpc}`
        """

        self._check_update_redshift(redshift)

        self._compute_p_2hss(redshift)
        wp2hss = ius( self.fftlog_2h.r, self.fftlog_2h.pk2wp(self.p_2hss) )(rp)
        return wp2hss

    def get_xi_gg(self, r, redshift):
        """get_xi_gg

        Compute galaxy auto-correlation function :math:`\\xi_\mathrm{gg}(r)`.

        Args:
            r (numpy array): 3 dimensional separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the galaxies are located

        Returns:
            numpy array: galaxy auto-correlation function
        """

        self._check_update_redshift(redshift)

        self._compute_p_1hcs(redshift)
        self._compute_p_1hss(redshift)
        self._compute_p_2hcc(redshift)
        self._compute_p_2hcs(redshift)
        self._compute_p_2hss(redshift)

        p_tot_1h = 2.*self.p_1hcs + self.p_1hss
        p_tot_2h = self.p_2hcc + 2.*self.p_2hcs + self.p_2hss
        xi_gg = ius( self.fftlog_1h.r, self.fftlog_1h.pk2xi(p_tot_1h) )(r) + ius( self.fftlog_2h.r, self.fftlog_2h.pk2xi(p_tot_2h) )(r)
        return xi_gg

    def get_xi_gg_1hcs(self, r, redshift):
        """get_xi_gg_1hcs

        Compute 1-halo correlation function between central and satellite galaxies :math:`\\xi_\mathrm{cen-sat}^\mathrm{1h}(r)`.

        Args:
            r (numpy array): 3 dimensional separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the galaxies are located

        Returns:
            numpy array: 1-halo correlation function between central and satellite galaxies
        """

        self._check_update_redshift(redshift)

        self._compute_p_1hcs(redshift)
        xi_gg_1hcs = ius(self.fftlog_1h.r, self.fftlog_1h.pk2xi(self.p_1hcs) )(r)
        return xi_gg_1hcs

    def get_xi_gg_1hss(self, r, redshift):
        """get_xi_gg_1hss

        Compute 1-halo correlation function between satellite galaxies :math:`\\xi_\mathrm{sat-sat}^\mathrm{1h}(r)`.

        Args:
            r (numpy array): 3 dimensional separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the galaxies are located

        Returns:
            numpy array: 1-halo correlation function between satellite galaxies
        """

        self._check_update_redshift(redshift)

        self._compute_p_1hss(redshift)
        xi_gg_1hss = ius(self.fftlog_1h.r, self.fftlog_1h.pk2xi(self.p_1hss) )(r)
        return xi_gg_1hss

    def get_xi_gg_2hcc(self, rp, redshift):
        """get_xi_gg_2hcc

        Compute 2-halo correlation function between central galaxies :math:`\\xi_\mathrm{cen-cen}^\mathrm{2h}(r)`.

        Args:
            r (numpy array): 3 dimensional separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the galaxies are located

        Returns:
            numpy array: 2-halo correlation function between central galaxies
        """

        self._check_update_redshift(redshift)

        self._compute_p_2hcc(redshift)
        xi_gg_2hcc = ius( self.fftlog_2h.r, self.fftlog_2h.pk2xi(self.p_2hcc) )(rp)
        #xi_gg_2hcc = ius(self.fftlog_2h.r, fftLog.pk2xi_fftlog_array(
        #    self.k_2h, self.r_2h, self.p_2hcc, self.kr, self.dlnk_2h))(rp)
        return xi_gg_2hcc

    def get_xi_gg_2hcs(self, rp, redshift):
        """get_xi_gg_2hcs

        Compute 2-halo correlation function between central and satellite galaxies :math:`\\xi_\mathrm{cen-sat}^\mathrm{2h}(r)`.

        Args:
            r (numpy array): 3 dimensional separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the galaxies are located

        Returns:
            numpy array: 2-halo correlation function between central and satellite galaxies
        """

        self._check_update_redshift(redshift)

        self._compute_p_2hcs(redshift)
        xi_gg_2hcs = ius(self.fftlog_2h.r, self.fftlog_2h.pk2xi(self.p_2hcs) )(rp)
        #xi_gg_2hcs = ius(self.fftlog_2h.r, fftLog.pk2xi_fftlog_array(
        #    self.k_2h, self.r_2h, self.p_2hcs, self.kr, self.dlnk_2h))(rp)
        return xi_gg_2hcs

    def get_xi_gg_2hss(self, rp, redshift):
        """get_xi_gg_2hss

        Compute 2-halo correlation function between satellite galaxies :math:`\\xi_\mathrm{sat-sat}^\mathrm{2h}(r)`.

        Args:
            r (numpy array): 3 dimensional separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the galaxies are located

        Returns:
            numpy array: 2-halo correlation function between satellite galaxies
        """

        self._check_update_redshift(redshift)

        self._compute_p_2hss(redshift)
        xi_gg_2hss = ius( self.fftlog_2h.r, self.fftlog_2h.pk2xi(self.p_2hss) )(rp)
        #xi_gg_2hss = ius(self.fftlog_2h.r, fftLog.pk2xi_fftlog_array(
        #    self.k_2h, self.r_2h, self.p_2hss, self.kr, self.dlnk_2h))(rp)
        return xi_gg_2hss

    def get_ds(self, rp, redshift):
        """get_ds

        Compute weak lensing signal :math:`\Delta\Sigma(r_\mathrm{p})`.

        Args:
            rp (numpy array): 2 dimensional projected separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the lens galaxies are located

        Returns:
            numpy array: excess surface density in :math:`h M_\odot \mathrm{pc}^{-2}`
        """
        self._check_update_redshift(redshift)

        self._compute_p_cen(redshift)
        self._compute_p_cen_off(redshift)
        self._compute_p_sat(redshift)

        p_tot = self.p_cen + self.p_cen_off + self.p_sat
        ds = self.rho_m/10**12 * ius( self.fftlog_1h.r, self.fftlog_1h.pk2dwp(p_tot) )(rp)
        #ds = self.rho_m/10**12*ius(self.fftlog_1h.r, fftLog.pk2xiproj_J2_fftlog_array(
        #    self.k_1h, self.r_1h, p_tot, self.kr, self.dlnk_1h))(rp)
        return ds

    def get_ds_cen(self, rp, redshift):
        """get_ds_cen

        Compute weak lensing signal of (centered) central galaxies :math:`\Delta\Sigma_\mathrm{cen}(r_\mathrm{p})`.

        Args:
            rp (numpy array): 2 dimensional projected separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the lens galaxies are located

        Returns:
            numpy array: excess surface density of (centered) central galaxies in :math:`h M_\odot \mathrm{pc}^{-2}`
        """

        self._check_update_redshift(redshift)

        self._compute_p_cen(redshift)
        return self.rho_m/10**12 * ius(self.fftlog_1h.r, self.fftlog_1h.pk2dwp(self.p_cen) )(rp)
        #return self.rho_m/10**12*ius(self.fftlog_1h.r, fftLog.pk2xiproj_J2_fftlog_array(self.k_1h, self.r_1h, self.p_cen, self.kr, self.dlnk_1h))(rp)

    def get_ds_cen_off(self, rp, redshift):
        """get_ds_cen_off

        Compute weak lensing signal of off-centered central galaxies :math:`\Delta\Sigma_\mathrm{off-cen}(r_\mathrm{p})`.

        Args:
            rp (numpy array): 2 dimensional projected separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the lens galaxies are located

        Returns:
            numpy array: excess surface density of off-centered central galaxies in :math:`h M_\odot \mathrm{pc}^{-2}`
        """

        self._check_update_redshift(redshift)

        self._compute_p_cen_off(redshift)
        return self.rho_m/10**12 * ius(self.fftlog_1h.r, self.fftlog_1h.pk2dwp(self.p_cen_off) )(rp)
        #return self.rho_m/10**12*ius(self.fftlog_1h.r, fftLog.pk2xiproj_J2_fftlog_array(self.k_1h, self.r_1h, self.p_cen_off, self.kr, self.dlnk_1h))(rp)

    def get_ds_sat(self, rp, redshift):
        """get_ds_sat

        Compute weak lensing signal of satellite galaxies :math:`\Delta\Sigma_\mathrm{sat}(r_\mathrm{p})`.

        Args:
            rp (numpy array): 2 dimensional projected separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the lens galaxies are located

        Returns:
            numpy array: excess surface density of satellite galaxies in :math:`h M_\odot \mathrm{pc}^{-2}`
        """

        self._check_update_redshift(redshift)

        self._compute_p_sat(redshift)
        return self.rho_m/10**12 * ius(self.fftlog_1h.r, self.fftlog_1h.pk2dwp(self.p_sat) )(rp)
        #return self.rho_m/10**12*ius(self.fftlog_1h.r, fftLog.pk2xiproj_J2_fftlog_array(self.k_1h, self.r_1h, self.p_sat, self.kr, self.dlnk_1h))(rp)

    def _get_wp_gm(self, rp, redshift):
        self._check_update_redshift(redshift)

        self._compute_p_cen(redshift)
        self._compute_p_cen_off(redshift)
        self._compute_p_sat(redshift)

        p_tot = self.p_cen + self.p_cen_off + self.p_sat
        wp = ius( self.fftlog_1h.r, self.fftlog_1h.pk2wp(p_tot) )(rp)
        return wp

    def _get_sigma_gm(self, rp, redshift):
        wp = self.wp_gm(rp, redshift)
        return self.rho_m/10**12*wp

    def get_xi_gm(self, r, redshift):
        """get_xi_gm

        Compute correlation function between galaxies and dark matter :math:`\\xi_\mathrm{gm}(r)`.

        Args:
            r (numpy array): 3 dimensional separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the galaxies are located

        Returns:
            numpy array: correlation function between galaxies and dark matter
        """

        self._check_update_redshift(redshift)

        self._compute_p_cen(redshift)
        self._compute_p_cen_off(redshift)
        self._compute_p_sat(redshift)

        p_tot = self.p_cen + self.p_cen_off + self.p_sat
        xi_gm = ius( self.fftlog_1h.r, self.fftlog_1h.pk2xi(p_tot) )(r)
        return xi_gm

    def get_xi_gm_cen(self, r, redshift):
        """get_xi_gm_cen

        Compute correlation function between (centered) central galaxies and dark matter :math:`\\xi_\mathrm{gm, cen}(r)`.

        Args:
            r (numpy array): 3 dimensional separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the galaxies are located

        Returns:
            numpy array: correlation function between (centered) central galaxies and dark matter
        """

        self._check_update_redshift(redshift)

        self._compute_p_cen(redshift)
        return ius( self.fftlog_1h.r, self.fftlog_1h.pk2xi(self.p_cen) )(r)

    def get_xi_gm_cen_off(self, r, redshift):
        """get_xi_gm_cen_off

        Compute correlation function between off-centered central galaxies and dark matter :math:`\\xi_\mathrm{gm, off-cen}(r)`.

        Args:
            r (numpy array): 3 dimensional separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the galaxies are located

        Returns:
            numpy array: correlation function between off-centered central galaxies and dark matter
        """

        self._check_update_redshift(redshift)

        self._compute_p_cen_off(redshift)
        return ius( self.fftlog_1h.r, self.fftlog_1h.pk2xi(self.p_cen_off) )(r)

    def get_xi_gm_sat(self, r, redshift):
        """get_xi_gm_sat

        Compute correlation function between satellite galaxies and dark matter :math:`\\xi_\mathrm{gm, sat}(r)`.

        Args:
            r (numpy array): 3 dimensional separation in :math:`h^{-1}\mathrm{Mpc}`
            redshift (float): redshift at which the galaxies are located

        Returns:
            numpy array: correlation function between satellite galaxies and dark matter
        """

        self._check_update_redshift(redshift)
        self._compute_p_sat(redshift)
        return ius( self.fftlog_1h.r, self.fftlog_1h.pk2xi(self.p_sat) )(r)

    def _get_wp_mm(self, rp, redshift):
        xi = self.get_xinl(self.fftlog_2h.r, redshift)
        pk = self.fftlog_2h.xi2pk(xi)
        wp = self.fftlog_2h.pk2wp(pk)
        #pk = fftLog.xi2pk_fftlog_array(
        #    self.r_2h, self.k_2h, xi, self.kr, self.dlnr_2h)
        #wp = ius(self.fftlog_2h.r, fftLog.pk2xiproj_J0_fftlog_array(
        #    self.k_2h, self.r_2h, pk, self.kr, self.dlnk_2h))(rp)
        return wp

    def get_ng(self, redshift):
        """get_ng

        Compute galaxy abundance :math:`n_g`.

        Args:
            redshift (float): redshift at which the galaxies are located

        Returns:
            float: galaxy abundance in :math:`h^3\mathrm{Mpc}^{-3}`
        """

        if self.dndM_computed == False:
            self._compute_dndM(redshift)
        if self.dndM_spl_computed == False:
            self._compute_dndM_spl(redshift)
        if self.HOD_computed == False:
            self._compute_HOD()
        if self.ng_computed == False:
            self._compute_ng()
        return self.ng

    # methods for concentration
    def _get_M_for_delta_c(self, delta_c = 1.686):
        M = self.massfunc.sM.Mlist
        sigmaM = self.massfunc.sM.pred_table
        M_of_sigmaM = ius(sigmaM[::-1], M[::-1])
        M_for_delta_c = M_of_sigmaM(delta_c)
        return M_for_delta_c

    def _get_z_for_delta_c(self, M, z_of_D0, delta_c = 1.686):
        sigmaM = self.massfunc.sM.get_sigma(M)
        z_c = z_of_D0(delta_c/sigmaM)
        return z_c

    def _get_fNFW(self, c):
        """
        f(c) in Takada M., Jain B., 2003, MNRAS, 344, 857
        """
        f = 1./(np.log(1.+c)-c/(1.+c))
        return f

    def _convert_concentration(self, c, delta, m_or_c, target_delta, target_m_or_c, z):
        if m_or_c == "m":
            rho = self.cosmo.get_Omega0()*(1.+z)**3*rho_cr
        elif m_or_c == "c":
            rho = self.cosmo.get_Ez(z)**2.*rho_cr

        if target_m_or_c == "m":
            target_rho = self.cosmo.get_Omega0()*(1.+z)**3*rho_cr
        elif target_m_or_c == "c":
            target_rho = self.cosmo.get_Ez(z)**2.*rho_cr

        def F(target_c):
            return target_delta*target_c**3*self._get_fNFW(target_c)*target_rho - delta*c**3*self._get_fNFW(c)*rho
        #print("c", c)
        target_c = optimize.newton(F, c)
        return target_c

    def _get_concentration_Bullock01(self, M_200m, z):
        """
        This is the implementation of Eq. (18) at Bullock et al. (2001). Note that this method is not tested yet.
        """
        c0 = 9.0
        beta = 0.13
        x = self.cosmo.get_Omega0()*(1.+z)**3 - 1.0
        # compute delta_vir following Bryan & Norman 1998 <http://adsabs.harvard.edu/abs/1998ApJ...495...80B>
        delta_vir = 18 * np.pi**2 + 82.0 * x - 39.0 * x**2

        M_star_200m = self._get_M_for_delta_c()
        c_star_200m = self._convert_concentration(c0, delta_vir, "m", 200., "m", z)
        M_star_vir = self._get_fNFW(c_star_200m)/self._get_fNFW(c0)*M_star_200m

        def F(c_200m, M_200m, M_star_vir, c0, beta, delta_vir, z):
            c_vir = self._convert_concentration(c_200m, 200., "m", delta_vir, "m", z)
            M_vir = self._get_fNFW(c_200m)/self._get_fNFW(c_vir)*M_200m
            c_vir_bullock01 = c0/(1.+z)*(M_vir/M_star_vir)**-beta
            return c_vir - c_vir_bullock01
        c_200m = optimize.newton(F, c0, args = (M_200m, M_star_vir, c0, beta, delta_vir, z))
        return c_200m

    def _get_concentration_Maccio08(self, M_200m, z):
        print("halo mass", M_200m/10**14.)
        K = 3.85
        F = 0.01

        l_z = np.linspace(0., 100., 1000)
        D0 = np.array([self.Dgrowth_from_z(_z) for _z in l_z])
        z_of_D0= ius(D0[::-1], l_z[::-1])
        def get_c_200m(c_200m):
            c_200c = self._convert_concentration(c_200m, 200., "m", 200., "c", z)
            M_200c = self._get_fNFW(c_200m)/self._get_fNFW(c_200c)*M_200m
            M_star_200c = F*M_200c
            c_star_200c = K
            z_c_ini = self._get_z_for_delta_c(M_star_200c, z_of_D0)
            def get_z_c(z_c):
                c_star_200m = self._convert_concentration(c_star_200c, 200., "c", 200., "m", z_c)
                M_star_200m = self._get_fNFW(c_star_200m)/self._get_fNFW(c_star_200c)*M_star_200c
                z_c_target = self._get_z_for_delta_c(M_star_200m, z_of_D0)
                return z_c - z_c_target
            z_c, r = optimize.newton(get_z_c, z_c_ini, full_output=True)
            #z_c, r = optimize.brentq(get_z_c, 0., 100., full_output=True)
            c_200c_Maccio08 = K*(self.cosmo.get_Ez(z_c)/self.cosmo.get_Ez(z))**(2./3.)
            c_200m_Maccio08 = self._convert_concentration(c_200c_Maccio08, 200., "c", 200., "m", z)
            return c_200m - c_200m_Maccio08
        c_200m = optimize.newton(get_c_200m, 10.)
        return c_200m

    def _get_n_Diemer15(self, k_R):
        # n = dlog(P)/dlog(k)
        k_min = np.min(k_R)*0.9
        k_max = np.max(k_R)*1.1
        logk = np.arange(np.log10(k_min), np.log10(k_max), 0.01)
        Pk = self.get_pklin(10**logk)
        interp = ius(logk, np.log10(Pk))
        n = interp(np.log10(k_R), nu = 1) # nu = 1 means getting a first derivative
        return n

    def _get_n_from_M_Diemer15(self, M_200c):
        kappa = 1.0
        rho_m0 = self.cosmo.get_Omega0()*rho_cr
        # compute R from M_200c, not M_200m. This is what is done in colossus, but is this okay?
        R = (3.*M_200c/4./np.pi/rho_m0)**(1./3.)
        k_R = 2.*np.pi/R*kappa
        n = self._get_n_Diemer15(k_R)
        return n

    def _get_concentration_Diemer15(self, M_200m, z, statistic = 'median'):

        def _get_c_200m(c_200m):
            # convert (M200m, c200m) to (M200c, c200c)
            c_200c = self._convert_concentration(c_200m, 200., "m", 200., "c", z)
            M_200c = self._get_fNFW(c_200m)/self._get_fNFW(c_200c)*M_200m

            # get a slope of power spectrum
            n = self._get_n_from_M_Diemer15(M_200c)

            # colossus computes nu from R computed with M200c and rho_m, i.e., R = (3.*M_200c/4./np.pi/rho_m0)**(1./3.). It's not really a right thing to do, but I'm following this...
            sigmaM = self.massfunc.sM.get_sigma(M_200c)
            D0 = self.Dgrowth_from_z(z)
            sigmaM *= D0
            nu200c = 1.686/sigmaM

            median_phi_0 = 6.58
            median_phi_1 = 1.27
            median_eta_0 = 7.28
            median_eta_1 = 1.56
            median_alpha = 1.08
            median_beta  = 1.77

            mean_phi_0 = 6.66
            mean_phi_1 = 1.37
            mean_eta_0 = 5.41
            mean_eta_1 = 1.06
            mean_alpha = 1.22
            mean_beta  = 1.22

            if statistic == 'median':
                floor = median_phi_0 + n * median_phi_1
                nu0 = median_eta_0 + n * median_eta_1
                alpha = median_alpha
                beta = median_beta
            elif statistic == 'mean':
                floor = mean_phi_0 + n * mean_phi_1
                nu0 = mean_eta_0 + n * mean_eta_1
                alpha = mean_alpha
                beta = mean_beta
            else:
                raise Exception("Unknown statistic.")

            c_200c_Diemer15 = 0.5 * floor * ((nu0 / nu200c)**alpha + (nu200c / nu0)**beta)
            c_200m_Diemer15 = self._convert_concentration(c_200c_Diemer15, 200., "c", 200., "m", z)
            return c_200m - c_200m_Diemer15

        c_200m = optimize.newton(_get_c_200m, 10.)
        return c_200m


def _linearGrowth(Ode,wde,z):
    Om = 1 - Ode
    a_scale = 1./(1.+z)
    alpha = -1./(3.*wde)
    beta = (wde-1.)/(2.*wde)
    gamma = 1.-5./(6.*wde)
    x = -Ode/Om * a_scale**(-3.*wde)
    res = integrate.quad(lambda t: t**(beta-1.)*(1.-t)**(gamma-beta-1.)*(1.-t*x)**(-alpha), 0, 1.)
    return a_scale * res[0]

def _compute_tinker10_bias(redshift, Mh, massfunc):
    delta = 200.
    delta_c = 1.686
    y = np.log10(delta)
    A = 1.0+0.24*y*np.exp(-(4./y)**4)
    a = 0.44*y-0.88
    B = 0.183
    b = 1.5
    C = 0.019+0.107*y+0.19*np.exp(-(4./y)**4)
    c = 2.4

    params = massfunc.cosmo_now.get_cosmology()
    Ode = params[0,2]
    wde = params[0,5]
    growth = _linearGrowth(Ode,wde,redshift)/_linearGrowth(Ode,wde,0.)
    sigM = growth*massfunc.sM.get_sigma(Mh)
    nu = delta_c/sigM
    b = 1.-A*nu**a/(nu**a+delta_c**a) + B*nu**b + C*nu**c
    return b
