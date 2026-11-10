"""Regression tests for issue #27: NaN in wp for some cosmology/HOD combinations.

The parameter sets below gave NaN for wp (all rp) in v1.1.2 and v1.2.0. They
were found in a random scan within the supported cosmology range, using the
ln(10^10 As) value from the original report.
"""
import logging

import numpy as np
import pytest

from dark_emulator import model_hod

LNAS_REPORT = 2.56935248

# Case 1: halo power spectrum extrapolated to negative values at the lowest k.
CPARAM_LOWK = [0.021486247065358267, 0.12896679664337887, 0.5994660291142397,
               LNAS_REPORT, 1.006600422971051, -1.0677034864754746]
GPARAM_LOWK = {'logMmin': 13.442382810764773, 'sigma_sq': 0.3643003196573667,
               'logM1': 12.889664059991997, 'alpha': 1.7775620417264408,
               'kappa': 0.19230764187060412, 'poff': 0.0482709156405553,
               'Roff': 0.44791239989356785, 'sat_dist_type': 'NFW',
               'alpha_inc': 4.115268896661691, 'logM_inc': 13.907088163336246}
Z_LOWK = 0.251

# Case 2: halo correlation function changes sign at the largest r, so the
# FFTLog extrapolation to larger r blew up; this HOD also has almost no galaxies.
CPARAM_SPARSE = [0.02270319004805944, 0.10964365756693839, 0.7199633979242999,
                 LNAS_REPORT, 0.9799844603062962, -0.9877734089525528]
GPARAM_SPARSE = {'logMmin': 14.4960143116711, 'sigma_sq': 0.38112677064639716,
                 'logM1': 15.283515000723925, 'alpha': 1.3096902607303873,
                 'kappa': 1.3095061600718454, 'poff': 0.9541088179653332,
                 'Roff': 0.2712190867894476, 'sat_dist_type': 'emulator',
                 'alpha_inc': 3.3810412510194237, 'logM_inc': 15.540070327639803}
Z_SPARSE = 0.484

RP = np.logspace(-1, 1.9, 20)


@pytest.fixture(scope="module")
def hod():
    return model_hod.darkemu_x_hod()


def test_wp_finite_lowk_case(hod):
    hod.set_cosmology(np.array(CPARAM_LOWK))
    hod.set_galaxy(GPARAM_LOWK)
    assert np.all(np.isfinite(hod.get_wp(RP, Z_LOWK)))
    assert np.all(np.isfinite(hod.get_wp(RP, Z_LOWK, pimax=100.)))
    assert np.all(np.isfinite(hod.get_wp(RP, Z_LOWK, pimax=100., rsd=True)))
    assert np.all(np.isfinite(hod.get_xi_gg(np.logspace(-1, 2, 10), Z_LOWK)))


def test_p_hh_follows_linear_shape_at_low_k(hod):
    hod.set_cosmology(np.array(CPARAM_LOWK))
    hod.set_galaxy(GPARAM_LOWK)
    hod.get_wp(RP, Z_LOWK)
    k = hod.fftlog_2h.k
    low = k < hod.config["p_hh_k_lin"]
    ratio = hod.p_hh_base[..., low] / hod.get_pklin(k[low])
    # P_hh/P_lin is constant (linear bias squared) below k_lin.
    assert np.allclose(ratio, ratio[..., :1], rtol=1e-10)


def test_wp_finite_and_warns_for_sparse_hod(hod, caplog):
    hod.set_cosmology(np.array(CPARAM_SPARSE))
    hod.set_galaxy(GPARAM_SPARSE)
    with caplog.at_level(logging.WARNING):
        wp = hod.get_wp(RP, Z_SPARSE)
    assert np.all(np.isfinite(wp))
    assert any("galaxy number density" in rec.getMessage() for rec in caplog.records)


def test_no_warning_for_typical_hod(hod, caplog):
    hod.set_cosmology(np.array([0.02225, 0.1198, 0.6844, 3.094, 0.9645, -1.]))
    hod.set_galaxy({'logMmin': 13.13, 'sigma_sq': 0.22, 'logM1': 14.21,
                    'alpha': 1.13, 'kappa': 1.25, 'poff': 0.2, 'Roff': 0.1,
                    'sat_dist_type': 'emulator', 'alpha_inc': 0.44,
                    'logM_inc': 13.57})
    with caplog.at_level(logging.WARNING):
        hod.get_wp(RP, 0.55)
    assert not any("galaxy number density" in rec.getMessage() for rec in caplog.records)
