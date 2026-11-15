import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.special import gamma

from dark_emulator import model_hod
from dark_emulator.darkemu.cosmo_util import cosmo_class, constants

CPARAM = np.array([0.02225, 0.1198, 0.6844, 3.094, 0.9645, -1.])
GPARAM = {"logMmin": 13.13, "sigma_sq": 0.22, "logM1": 14.21, "alpha": 1.13, "kappa": 1.25,
          "poff": 0.2, "Roff": 0.1, "sat_dist_type": "emulator", "alpha_inc": 0.44, "logM_inc": 13.57}


@pytest.fixture(scope="module")
def hod():
    h = model_hod.darkemu_x_hod()
    h.set_cosmology(CPARAM)
    h.set_galaxy(GPARAM)
    return h


def _gaussian_dndz(zc=0.55, sigma=0.07, n=31):
    zs = np.linspace(zc - 3.5 * sigma, zc + 3.5 * sigma, n)
    return zs, np.exp(-0.5 * ((zs - zc) / sigma)**2)


def test_wtheta_power_law_matches_limber_formula(hod, monkeypatch):
    """For xi = (r/r0)^-gamma, w(theta) must equal Peebles' Limber formula (e.g., Eqs. 31-33 of Harikane et al. 2016)."""
    r0, gam = 5.0, 1.8
    C = np.sqrt(np.pi) * gamma((gam - 1) / 2) / gamma(gam / 2)
    monkeypatch.setattr(hod, "get_wp", lambda rp, z, **kw: C * r0**gam * np.asarray(rp)**(1 - gam))
    zs, dndz = _gaussian_dndz()
    theta = np.logspace(-3, 0, 7)
    w = hod.get_wtheta(theta, zs, dndz)

    cosmo = cosmo_class()
    cosmo.cparam = CPARAM.reshape(1, 6)
    chi = cosmo.get_comoving_distance(zs)
    hz = np.array([cosmo.get_Ez(z) for z in zs]) / (constants.c_km_s / 100.)
    n = dndz / trapezoid(dndz, zs)
    expected = C * r0**gam * trapezoid(n**2 * hz * chi**(1 - gam), zs) * np.deg2rad(theta)**(1 - gam)
    np.testing.assert_allclose(w, expected, rtol=1e-10)


def test_wtheta_is_positive_and_decreasing(hod):
    zs, dndz = _gaussian_dndz(n=11)
    theta = np.logspace(-3, 0, 7)
    w = hod.get_wtheta(theta, zs, dndz)
    assert np.all(np.isfinite(w))
    assert np.all(w > 0)
    assert np.all(np.diff(w) < 0)
    # the normalization of dndz must not matter
    np.testing.assert_allclose(hod.get_wtheta(theta, zs, 3.7 * dndz), w, rtol=1e-12)


@pytest.mark.parametrize("zs, dndz", [
    (np.array([0.5]), np.array([1.])),                     # too short
    (np.array([0.6, 0.5]), np.array([1., 1.])),            # not ascending
    (np.array([0.5, 0.6]), np.array([1., -1.])),           # negative dndz
    (np.array([0.5, 0.6]), np.array([0., 0.])),            # zero integral
    (np.array([0.5, 0.6, 0.7]), np.array([1., 1.])),       # shape mismatch
])
def test_wtheta_rejects_invalid_input(hod, zs, dndz):
    with pytest.raises(ValueError):
        hod.get_wtheta(np.array([0.1]), zs, dndz)
