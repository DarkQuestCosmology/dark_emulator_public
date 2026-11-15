"""The NFW satellite distribution needs colossus; without it a clear ImportError is raised (#43)."""
import numpy as np
import pytest

from dark_emulator import model_hod
from dark_emulator.model_hod import hod_interface

GPARAM_NFW = {'logMmin': 13.13, 'sigma_sq': 0.22, 'logM1': 14.21, 'alpha': 1.13,
              'kappa': 1.25, 'poff': 0.2, 'Roff': 0.1, 'sat_dist_type': 'NFW',
              'alpha_inc': 0.44, 'logM_inc': 13.57}


def test_nfw_without_colossus_raises_import_error(monkeypatch):
    monkeypatch.setattr(hod_interface, "_colossus_available", False)
    hod = model_hod.darkemu_x_hod()
    hod.set_cosmology(np.array([0.02225, 0.1198, 0.6844, 3.094, 0.9645, -1.]))
    hod.set_galaxy(GPARAM_NFW)
    with pytest.raises(ImportError, match="pip install colossus"):
        hod.get_wp(np.logspace(-1, 1.5, 5), 0.55)
