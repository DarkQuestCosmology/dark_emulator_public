"""The undocumented 'baryon' galaxy parameter is not supported in the public version (#47)."""
import sys

import numpy as np
import pytest

from dark_emulator import model_hod

GPARAM = {'logMmin': 13.13, 'sigma_sq': 0.22, 'logM1': 14.21, 'alpha': 1.13,
          'kappa': 1.25, 'poff': 0.2, 'Roff': 0.1, 'sat_dist_type': 'emulator',
          'alpha_inc': 0.44, 'logM_inc': 13.57}


def test_baryon_parameter_raises_not_implemented():
    hod = model_hod.darkemu_x_hod()
    hod.set_cosmology(np.array([0.02225, 0.1198, 0.6844, 3.094, 0.9645, -1.]))
    with pytest.raises(NotImplementedError, match="baryon"):
        hod.set_galaxy(dict(GPARAM, baryon=[13., 0.5, 0.5]))


def test_import_does_not_modify_sys_path():
    assert not any(p.rstrip("/").endswith("baryon") for p in sys.path)
