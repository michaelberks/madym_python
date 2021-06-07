'''
Test file for the dce_models sub-module
'''
import pytest
import os
import sys
import numpy as np
from tempfile import TemporaryDirectory
sys.path.insert(0, 
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

#-------------------------------------------------------------------------------
# Tests for two_cxm_model module
from QbiPy.dce_models import two_cxm_model, dce_aif
#-------------------------------------------------------------------------------
def test_concentration_from_model_scalar():
    times = np.linspace(0, 6, 50)
    aif = dce_aif.Aif(times = times)
    F_p = 1.0
    PS = 0.2
    v_e = 0.1
    v_p = 0.05
    tau_a = 0.1
    C_t = two_cxm_model.concentration_from_model(
        aif, F_p, PS, v_e, v_p, tau_a)

    assert C_t.size == times.size
    assert np.all(np.isfinite(C_t))

def test_concentration_from_model_array():
    times = np.linspace(0, 6, 50)
    aif = dce_aif.Aif(times = times)
    F_p = [1.0, 1.25]
    PS = [0.2, 0.25]
    v_e = [0.1, 0.15]
    v_p = [0.05, 0.055]
    tau_a = [0.1, 0.15]
    C_t = two_cxm_model.concentration_from_model(
        aif, F_p, PS, v_e, v_p, tau_a)

    assert C_t.size == 2*times.size
    assert np.all(np.isfinite(C_t))