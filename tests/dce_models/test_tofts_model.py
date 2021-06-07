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
# Tests for tofts_model module
from QbiPy.dce_models import tofts_model, dce_aif
#-------------------------------------------------------------------------------
def test_concentration_from_model_scalar():
    times = np.linspace(0, 6, 50)
    aif = dce_aif.Aif(times = times)
    Ktrans = 0.2
    v_e = 0.1
    v_p = 0.05
    tau_a = 0.1
    C_t = tofts_model.concentration_from_model(
        aif, Ktrans, v_e, v_p, tau_a)

    assert C_t.size == times.size
    assert np.all(np.isfinite(C_t))

def test_concentration_from_model_array():
    times = np.linspace(0, 6, 50)
    aif = dce_aif.Aif(times = times)
    Ktrans = [0.2, 0.25]
    v_e = [0.1, 0.15]
    v_p = [0.05, 0.055]
    tau_a = [0.1, 0.15]
    C_t = tofts_model.concentration_from_model(
        aif, Ktrans, v_e, v_p, tau_a)

    assert C_t.size == 2*times.size
    assert np.all(np.isfinite(C_t))