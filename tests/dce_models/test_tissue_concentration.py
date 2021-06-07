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
# Tests for tissue_concentration module
from QbiPy.dce_models import tissue_concentration
#-------------------------------------------------------------------------------
@pytest.mark.parametrize("use_M0_ratio",
    [(True),(False)])
def test_signal_tofrom_concentration(use_M0_ratio):
    n_t = 10
    inj_img = 2
    n_vox = 5
    S_t = 1e4 + 100*np.random.randn(n_vox, n_t)
    S_t[:,inj_img:] += 1e4 + 100*np.random.randn(n_vox, n_t-inj_img)
    T1_0 = 1e3 + 10*np.random.randn(n_vox)
    
    FA = 20
    TR = 4
    r1 = 3.4

    if use_M0_ratio:
        C_t = tissue_concentration.signal_to_concentration(
            S_t, T1_0, [], FA, TR, r1, inj_img)
        S_0 = np.mean(S_t[:,:inj_img],1)
        S_t2 = tissue_concentration.concentration_to_signal(
            C_t, T1_0, S_0, FA, TR, r1, inj_img)
    else:
        M0 = 5e5 + 1e5*np.random.randn(n_vox)
        C_t = tissue_concentration.signal_to_concentration(
            S_t, T1_0, M0, FA, TR, r1, 0)
        S_t2 = tissue_concentration.concentration_to_signal(
            C_t, T1_0, M0, FA, TR, r1, 0)

    assert C_t.shape == (n_vox, n_t)
    assert S_t.shape == (n_vox, n_t)
    np.testing.assert_almost_equal(S_t, S_t2)

def test_compute_IAUC():
    n_t = 10
    inj_img = 2
    n_vox = 5
    C_t = 0.1*np.random.randn(n_vox, n_t)
    C_t[:,inj_img:] += 1 + 0.1*np.random.randn(n_vox, n_t-inj_img)
    t = np.linspace(0, 5, n_t)
    interval = 60.0 
    time_scaling = 60.0
    iauc = tissue_concentration.compute_IAUC(
        C_t, t, inj_img, interval, time_scaling)
    assert iauc.size == n_vox
    assert np.all(np.isfinite(iauc))