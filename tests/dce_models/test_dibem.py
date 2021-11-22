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
# Tests for dibem module
from QbiPy.dce_models import dibem, two_cxm_model, active_uptake_model
from QbiPy.dce_models import dce_aif
#-------------------------------------------------------------------------------

def test_params_AUEM():
    #Test consistency of conversion to/from DIBEM
    F_p = 1.0
    v_ecs = 0.2 
    k_i = 0.25 
    k_ef = 0.05

    dibem_params = active_uptake_model.params_to_DIBEM(F_p, v_ecs, k_i, k_ef, False)
    F_p2, v_ecs2, k_i2, k_ef2 = active_uptake_model.params_from_DIBEM(*dibem_params, False)

    dibem_params = active_uptake_model.params_to_DIBEM(F_p, v_ecs, k_i, k_ef, True)
    F_p3, v_ecs3, k_i3, k_ef3 = active_uptake_model.params_from_DIBEM(*dibem_params, True)

    assert F_p == pytest.approx(F_p2)
    assert v_ecs == pytest.approx(v_ecs2)
    assert k_i == pytest.approx(k_i2)
    assert k_ef == pytest.approx(k_ef2)
    assert F_p == pytest.approx(F_p3)
    assert v_ecs == pytest.approx(v_ecs3)
    assert k_i == pytest.approx(k_i3)
    assert k_ef == pytest.approx(k_ef3)

def test_params_2CXM():
    F_p = 1.0
    PS = 0.2 
    v_e = 0.2 
    v_p = 0.1

    dibem_params = two_cxm_model.params_to_DIBEM(F_p, PS, v_e, v_p, False)
    print('params_2CXM_to_DIBEM: ', dibem_params)
    F_p2, PS2, v_e2, v_p2 = two_cxm_model.params_from_DIBEM(*dibem_params, False)

    dibem_params = two_cxm_model.params_to_DIBEM(F_p, PS, v_e, v_p, True)
    print('params_2CXM_to_DIBEM: ', dibem_params)
    F_p3, PS3, v_e3, v_p3 = two_cxm_model.params_from_DIBEM(*dibem_params, True)

    assert F_p == pytest.approx(F_p2)
    assert PS == pytest.approx(PS2)
    assert v_e == pytest.approx(v_e2)
    assert v_p == pytest.approx(v_p2)
    assert F_p == pytest.approx(F_p3)
    assert PS == pytest.approx(PS3)
    assert v_e == pytest.approx(v_e3)
    assert v_p == pytest.approx(v_p3)

def test_concentration_from_model_scalar():
    times = np.linspace(0, 6, 50)
    aif = dce_aif.Aif(times = times)
    F_pos = 0.2
    F_neg = 0.2 
    K_pos = 0.5 
    K_neg = 4.0 
    f_a = 0.5 
    tau_a = 0.1 
    tau_v = 0.05
    C_t = dibem.concentration_from_model(
        aif, 
        F_pos, F_neg, K_pos, K_neg, 
        f_a, tau_a, tau_v
    )
    assert C_t.size == times.size
    assert np.all(np.isfinite(C_t))

def test_concentration_from_array():
    times = np.linspace(0, 6, 50)
    aif = dce_aif.Aif(times = times)
    F_pos = [0.2, 0.25]
    F_neg = [0.2, 0.25] 
    K_pos = [0.5, 0.55] 
    K_neg = [4.0, 4.5] 
    f_a = 0.5 
    tau_a = 0.1 
    tau_v = 0.05
    C_t = dibem.concentration_from_model(
        aif, 
        F_pos, F_neg, K_pos, K_neg, 
        f_a, tau_a, tau_v
    )
    assert C_t.size == 2*times.size
    assert np.all(np.isfinite(C_t))