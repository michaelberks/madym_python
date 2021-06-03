'''
Test file for the dce_models sub-module
'''
import pytest
import os
import sys
import numpy as np
from tempfile import TemporaryDirectory
sys.path.insert(0, 
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

#-------------------------------------------------------------------------------
# Tests for data_io module
from QbiPy.dce_models import data_io
#-------------------------------------------------------------------------------

def test_get_dyn_vals():
    #data_io.get_dyn_vals(root_path, num_vols, roi, index_fmt = '01d')
    assert True

#-------------------------------------------------------------------------------
def test_get_dyn_vols():
    #data_io.get_dyn_vols(root_path, num_vols, apply_smoothing=False, 
    #    index_fmt = '01d', load_headers=False)
    assert True

#-------------------------------------------------------------------------------
def test_get_dyn_xtr_data():
    #data_io.get_dyn_xtr_data(root_path, num_vols, index_fmt = '01d')
    assert True

#-------------------------------------------------------------------------------
# Tests for dce_aif module
from QbiPy.dce_models import dce_aif
#-------------------------------------------------------------------------------

def test_aif_default_aif_type():
    aif = dce_aif.Aif()
    assert aif.type_ == dce_aif.AifType.POP 

def test_aif_set_initial_type():
    aif_POP = dce_aif.Aif(aif_type=dce_aif.AifType.POP)
    aif_STD = dce_aif.Aif(aif_type=dce_aif.AifType.STD)
    aif_FILE = dce_aif.Aif(aif_type=dce_aif.AifType.FILE)
    aif_ARRAY = dce_aif.Aif(aif_type=dce_aif.AifType.ARRAY)
    
    assert aif_POP.type_ == dce_aif.AifType.POP
    assert aif_STD.type_ == dce_aif.AifType.STD
    assert aif_FILE.type_ == dce_aif.AifType.FILE
    assert aif_ARRAY.type_ == dce_aif.AifType.ARRAY

def test_aif_set_initial_file():
    #This also test read and write AIF
    times = np.arange(10)
    base_aif = 0.1*np.arange(10)  

    #Create AIF and write to a temp dir
    aif = dce_aif.Aif(
        times = times, aif_type=dce_aif.AifType.ARRAY, base_aif=base_aif)
    temp_dir = TemporaryDirectory()
    aif_path = os.path.join(temp_dir.name, 'aif.txt')
    aif.write_AIF(aif_path)

    #Read in AIF from file
    aif_in = dce_aif.Aif(
        aif_type=dce_aif.AifType.FILE, filename=aif_path)

    #Check if we supply times in constructor they're not overwritten
    times2 = np.arange(10)+1
    aif_in_t = dce_aif.Aif(
        times = times2, aif_type=dce_aif.AifType.FILE, filename=aif_path)

    temp_dir.cleanup()

    assert np.all(aif_in.base_aif_ == base_aif)
    assert np.all(aif_in.times_ == times)
    assert np.all(aif_in_t.base_aif_ == base_aif)
    assert np.all(aif_in_t.times_ == times2)

def test_aif_set_initial_times():
    times = np.arange(10)
    aif = dce_aif.Aif(times = times)
    assert np.all(aif.times_ == times)

def test_aif_set_initial_base_aif():
    base_aif = np.array([0, 0, 1, 0.5, 0.2])
    aif = dce_aif.Aif(base_aif=base_aif, aif_type=dce_aif.AifType.ARRAY)
    assert np.all(aif.base_aif_ == base_aif)

def test_aif_set_initial_base_pif():
    base_pif = np.array([0, 0, 1, 0.5, 0.2])
    aif = dce_aif.Aif(base_pif=base_pif)
    assert np.all(aif.base_pif_ == base_pif)

def test_aif_set_initial_prebolus():
    aif = dce_aif.Aif(prebolus=5)
    assert aif.prebolus_ == 5

def test_aif_set_initial_hct():
    aif = dce_aif.Aif(hct=0.4)
    assert aif.hct_ == 0.4

def test_aif_set_initial_dose():
    aif = dce_aif.Aif(dose=0.15)
    assert aif.dose_ == 0.15

def test_aif_num_times():
    times = np.arange(5)
    aif = dce_aif.Aif(times = times, aif_type=dce_aif.AifType.ARRAY)
    empty_aif = dce_aif.Aif()
    assert aif.num_times() == times.size
    assert empty_aif.num_times() == 0

def test_aif_compute_population_AIF():
    times = np.arange(100)
    aif = dce_aif.Aif(times = times, aif_type=dce_aif.AifType.POP)
    assert aif.base_aif_.size == times.size

def test_aif_resample_AIF():
    times = np.arange(100)
    aif = dce_aif.Aif(times = times, aif_type=dce_aif.AifType.POP)
    aif.resample_AIF(0.1)

def test_aif_resample_PIF():
    times = np.arange(100)
    aif = dce_aif.Aif(times = times, aif_type=dce_aif.AifType.POP)
    aif.resample_PIF(0.1, True, True)

#-------------------------------------------------------------------------------
# Tests for dibem module
from QbiPy.dce_models import dibem
#-------------------------------------------------------------------------------

def test_dibem_params_AUEM():
    #Test consistency of conversion to/from DIBEM
    F_p = 1.0
    v_ecs = 0.2 
    k_i = 0.25 
    k_ef = 0.05

    dibem_params = dibem.params_AUEM_to_DIBEM(F_p, v_ecs, k_i, k_ef, False)
    F_p2, v_ecs2, k_i2, k_ef2 = dibem.params_DIBEM_to_AUEM(*dibem_params, False)

    dibem_params = dibem.params_AUEM_to_DIBEM(F_p, v_ecs, k_i, k_ef, True)
    F_p3, v_ecs3, k_i3, k_ef3 = dibem.params_DIBEM_to_AUEM(*dibem_params, True)

    assert F_p == pytest.approx(F_p2)
    assert v_ecs == pytest.approx(v_ecs2)
    assert k_i == pytest.approx(k_i2)
    assert k_ef == pytest.approx(k_ef2)
    assert F_p == pytest.approx(F_p3)
    assert v_ecs == pytest.approx(v_ecs3)
    assert k_i == pytest.approx(k_i3)
    assert k_ef == pytest.approx(k_ef3)

def test_dibem_params_2CXM():
    F_p = 1.0
    PS = 0.2 
    v_e = 0.2 
    v_p = 0.1

    dibem_params = dibem.params_2CXM_to_DIBEM(F_p, PS, v_e, v_p, False)
    print('params_2CXM_to_DIBEM: ', dibem_params)
    F_p2, PS2, v_e2, v_p2 = dibem.params_DIBEM_to_2CXM(*dibem_params, False)

    dibem_params = dibem.params_2CXM_to_DIBEM(F_p, PS, v_e, v_p, True)
    print('params_2CXM_to_DIBEM: ', dibem_params)
    F_p3, PS3, v_e3, v_p3 = dibem.params_DIBEM_to_2CXM(*dibem_params, True)

    assert F_p == pytest.approx(F_p2)
    assert PS == pytest.approx(PS2)
    assert v_e == pytest.approx(v_e2)
    assert v_p == pytest.approx(v_p2)
    assert F_p == pytest.approx(F_p3)
    assert PS == pytest.approx(PS3)
    assert v_e == pytest.approx(v_e3)
    assert v_p == pytest.approx(v_p3)

def test_dibem_concentration_from_model():
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

#-------------------------------------------------------------------------------
# Tests for tissue_concentration module
from QbiPy.dce_models import tissue_concentration
#-------------------------------------------------------------------------------
def test_tissue_concentration_signal_to_concentration():
    assert True

def test_tissue_concentration_concentration_to_signal():
    assert True

def test_tissue_concentration_compute_IAUC():
    assert True

#-------------------------------------------------------------------------------
# Tests for tofts_model module
from QbiPy.dce_models import tofts_model
#-------------------------------------------------------------------------------
def test_tofts_model_concentration_from_model():
    assert True

#-------------------------------------------------------------------------------
# Tests for two_cxm_model module
from QbiPy.dce_models import two_cxm_model
#-------------------------------------------------------------------------------
def test_two_cxm_model_params_phys_to_model():
    assert True

def test_two_cxm_model_params_model_to_phys():
    assert True

def test_two_cxm_model_concentration_from_model():
    assert True