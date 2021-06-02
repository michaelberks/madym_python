'''
Test file for the dce_models sub-module
'''
import pytest
import os
import sys
sys.path.insert(0, 
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

#-------------------------------------------------------------------------------
# Tests for data_io module
from QbiPy.dce_models import data_io
#-------------------------------------------------------------------------------

def test_get_dyn_vals():
    assert True

#-------------------------------------------------------------------------------
def test_get_dyn_vols():
    assert True

#-------------------------------------------------------------------------------
def test_get_dyn_xtr_data():
    assert True

#-------------------------------------------------------------------------------
# Tests for dce_aif module
from QbiPy.dce_models import dce_aif
#-------------------------------------------------------------------------------

def test_aif_default_aif_type():
    assert True

def test_aif_set_initial_file():
    assert True

def test_aif_set_initial_times():
    assert True

def test_aif_set_initial_base_aif():
    assert True

def test_aif_set_initial_base_pif():
    assert True

def test_aif_set_initial_prebolus():
    assert True

def test_aif_set_initial_hct():
    assert True

def test_aif_set_initial_dose():
    assert True

def test_aif_num_times():
    assert True

def test_aif_compute_population_AIF():
    assert True

def test_aif_read_AIF():
    assert True

def test_aif_resample_AIF():
    assert True

def test_aif_resample_PIF():
    assert True

#-------------------------------------------------------------------------------
# Tests for dibem module
from QbiPy.dce_models import dibem
#-------------------------------------------------------------------------------

def test_dibem_params_phys_to_model():
    assert True

def test_dibem_params_model_to_phys():
    assert True

def test_dibem_concentration_from_model():
    assert True

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