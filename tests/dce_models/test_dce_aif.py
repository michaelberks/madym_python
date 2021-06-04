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

    np.testing.assert_equal(aif_in.base_aif_, base_aif)
    np.testing.assert_equal(aif_in.times_, times)
    np.testing.assert_equal(aif_in_t.base_aif_, base_aif)
    np.testing.assert_equal(aif_in_t.times_, times2)

def test_aif_set_initial_times():
    times = np.arange(10)
    aif = dce_aif.Aif(times = times)
    np.testing.assert_equal(aif.times_, times)

def test_aif_set_initial_base_aif():
    base_aif = np.array([0, 0, 1, 0.5, 0.2])
    aif = dce_aif.Aif(base_aif=base_aif, aif_type=dce_aif.AifType.ARRAY)
    np.testing.assert_equal(aif.base_aif_, base_aif)

def test_aif_set_initial_base_pif():
    base_pif = np.array([0, 0, 1, 0.5, 0.2])
    aif = dce_aif.Aif(base_pif=base_pif)
    np.testing.assert_equal(aif.base_pif_, base_pif)

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