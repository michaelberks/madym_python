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
# Tests for data_io module
from QbiPy.dce_models import data_io
from QbiPy.image_io import analyze_format, xtr_files
#-------------------------------------------------------------------------------

def write_sequence_volumes(nDyns):
    C_t = np.random.randn(2,2,2,nDyns)

    #Create a temporary directory where we'll run these tests, which we can then cleanup
    #easily at the end
    temp_dir = TemporaryDirectory()
    C_t_root = os.path.join(temp_dir.name, 'Ct_')
    for i_dyn in range(nDyns):
        
        #Write out 1x1 concentration maps and xtr files
        analyze_format.write_analyze(
            C_t[:,:,:,i_dyn], f'{C_t_root}{i_dyn+1}.hdr')

    return C_t, temp_dir, C_t_root

def write_sequence_xtr(nDyns, FA, TR):
    t = np.linspace(0, 5, nDyns)
    noise = np.random.randn(nDyns)

    #Create a temporary directory where we'll run these tests, which we can then cleanup
    #easily at the end
    temp_dir = TemporaryDirectory()
    C_t_root = os.path.join(temp_dir.name, 'Ct_')
    for i_dyn in range(nDyns):
        
        #Write out 1x1 concentration maps and xtr files
        timestamp = xtr_files.mins_to_timestamp(t[i_dyn])

        xtr_files.write_xtr_file(
            f'{C_t_root}{i_dyn+1}.xtr', append=False,
            FlipAngle=FA,
            TR=TR,
            TimeStamp=timestamp,
            NoiseSigma=noise[i_dyn])

    return t, noise, temp_dir, C_t_root

def test_get_dyn_vals():
    n_dyns = 50
    C_t, temp_dir, C_t_root = write_sequence_volumes(n_dyns)
    
    roi = C_t[:,:,:,0] > 0
    n_pos = np.sum(roi)
    roi_name = os.path.join(temp_dir.name, 'roi.hdr')
    analyze_format.write_analyze(roi, roi_name)

    C_t_pos1 = data_io.get_dyn_vals(C_t, n_dyns, roi)
    C_t_pos2 = data_io.get_dyn_vals(C_t, n_dyns, roi_name)
    C_t_pos3 = data_io.get_dyn_vals(C_t_root, n_dyns, roi_name)
    temp_dir.cleanup()

    assert C_t_pos1.shape == (n_pos, n_dyns)
    np.testing.assert_equal(C_t_pos1, C_t_pos2)
    np.testing.assert_almost_equal(C_t_pos1, C_t_pos3, 6)

#-------------------------------------------------------------------------------
def test_get_dyn_vols():
    n_dyns = 50
    C_t, temp_dir, C_t_root = write_sequence_volumes(n_dyns)
    C_t_in, C_t_hdrs = data_io.get_dyn_vols(C_t_root, n_dyns)
    np.testing.assert_almost_equal(C_t, C_t_in)
    assert len(C_t_hdrs) == n_dyns

#-------------------------------------------------------------------------------
def test_get_dyn_xtr_data():
    n_dyns = 50
    FA = 20
    TR = 3.5
    t, noise, temp_dir, C_t_root = write_sequence_xtr(n_dyns, FA, TR)

    dyn_times, dyn_TR, dyn_FA, dyn_noise = data_io.get_dyn_xtr_data(
        C_t_root, n_dyns)
    temp_dir.cleanup()
    
    np.testing.assert_almost_equal(FA, dyn_FA, 6)
    np.testing.assert_almost_equal(TR, dyn_TR, 6)
    np.testing.assert_almost_equal(noise, dyn_noise, 6)