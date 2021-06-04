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
from QbiPy.dce_models import two_cxm_model
#-------------------------------------------------------------------------------
def test_two_cxm_model_params_phys_to_model():
    assert True

def test_two_cxm_model_params_model_to_phys():
    assert True

def test_two_cxm_model_concentration_from_model():
    assert True