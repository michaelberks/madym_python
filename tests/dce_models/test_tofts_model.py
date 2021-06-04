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
from QbiPy.dce_models import tofts_model
#-------------------------------------------------------------------------------
def test_tofts_model_concentration_from_model():
    tofts_model.concentration_from_model()
    
    assert True