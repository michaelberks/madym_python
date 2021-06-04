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
def test_tissue_concentration_signal_to_concentration():
    assert True

def test_tissue_concentration_concentration_to_signal():
    assert True

def test_tissue_concentration_compute_IAUC():
    assert True