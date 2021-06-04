'''
Test read/write functions for XTR files
'''

import pytest
import os
import sys
import numpy as np
from tempfile import TemporaryDirectory
sys.path.insert(0, 
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from QbiPy.image_io import xtr_files
   
#Test for writing/reading xtr file
def test_xtr():
    FA = 20
    TR = 3
    TE = 1
    time = 123456.789

    temp_dir = TemporaryDirectory()
    xtr_name = os.path.join(temp_dir.name, 'temp.xtr')

    xtr_files.write_xtr_file(
        xtr_name, append =False, 
        TR=TR, FlipAngle=FA, TimeStamp=time,
    )
    xtr_files.write_xtr_file(
        xtr_name, append =True, 
        TE=TE,
    )

    xtr_data = xtr_files.read_xtr_file(xtr_name)
    assert FA == pytest.approx(xtr_data['FlipAngle'])
    assert TR == pytest.approx(xtr_data['TR'])
    assert TE == pytest.approx(xtr_data['TE'])
    assert time == pytest.approx(xtr_data['TimeStamp'])
    temp_dir.cleanup()
    