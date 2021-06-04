'''
Test read/write functions JIM roi files

read_jim_roi(roi_path, roi_dims, vox_size, make_mask = False, mask_res = 1)
write_jim_roi(roi, thresh, vox_size, jim_path, min_contour=0,
    version_str="", image_src="", operator="AutoGenerated")
write_jim_roi_from_list(contour_list, jim_path,
    version_str="", image_src="", operator="AutoGenerated")
write_jim_roi_from_slice_info(slice_info, jim_path,
    operator="AutoGenerated", scale=(1,1), offset=(0,0))
'''

import pytest
import os
import sys
import numpy as np
from tempfile import TemporaryDirectory
sys.path.insert(0, 
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from QbiPy.image_io import jim_io
def test_write_read_jim_roi():
    assert True