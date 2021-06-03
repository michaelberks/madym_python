'''
Test for sub-module image_io, and in particular the read/write functions
for Analyze 7.5 data

read_analyze(filename: str=None,
    output_type:np.dtype=np.float64, scale:float = 1.0, 
    flip_y: bool = True, flip_x: bool = False,
    swap_axes: bool = True)

write_analyze(img_data: np.array, filename: str,
    scale:float = 1.0, swap_axes:bool = True,
    flip_x: bool = False, flip_y: bool = True,
    voxel_size=[1,1,1], dtype=None, sparse=False)
'''

import pytest
import os
import sys
import numpy as np
from tempfile import TemporaryDirectory
sys.path.insert(0, 
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from QbiPy.image_io import analyze_format, xtr_files

img_integer = np.array([
        [[1, 2], [3,4]],
        [[0, 0], [0,0]] ] )

img_real = np.array([
        [[1.1, 2.2], [3.3,4.4]],
        [[0, 0], [0,0]] ] )

#-----------------------------------------------------------------
# Test writing out analyze image in various formats
#-----------------------------------------------------------------
@pytest.mark.parametrize("img, format, sparse", [
    #unsigned char (uint8) format
    (img_integer, "DT_UNSIGNED_CHAR", False),

    #Short format (16)
    (img_integer, "DT_SIGNED_SHORT", False),

    #Integer format
    (img_integer, "DT_SIGNED_INT", False),

    #Float format
    (img_integer, "DT_FLOAT", False),

    #Double format
    (img_integer, "DT_DOUBLE", False),

    #-----------------------------------------------------------------
    # Check it works for real valued data
    #-----------------------------------------------------------------

    #Float format - real data
    (img_real, "DT_FLOAT", False),

    #Double format - real data
    (img_real, "DT_DOUBLE", False),

    #-----------------------------------------------------------------
    # Now repeat the tests for sparse writing/reading
    #-----------------------------------------------------------------

    #unsigned char (uint8) format
    (img_integer, "DT_UNSIGNED_CHAR", True),

    #Short format (16)
    (img_integer, "DT_SIGNED_SHORT", True),

    #Integer format
    (img_integer, "DT_SIGNED_INT", True),

    #Float format
    (img_integer, "DT_FLOAT", True),

    #Double format
    (img_integer, "DT_DOUBLE", True),

    #Float format - real data
    (img_real, "DT_FLOAT", True),

    #Double format - real data
    (img_real, "DT_DOUBLE", True)
])

def test_analyze_write_read_format(img, format, sparse):
    #Create temp location for the read/write
    temp_dir = TemporaryDirectory()

    sparse_str = ""
    if sparse:
        sparse_str = " - sparse"

    #Write out image
    print(f"Test write: format {format}{sparse_str}")
    img_name = os.path.join(temp_dir.name, format)
    dtype = analyze_format.format_str_analyze_to_numpy(format)
    analyze_format.write_analyze(img, img_name, 
        dtype = dtype, sparse=sparse)

    #Read it back in
    img_r = analyze_format.read_analyze(img_name)[0]
    temp_dir.cleanup()

    #Check read image matches original
    print(f"Test read: format {format}{sparse_str}")
    assert img.shape == img_r.shape
    assert np.all(np.abs(img - img_r) < 1e-4)

def test_analyze_write_read_flip():
    #Create temp location for the read/write
    temp_dir = TemporaryDirectory()

    img_name = os.path.join(temp_dir.name, 'test_img')
    for flip_x in [True, False]:
        for flip_y in [True, False]:
            for swap_axes in [True, False]:

                #Write out image
                print(f"Test write: flip ({flip_x}, {flip_y}, {swap_axes})")
                analyze_format.write_analyze(
                    img_integer, img_name, 
                    flip_y = flip_y, flip_x = flip_x,
                    swap_axes = swap_axes)

                #Read it back in
                img_r = analyze_format.read_analyze(
                    img_name, 
                    flip_y = flip_y, flip_x = flip_x,
                    swap_axes = swap_axes)[0]

                #Check read image matches original
                print(f"Test read: flip ({flip_x}, {flip_y}, {swap_axes})")
                assert img_integer.shape == img_r.shape
                assert np.all(img_integer == img_r)

    temp_dir.cleanup()

@pytest.mark.parametrize("scale",
    [(1), (0.1), (10)])
def test_analyze_write_read_scale(scale):
    #Create temp location for the read/write
    temp_dir = TemporaryDirectory()
    img_name = os.path.join(temp_dir.name, 'test_img')

    #Write out image
    print(f"Test write: scale {scale}")
    analyze_format.write_analyze(
        img_real, img_name, 
        scale = scale)

    #Read it back in
    img_r = analyze_format.read_analyze(
        img_name, 
        scale = scale)[0]

    #Check read image matches original
    print(f"Test read: flip {scale}")
    assert img_real.shape == img_r.shape
    assert np.all(img_real == img_r)

    temp_dir.cleanup()

#Test the format conversion aux fucntions for consistency
@pytest.mark.parametrize("ana_str", [
    #unsigned char (uint8) format
    ("DT_UNSIGNED_CHAR"),

    #Short format (16)
    ("DT_SIGNED_SHORT"),

    #Integer format
    ("DT_SIGNED_INT"),

    #Float format
    ("DT_FLOAT"),

    #Double format
    ("DT_DOUBLE")])

def test_format_specifiers(ana_str):

    np_str = analyze_format.format_str_analyze_to_numpy(ana_str)
    assert ana_str == analyze_format.format_str_numpy_to_analyze(np_str)
    
    #format_str_analyze_to_struct(ana_str)
    #format_str_struct_to_analyze(struct_str)
    
    #format_str_numpy_to_struct(np_str)
    #format_str_struct_to_numpy(struct_str)
    
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

    '''
    read_jim_roi(roi_path, roi_dims, vox_size, make_mask = False, mask_res = 1)
    write_jim_roi(roi, thresh, vox_size, jim_path, min_contour=0,
        version_str="", image_src="", operator="AutoGenerated")
    write_jim_roi_from_list(contour_list, jim_path,
        version_str="", image_src="", operator="AutoGenerated")
    write_jim_roi_from_slice_info(slice_info, jim_path,
        operator="AutoGenerated", scale=(1,1), offset=(0,0))
    '''
    from QbiPy.image_io import jim_io
    def test_write_read_jim_roi():
        assert True
    