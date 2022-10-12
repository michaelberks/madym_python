'''
Test read/write functions for Analyze 7.5 data

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
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from QbiPy.image_io import analyze_format

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
    np.testing.assert_almost_equal(img, img_r, 6)

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
                np.testing.assert_equal(img_integer, img_r)

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
    temp_dir.cleanup()

    #Check read image matches original
    print(f"Test read: flip {scale}")
    assert img_real.shape == img_r.shape
    assert np.all(img_real == img_r)

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
    
def test_big_endian():
    #Because we never write to big-endian, our only way of testing
    #is to load in an old big-endian image we have
    img_path = os.path.join(os.path.dirname(__file__), 'dyn_1')
    img,hdr = analyze_format.read_analyze(img_path)
    assert img.shape == (128,128,25)
    assert hdr.ByteOrder == 'ieee-be'

@pytest.mark.parametrize("ext",
    [('.nii.gz'), ('.nii.gz')])
def test_read_write_nifti_ext(ext):
    #Create temp location for the read/write
    temp_dir = TemporaryDirectory()
    img_name = os.path.join(temp_dir.name, 'test_img' + ext)

    #Write to nifti image
    print(f"Test nifti write: {ext}")
    analyze_format.write_analyze(img_real, img_name)
    img_r, img_hdr = analyze_format.read_analyze(img_name)
    temp_dir.cleanup()

    #Check read image matches original
    print(f"Test nifti read: {ext}")
    assert img_real.shape == img_r.shape
    assert np.all(img_real == img_r)
    assert np.all(img_hdr.SformMatrix == np.eye(4))

@pytest.mark.parametrize("sform_matrix",
    [(np.array([2,3,4])), (np.diag([2,3,4,1]))])
def test_read_write_nifti_sform(sform_matrix):
    #Create temp location for the read/write
    temp_dir = TemporaryDirectory()
    img_name = os.path.join(temp_dir.name, 'test_img.nii.gz')

    #Write to nifti image
    print(f"Test nifti write: sform_matrix shape = {sform_matrix.shape}")
    analyze_format.write_analyze(img_real, img_name, voxel_size=sform_matrix)
    img_r, img_hdr = analyze_format.read_analyze(img_name)
    temp_dir.cleanup()

    #Check read image matches original
    print(f"Test read: sform_matrix shape = {sform_matrix.shape}")
    assert img_real.shape == img_r.shape
    assert np.all(img_real == img_r)
    if sform_matrix.size == 3:
        assert np.all(img_hdr.SformMatrix[(0,1,2),(0,1,2)] == sform_matrix)
    else:
        assert np.all(img_hdr.SformMatrix == sform_matrix)
    