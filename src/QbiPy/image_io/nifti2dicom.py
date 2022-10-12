'''

Hat tip to: https:#pycad.co/nifti2dicom/ for an overview of the
basic approach
'''
#%%
import os
import glob

import numpy as np
import nibabel as nib
import pydicom



#%%
def convert_nifti(nifti_im:str, dicom_im:str, output_dir:str,
    series_name:str, series_number:str,
    slice_name:str = 'DCM', sequence_fmt:str = '04d',
    start_index = 1,
    acquisition_number = None,
    number_of_temporal_positions = None,
    temporal_position = None,
    override_voxel_spacing = False,
    override_orientation = False,
    voxel_min = None,
    voxel_max = None,
    nan_default = 0,
    scaling = None,
    offset = None,
    flip_x = False,
    flip_y = False,
    flip_z = False):
    '''
    '''
    #Get nifti dims
    if type(nifti_im) == str:
        nifti_im = nib.load(nifti_im)
    img = nifti_im.get_fdata()
    hdr = nifti_im.header
    
    #Make sure image is 4D - expanding dims if necssary - then get dimensions
    img.shape += (1,) * (4 - img.ndim)
    n_rows, n_cols, n_slices, n_times = img.shape

    #Apply axes flips
    img = np.swapaxes(img, 0, 1)
    if flip_x:
        img = np.flip(img, 1)
    if flip_y:
        img = np.flip(img, 0)
    if flip_z:
        img = np.flip(img, 2)

    #Get sform and negate the 3rd row due to NIFTI vs DICOM
    #differences in axes directions
    sform = nifti_im.get_sform()
    sform[2,:] *= -1

    #Load dicom base image and set fields common to all slices
    if type(dicom_im) == str:
        dicom_im = pydicom.dcmread(dicom_im)
    dicom_im.ProtocolName = series_name
    dicom_im.SeriesDescription = series_name
    dicom_im.SeriesNumber = series_number
    if acquisition_number is None:
        acquisition_number = series_number
    dicom_im.AcquisitionNumber = acquisition_number
    
    #Set slice dimensions
    dicom_im.Rows = n_rows
    dicom_im.Columns = n_cols

    #Set number of temporal positions
    if number_of_temporal_positions is None:
        number_of_temporal_positions = n_times
    dicom_im.NumberOfTemporalPositions = number_of_temporal_positions

    #Set voxel spacing from NIFTI header info
    if override_voxel_spacing:
        dicom_im.SpacingBetweenSlices = np.linalg.norm(sform[:,2])
        dicom_im.SliceThickness = hdr.get('pixdim')[3]
        dicom_im.PixelSpacing  = hdr.get('pixdim')[1:3].tolist()
    
    #Set axes orientation
    if override_orientation or flip_x or flip_y:
        dicom_im.ImageOrientationPatient = orientation_from_sform(sform, flip_x, flip_y)

    #Get transformed origin
    origin,slice_axis = origin_from_sform(
        sform, flip_x, flip_y, flip_z, n_cols, n_rows, n_slices)

    #Deal with NaNs and limits
    img[np.isnan(img)] = nan_default

    if voxel_min is not None:
        img[img < voxel_min] = voxel_min

    if voxel_max is not None:
        img[img > voxel_max] = voxel_max

    #Set pixel value scaling and representation 
    if offset is None:
        offset = np.min(img[np.isfinite(img)])

    if scaling is None:
        img_max = np.max(img[np.isfinite(img)])
        scaling = (img_max - offset) / (2**16 - 1)
    
    dicom_im.RescaleIntercept = offset
    dicom_im.RescaleSlope = scaling
    dicom_im.RescaleType = 'normalized'

    dicom_im.PhotometricInterpretation = "MONOCHROME2"
    dicom_im.SamplesPerPixel = 1
    dicom_im.BitsStored = 16
    dicom_im.BitsAllocated = 16
    dicom_im.HighBit = 15
    dicom_im.PixelRepresentation = 0

    #Get Instance UID
    instance_UID = '.'.join(dicom_im.SOPInstanceUID.split('.')[:-2])
    instance_UID += f'.{series_number}'
    dicom_im.SeriesInstanceUID = instance_UID

    #Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    #Loop over temporal positions
    converted_slices = 0
    for time in range(n_times):
        
        #Set temporal position identifier
        if temporal_position is None:
            temporal_position = time + 1
        dicom_im.TemporalPositionIdentifier = temporal_position

        #Loop over slices
        for slice in range(n_slices):
            idx = time*n_slices + slice + start_index
            dicom_name = os.path.join(output_dir, f'{slice_name}{idx:{sequence_fmt}}')
            dicom_im.SOPInstanceUID = instance_UID + f'.{idx}'
            write_slice(
                dicom_im, dicom_name, slice, img[:,:,slice,time], 
                origin, slice_axis, idx)
            converted_slices += 1

    print(f'Successfully converted {converted_slices} slices '
        f'for series {series_name} ({series_number})')
    return converted_slices

def origin_from_sform(sform, flip_x, flip_y, flip_z, nx, ny, nz):
    '''
    '''
    #The origin for NIFTI will be offset from the DICOM origin, depending
    #on whether the image was flipped in X/Y/Z when loaded from the DICOM slices
    #so account for these offsets
    offset_u = -sform[0:3,0] * (nx - 1) if flip_x else 0
    offset_v = -sform[0:3,1] * (ny - 1) if flip_y else 0
    offset_w = -sform[0:3,2] * (nz - 1) if flip_z else 0

    origin = -sform[0:3,3] + offset_u + offset_v + offset_w
    slice_axis = sform[0:3,2] if flip_z else -sform[0:3,2]
    return origin, slice_axis

def orientation_from_sform(sform, flip_x, flip_y):
    '''
    Convert the image position and orientation from NIFTI's sform fields
    to Madym's 3D image meta data
    '''
    #Compute the row (u) and column (v) axes vectors from the Sform matrix
    dx = np.linalg.norm(sform[:,0])
    dy = np.linalg.norm(sform[:,1])
    
    #Flipping axes also changes their sign in the transform matrix
    sign_u = 1.0 if flip_x else -1.0
    sign_v = 1.0 if flip_y else -1.0

    axes_orientation = np.empty((6))
    axes_orientation[0:3] = sign_u * sform[0:3,0] / dx
    axes_orientation[3:6] = sign_v * sform[0:3,1] / dy

    return axes_orientation.tolist()

def get_slice_origin(origin, slice_num, slice_axis):
    '''

    '''
    slice_origin = origin + slice_axis*slice_num
    return slice_origin.tolist()

def set_pixel_array(slice_array, dicom_im):
    '''
    '''
    scaled_array = (slice_array - dicom_im.RescaleIntercept) / dicom_im.RescaleSlope
    dicom_im.PixelData = scaled_array.astype('uint16').tobytes()
    if 'LargestImagePixelValue' in dicom_im: 
        dicom_im.LargestImagePixelValue.VR = 'US'
        dicom_im.LargestImagePixelValue = np.max(scaled_array.astype('uint16'))
    
def write_slice(dicom_im, dicom_name, slice, slice_array, 
    origin, slice_axis, index):
    """
    `arr`: parameter will take a numpy array that represents only one slice.
    `file_dir`: parameter will take the path to save the slices
    `index`: parameter will represent the index of the slice, so this parameter will be used to put 
    the name of each slice while using a for loop to convert all the slices
    """
    #Set slice index
    dicom_im.InstanceNumber = index
    dicom_im.InStackPositionNumber = slice + 1

    #Get slice position from NIFTI header information
    dicom_im.ImagePositionPatient = get_slice_origin(origin, slice, slice_axis)
    dicom_im.SliceLocation = slice*dicom_im.SpacingBetweenSlices
        
    #Set slice pixel data
    set_pixel_array(slice_array, dicom_im)

    #Save the dicom_im
    dicom_im.save_as(dicom_name)
    #print('Saved ', dicom_name)

def convert_nifti_dir(nifti_dir:str, dicom_im:str, 
    time_series:bool = False,
    series_start = None,
    series_step = 1,
    ext:str = 'nii.gz',
    slice_name:str = 'DCM', 
    sequence_fmt:str = '04d',
    start_index = 1,
    override_voxel_spacing = False,
    override_orientation = False,
    voxel_min = None,
    voxel_max = None,
    nan_default = 0,
    scaling = None,
    offset = None,
    flip_x = False,
    flip_y = False,
    flip_z = False):
    '''
    '''
    #Load the dicom template, we only need to do this once,
    #then reuse for all nifti images
    if type(dicom_im) == str:
        dicom_im = pydicom.dcmread(dicom_im)

    #Get list of NIFTI images
    nii_list = \
        sorted(glob.glob(os.path.join(nifti_dir, '*' + ext), recursive=False))

    print(f'Found {len(nii_list)} images to convert in {nifti_dir}')

    #Get series start if not set
    if series_start is None:
        series_start = dicom_im.SeriesNumber * 100 + 1

    #If time-series, create single dicom dir
    if time_series:
        number_of_temporal_positions = len(nii_list)
        series_name = os.path.basename(nifti_dir)
        series_number = series_start
        dcm_dir = os.path.join(os.path.dirname(nifti_dir), 'DICOM', series_name)
    else:
        number_of_temporal_positions = 1
        temporal_position = 1

    #Loop through images
    for i_im, nii_im in enumerate(nii_list):

        if time_series:
            #Set temporal position
            temporal_position = i_im + 1

        else:
            #Create a new dir for each image
            series_name = os.path.basename(nii_im).split(".")[0]
            dcm_dir = os.path.join(nifti_dir, 'DICOM', series_name)
            
            #Generate a new series index
            series_number = series_start + i_im * series_step

        #Convert image
        converted_slices = convert_nifti(nii_im, dicom_im, dcm_dir,
            series_name = series_name, 
            series_number = series_number,
            slice_name = slice_name, 
            sequence_fmt = sequence_fmt,
            start_index = start_index,
            number_of_temporal_positions = number_of_temporal_positions,
            temporal_position = temporal_position,
            override_voxel_spacing = override_voxel_spacing,
            override_orientation = override_orientation,
            voxel_min = voxel_min,
            voxel_max = voxel_max,
            nan_default = nan_default,
            scaling = scaling,
            offset = offset,
            flip_x = flip_x,
            flip_y = flip_y,
            flip_z = flip_z)

        if time_series:
            #Increment the start_index by the number of slices converted
            start_index += converted_slices

    return series_number
    
