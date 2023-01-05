'''

Hat tip to: https:#pycad.co/nifti2dicom/ for an overview of the
basic approach
'''
#%%
import os
import glob
import warnings
from datetime import datetime

import numpy as np
import nibabel as nib

import pydicom
from pydicom import uid
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence

default_attributes_from_base = [
    'PatientName',
    'PatientID',
    'PatientBirthDate',
    'PatientSex',
    'PatientWeight',
    'BodyPartExamined',
    'StudyDate',
    'StudyTime',
    'MRAcquisitionType',
    'PatientPosition',
    'StudyInstanceUID',
    'StudyID',
    'FrameOfReferenceUID',
    'PositionReferenceIndicator'
]

#%%
def convert_nifti(
    nifti_im:str, 
    dicom_base:str, 
    output_dir:str,
    series_name:str, 
    series_number:str,
    slice_name:str = 'DCM', 
    sequence_fmt:str = '04d',
    start_index:int = 1,
    acquisition_number:int = None,
    number_of_temporal_positions:int = None,
    temporal_position:int = None,
    override_voxel_spacing:bool = False,
    override_orientation:bool = False,
    voxel_min:float = None,
    voxel_max:float = None,
    nan_default:float = 0,
    scaling:float = None,
    offset:float = None,
    flip_x:bool = False,
    flip_y:bool = False,
    flip_z:bool = False,
    uid_org_root:str = None,
    creator_uid:str = None,
    creator_uid_names:list = None,
    attributes_from_base:list = None,
    manufacturer:str = '',
    analysis_type:str = '',
    modality:str = 'qMRI analysis',
    map_units:str = 'normalized',
    image_type:list = [],
    software_versions:list = None,
    LUT_explanation:str = None,
    coding_scheme_designator:str = 'UCUM'):
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

    #Set creator UID
    if creator_uid is None:
        creator_uid = uid.generate_uid(uid_org_root, entropy_srcs = creator_uid_names)

    #Create new DICOM dataset and fill some generic attributes
    dicom_im = Dataset()
    dicom_im.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
    dicom_im.Modality = modality
    dicom_im.ImageType = image_type
    dicom_im.InstanceCreatorUID = creator_uid
    dicom_im.RequestedProcedureDescription = analysis_type
    dicom_im.Manufacturer = manufacturer
    dicom_im.is_implicit_VR = False
    dicom_im.is_little_endian = True
    if software_versions is not None:
        dicom_im.SoftwareVersions = software_versions

    #Set attributes from the base image
    if type(dicom_base) == str:
        dicom_base = pydicom.dcmread(dicom_base)
    if attributes_from_base is None:
        attributes_from_base = default_attributes_from_base
    for attribute in attributes_from_base:
        set_attribute_from_base(dicom_im, dicom_base, attribute)

    #Set series name and number
    dicom_im.ProtocolName = series_name
    dicom_im.SeriesDescription = series_name
    dicom_im.SeriesNumber = series_number
    if acquisition_number is None:
        acquisition_number = series_number
    dicom_im.AcquisitionNumber = acquisition_number

    #Set time and date fields
    time = datetime.utcnow()
    datestr_day = time.strftime('%Y%m%d')
    datestr_time = time.strftime('%H%M%S.%f')[:-3]
    dicom_im.InstanceCreationDate = datestr_day
    dicom_im.InstanceCreationTime = datestr_time
    dicom_im.SeriesDate = datestr_day
    dicom_im.AcquisitionDate = datestr_day
    dicom_im.ContentDate = datestr_day
    dicom_im.SeriesTime = datestr_time
    dicom_im.AcquisitionTime = datestr_time
    dicom_im.ContentTime = datestr_time

    #Set required MR fields that make more sense to be empty/0 for an output map
    dicom_im.ReferringPhysicianName = ''
    dicom_im.AccessionNumber = ''
    dicom_im.ScanningSequence = 'RM'
    dicom_im.SequenceVariant = 'NONE'
    dicom_im.ScanOptions = ''
    dicom_im.RepetitionTime = 0
    dicom_im.EchoTime = 0
    dicom_im.EchoTrainLength = 0

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
    else:
        for attribute in ['SpacingBetweenSlices', 'SliceThickness', 'PixelSpacing']:
            set_attribute_from_base(dicom_im, dicom_base, attribute)
    
    #Set axes orientation
    if override_orientation or flip_x or flip_y:
        dicom_im.ImageOrientationPatient = orientation_from_sform(sform, flip_x, flip_y)
    else:
        set_attribute_from_base(dicom_im, dicom_base, 'ImageOrientationPatient')

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

    print(f'Grayscale offset = {offset}, scaling = {scaling}')
    
    dicom_im.RescaleIntercept = truncate_num(offset)
    dicom_im.RescaleSlope = truncate_num(scaling)
    dicom_im.RescaleType = map_units

    #Set real world mapping info
    if LUT_explanation is not None:
        real_world_value_mapping_sequence = Sequence()
        dicom_im.RealWorldValueMappingSequence = real_world_value_mapping_sequence

        # Real World Value Mapping Sequence: Real World Value Mapping 1
        real_world_value_mapping1 = Dataset()
        real_world_value_mapping1.LUTExplanation = LUT_explanation

        # Measurement Units Code Sequence
        measurement_units_code_sequence = Sequence()
        real_world_value_mapping1.MeasurementUnitsCodeSequence = measurement_units_code_sequence

        # Measurement Units Code Sequence: Measurement Units Code 1
        measurement_units_code1 = Dataset()
        measurement_units_code1.CodeValue = map_units
        measurement_units_code1.CodingSchemeDesignator = coding_scheme_designator
        measurement_units_code1.CodeMeaning = map_units
        measurement_units_code_sequence.append(measurement_units_code1)

        real_world_value_mapping1.LUTLabel = manufacturer
        real_world_value_mapping1.RealWorldValueLastValueMapped = 4095
        real_world_value_mapping1.RealWorldValueFirstValueMapped = 0
        real_world_value_mapping1.RealWorldValueIntercept = offset
        real_world_value_mapping1.RealWorldValueSlope = scaling
        real_world_value_mapping_sequence.append(real_world_value_mapping1)

    #Set pixel storage information
    dicom_im.PhotometricInterpretation = "MONOCHROME2"
    dicom_im.PresentationLUTShape = 'IDENTITY'
    dicom_im.SamplesPerPixel = 1
    dicom_im.BitsStored = 16
    dicom_im.BitsAllocated = 16
    dicom_im.HighBit = 15
    dicom_im.PixelRepresentation = 0

    #Set series UID
    dicom_im.SeriesInstanceUID = uid.generate_uid(uid_org_root)

    #Set file meta data
    # File meta info data elements
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = uid.UID('1.2.840.10008.5.1.4.1.1.4')
    file_meta.TransferSyntaxUID = uid.ExplicitVRLittleEndian
    dicom_im.file_meta = file_meta

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
            instance_uid = uid.generate_uid(uid_org_root)
            dicom_im.SOPInstanceUID = instance_uid
            dicom_im.file_meta.MediaStorageSOPInstanceUID = instance_uid
            write_slice(
                dicom_im, dicom_name, slice, img[:,:,slice,time], 
                origin, slice_axis, idx)
            converted_slices += 1

    print(f'Successfully converted {converted_slices} slices '
        f'for series {series_name} ({series_number})')
    return converted_slices

#-----------------------------------------------------------------
def truncate_num(num, max_len = 16):
    '''
    '''
    return f'{num:.8e}'

#-----------------------------------------------------------------
def set_attribute_from_base(dicom_image, dicom_base, attribute):
    '''
    '''
    try:
        dicom_image[attribute] = dicom_base[attribute]

    except:
        warnings.warn(
            f'Cannot set {attribute} in converted DICOM as it is not set in base image')

#-----------------------------------------------------------------
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

#------------------------------------------------------------------
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

    axes_orientation_list = [truncate_num(o) for o in axes_orientation.tolist()]
    return axes_orientation_list

def get_slice_origin(origin, slice_num, slice_axis):
    '''

    '''
    slice_origin = origin + slice_axis*slice_num
    slice_origin_list = [truncate_num(o) for o in slice_origin.tolist()]
    return slice_origin_list

def set_pixel_array(slice_array, dicom_im):
    '''
    '''
    scaled_array = (slice_array - float(dicom_im.RescaleIntercept)) / float(dicom_im.RescaleSlope)
    dicom_im.PixelData = scaled_array.astype('uint16').tobytes()
    if 'LargestImagePixelValue' in dicom_im:
        try: 
            dicom_im['LargestImagePixelValue'].VR = 'US'
            dicom_im.LargestImagePixelValue = np.max(scaled_array.astype('uint16'))
        except:
            pass
    
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
    dicom_im.save_as(dicom_name, write_like_original=False)
    #print('Saved ', dicom_name)

def convert_nifti_dir(
    nifti_dir:str, 
    dicom_base:str, 
    time_series:bool = False,
    series_start:int = None,
    series_step:int = 1,
    ext:str = 'nii.gz',
    slice_name:str = 'DCM', 
    sequence_fmt:str = '04d',
    start_index:int = 1,
    override_voxel_spacing:bool = False,
    override_orientation:bool = False,
    voxel_min:float = None,
    voxel_max:float = None,
    nan_default:float = 0,
    scaling:float = None,
    offset:float = None,
    flip_x:bool = False,
    flip_y:bool = False,
    flip_z:bool = False,
    uid_org_root:str = None,
    creator_uid:str = None,
    creator_uid_names:list = None,
    attributes_from_base:list = None,
    manufacturer:str = '',
    analysis_type:str = '',
    modality:str = 'qMRI analysis',
    map_units:str = 'normalized',
    image_type:list = [],
    software_versions:list = None,
    LUT_explanation:str = None,
    coding_scheme_designator:str = 'UCUM'):
    '''
    '''
    #Load the dicom template, we only need to do this once,
    #then reuse for all nifti images
    if type(dicom_base) == str:
        dicom_base = pydicom.dcmread(dicom_base)

    #Get list of NIFTI images
    nii_list = \
        sorted(glob.glob(os.path.join(nifti_dir, '*' + ext), recursive=False))

    print(f'Found {len(nii_list)} images to convert in {nifti_dir}')

    #Get series start if not set
    if series_start is None:
        series_start = dicom_base.SeriesNumber * 100 + 1

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
        converted_slices = convert_nifti(
            nii_im, 
            dicom_base, 
            dcm_dir,
            uid_org_root = uid_org_root,
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
            flip_z = flip_z,
            creator_uid = creator_uid,
            creator_uid_names = creator_uid_names,
            attributes_from_base = attributes_from_base,
            manufacturer = manufacturer,
            analysis_type = analysis_type,
            modality = modality,
            map_units = map_units,
            image_type = image_type,
            software_versions = software_versions,
            LUT_explanation = LUT_explanation,
            coding_scheme_designator = coding_scheme_designator)

        if time_series:
            #Increment the start_index by the number of slices converted
            start_index += converted_slices

    return series_number
    
