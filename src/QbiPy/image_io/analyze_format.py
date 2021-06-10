'''
Functions for reading and writing Analyze 7.5 format images

These are often available in external packages, but it was helpful
to have our own implementation that used native code with few
external dependencies
'''
import numpy as np
import struct
import os
from types import SimpleNamespace
import warnings
from datetime import datetime

from QbiPy.image_io import xtr_files

#Define matching format strings in analyze, numpy and struct
#and methods for looking up one given the other
format_strings = [
    ('DT_BINARY', np.bool, 'B'),
    ('DT_UNSIGNED_CHAR', np.uint8, 'B'),
    ('DT_SIGNED_SHORT', np.int16, 'h'),
    ('DT_SIGNED_INT', np.int32, 'i'),
    ('DT_FLOAT', np.float32, 'f'),
    ('DT_DOUBLE', np.float64, 'd'),
]

def format_str_analyze_to_struct(ana_str):
    '''
    Convert analyze format string to struct format string
    
    Parameters:
        ana_str : str
            Analyze data type string
    Returns:
        struct_str : str
            Format string used by python's struct class
    '''
    return lookup_format(0, 2, ana_str)

def format_str_analyze_to_numpy(ana_str):
    '''
    Convert analyze format string to struct format string
    
    Parameters:
        ana_str : str
            Analyze data type string
    Returns:
        np_str : str
            Dtype used by numpy
    '''
    return lookup_format(0, 1, ana_str)

def format_str_struct_to_analyze(struct_str):
    '''
    Convert analyze format string to struct format string
    
    Parameters:
    struct_str : str
            Format string used by python's struct class
    Returns:
        ana_str : str
            Analyze data type string
    '''
    return lookup_format(2, 0, struct_str)

def format_str_numpy_to_analyze(np_str):
    '''
    Convert analyze format string to struct format string
    
    Parameters:
        np_str : str
            Dtype used by numpy
    Returns:
        ana_str : str
            Analyze data type string
    '''
    return lookup_format(1, 0, np_str)

def format_str_struct_to_numpy(struct_str):
    '''
    Convert analyze format string to struct format string
    
    Parameters:
        struct_str : str
            Format string used by python's struct class
    Returns:
        np_str : str
            Dtype used by numpy
    '''
    return lookup_format(2, 1, struct_str)

def format_str_numpy_to_struct(np_str):
    '''
    Convert analyze format string to struct format string
    
    Parameters:
        np_str : str
            Dtype used by numpy
    Returns:
        struct_str : str
            Format string used by python's struct class
    '''
    return lookup_format(1, 2, np_str)

def lookup_format(type_in, type_out, val):
    types = ['Analyze', 'numpy', 'struct']
    try:
        idx = [f[type_in] for f in format_strings].index(val)
    except:
        raise ValueError(f'Data type {val} not recognised as {types[type_in]} format')
    return format_strings[idx][type_out]
 

##------------------------------------------------------------------------------
##------------------------------------------------------------------------------
def read_analyze(filename: str=None,
    output_type:np.dtype=np.float64, scale:float = 1.0, 
    flip_y: bool = True, flip_x: bool = False,
    swap_axes: bool = True):
    '''
    Read image file and header data of Mayo Analyze 7.5 data set.

    Wrapper for read_analyze_img and read_analyze_hdr

    Parameters:
        filename : str default None
            filename to analyze image/header pair to read. Should be extension
            free, or either of the filename.hdr or filename.img pair. Must be
            set if hdr_data not set
        output_type : np.dtype, default np.float64,
            Numpy datatype output image array converted to. Use None to leave unchanged
            from type specified in header data
        scale : float default 1.0,
            Value by which output array is scaled by, if != 1.0, ouput will be divided
            (NOT multplied) by scale
        flip_y : bool default True,
            If true, flips the output image about array axis 0 (vertical flip)
        flip_x : bool default False,
            If true, flips the output image about array axis 1 (horizontal flip)

    Returns:
        img : np.array
            Numpy array of image data in data type specified by output data. Will
            be 2, 3 or 4D (n_y, n_x, n_z = 1, n_v = 1)
        hdr_data : SimpleNamespace
            analyze format header data structure.
    '''
    hdr = read_analyze_hdr(filename)
    img = read_analyze_img(hdr_data=hdr,
        output_type=output_type, scale=scale, 
        flip_y=flip_y, flip_x=flip_x, swap_axes=swap_axes)
    return img, hdr

##------------------------------------------------------------------------------
##------------------------------------------------------------------------------
def read_analyze_img(filename: str=None, hdr_data=None,
    output_type:np.dtype=np.float64, scale:float = 1.0, 
    flip_y: bool = True, flip_x: bool = False,
    swap_axes: bool = True)->np.array:
    '''
    Read image file of Mayo Analyze 7.5 data set. 
    
    Based on Matlab function analyze75read. Reads image data from the IMG file of
    an Analyze 7.5 format data set (a pair of FILENAME.HDR and FILENAME.IMG
    files).  For single-frame images, I is an M-by-N array where M is the
    number of rows and N is the number of columns. For multi-dimensional
    images, I can be an M-by-N-by-O or M-by-N-by-O-by-P array where M is
    the number of rows, N is the number of columns, O is the number of
    slices per volume and P is the number of volumes or time points. The
    data type of I is consistent with the image data type specified in 
    the metadata obtained from the header file. 
    
    Parameters:
        filename : str default None
            filename to analyze image/header pair to read. Should be extension
            free, or either of the filename.hdr or filename.img pair. Must be
            set if hdr_data not set
        hdr_data : SimpleNamespace, default None,
            analyze format header data structure. If None will be read from header
            file, in which case filename must not be None
        output_type : np.dtype, default np.float64,
            Numpy datatype output image array converted to. Use None to leave unchanged
            from type specified in header data
        scale : float default 1.0,
            Value by which output array is scaled by, if != 1.0, ouput will be divided
            (NOT multplied) by scale
        flip_y : bool default True,
            If true, flips the output image about array axis 0 (vertical flip)
        flip_x : bool default False,
            If true, flips the output image about array axis 1 (horizontal flip)

    Returns:
        img : np.array
            Numpy array of image data in data type specified by output data. Will
            be 2, 3 or 4D (n_y, n_x, n_z = 1, n_v = 1)

    Class support
    -------------
    I can be logical, uint8, int16, int32, single, or double. Complex and
    RGB data types are not supported.
    
    Based on Matlab function ANALYZE75READ, copyright 2005-2011 The MathWorks, Inc.
    rewritten for python by Mike Berks
    '''     
    #Get image and header filename
    if filename is not None:
        hdr_filename = os.path.splitext(filename)[0] + '.hdr'

    #Check if given metadata, if not, need to load header
    if hdr_data is None:
        # Will crash if no filename, that's desired behaviour
        hdr_data = read_analyze_hdr(hdr_filename)

    img_filename = os.path.splitext(hdr_data.Filename)[0]+ '.img'
        
    #Read binary contents of header file into buffer
    with open(img_filename, 'br') as f:
        buffer = f.read()
    
    # Unpack data from the bytes buffer into a numpy array of correct
    # shape and data format, based on info in image header 
    try:
        n_x = hdr_data.Dimensions[0]
        n_y = hdr_data.Dimensions[1]
        n_z = hdr_data.Dimensions[2]
        n_v = hdr_data.Dimensions[3] 
    except:       
        raise ValueError('Incorrect header metadata structure, missing dimensions')
    
    # Obtain precision_string for reading in the data in the right format.  
    precision  = format_str_analyze_to_struct(hdr_data.ImgDataType)

    # Get byte-order
    if hdr_data.ByteOrder == 'ieee-le':
        endian = '<'
    elif hdr_data.ByteOrder == 'ieee-be':
        endian = '>'
    else:
        pass
        
    #We unpack differently for sparse and full, if sparse
    #not in header (it won't be unless image written by us) assume full
    try: 
        sparse = hdr_data.Sparse
    except:
        sparse = False 

    if sparse:
        #Create correct shape container and put data into
        #indexed elements
        img = np.zeros((n_x,n_y,n_z,n_v), 
            dtype=format_str_analyze_to_numpy(hdr_data.ImgDataType))

        #Need to work out how many elements we have
        element_sz = struct.calcsize(precision) + struct.calcsize("i")
        n_idx = hdr_data.ImgFileSize / element_sz

        #Size of file should divide by element size into an into
        #throw error if not
        if n_idx % 1:
            raise ValueError(f'Image filesize ({hdr_data.ImgFileSize}) not divisble by '
                f'element size ({element_sz})')
        n_idx = int(n_idx)

        #Can have no non-zero elements, in which case return
        if n_idx:
            #Unpack the data values then indices
            data_buf_sz = struct.calcsize(str(n_idx)+precision)
            vals = np.array(
                struct.unpack_from(endian+str(n_idx)+precision, buffer, 0))
            idx = np.array(
                struct.unpack_from(endian+str(n_idx)+'i', buffer, data_buf_sz)
            )
            data_size = (n_x,n_y,n_z,n_v)

            idx_C = np.ravel_multi_index(
                np.unravel_index(idx, data_size, order='F'),
                data_size, order='C'
            )

            np.put(img, idx_C, vals)

    else:
        # Compute number of elements to be read.
        count = np.prod(hdr_data.Dimensions)

        # Call struct unpack with the correct precision string
        img_data = struct.unpack_from(
                endian+str(count)+precision, buffer, 0)
            
        # Read data in fortrat order, then reshape to dimensions from hdr
        img = np.reshape(img_data, (n_x, n_y, n_z, n_v), order='F')

    #Convert to 3D if num vols is 1
    if n_v == 1:
        img = np.squeeze(img, axis=3)

    #Convert to 2D if num slices is 1 (may still be multiple vols
    # which will now be on 3rd dimension)
    if n_z == 1:
        img = np.squeeze(img, axis=2)

    #For binary data convert to bool
    if hdr_data.ImgDataType == 'DT_BINARY':
        img = img.astype(bool, copy=False)

    #images also get loaded in with the y-axis (after swapping!) reversed
    #(eg upside down). Use the flip_y flag to correct this
    if swap_axes:
        img = np.swapaxes(img, 0, 1)
    if flip_y:
        img = np.flip(img, 0)
    if flip_x:
        img = np.flip(img, 1)
    

    #Now cast to out datatype (leave unchanged if output_type is None)
    if output_type is not None:
        img = img.astype(output_type, copy=False)

    #Finally, the QBI lab often used to save scaled output, we may want
    #to scale that during loading. Note the out data type should be a
    #float of some sort for this.
    if scale != 1.0:
        try:
            img /= scale  
        except:
            raise ValueError(f'Ouput datatype {img.dtype} cannot be scaled.',
            'Use float format.')

    return img

##-----------------------------------------------------------------------------
##-----------------------------------------------------------------------------
def read_analyze_hdr(filename, byte_order:str = None):
    '''
    Read metadata from header file of Mayo Analyze 7.5 data set. Based
    on Matlab function analyze75info. Reads the HDR file of an Analyze 7.5
    format data set (a pair of FILENAME.HDR and FILENAME.IMG files) and
    returns a structure METADATA whose fields contain information about the
    data set. FILENAME is a string that specifies the name of the Analyze
    file pair.

    Parameters:
        filename : str default None
            filename of analyze header to read. Should be extension
            free, or either of the filename.hdr or filename.img pair.
        byte_order : str default None, options ['ieee-le', 'ieee-be']
            Specify whether to use little or big endian byte order to read
            binary data. If None (recommended), tries both and uses whichever
            returns a valid header size field

    Returns:
        hdr_data : SimpleNamespace,
            Namespace object with image metadata. The  following is a 
            partial list of fields in the hdr_data structure:
    
        Filename          A string containing the name of the file
    
        FileModDate       A string containing the modification date of
                            the file
    
        HdrFileSize       An integer indicating the size of the HDR file in
                            bytes
    
        ImgFileSize       An integer indicating the size of the IMG file in
                            bytes
    
        Format            A string containing the file format. This value is
                            set to 'Analyze' for valid Analyze data sets
    
        FormatVersion     A string or number specifying the file format
                            version
    
        Width             An integer indicating the width of the image
                            in pixels
    
        Height            An integer indicating the height of the image
                            in pixels
    
        BitDepth          An integer indicating the number of bits per
                            pixel
    
        ColorType         A string indicating the type of image either
                            'truecolor' for a truecolor (RGB) image, or
                            'grayscale' for a grayscale image,
    
        ByteOrder         A string containing the byte-ordering used to
                            successfully read in the HDR file.
    
        HdrDataType       Data type of the HDR file.
    
        DatabaseName      Name of the image database.
    
        Extents           An integer which is a required field in the header
                            file. This value should be 16384.
    
        SessionError      An integer indicating session error number.
    
        Regular           A character indicating whether or not all images
                            and volumes are of the same size. A value '1'
                            indicates that the data is regular while '0'
                            indicates the data is not regular.
    
        Dimensions        A vector providing information on the image
                            dimensions. The vector is of the form
                                [X Y Z T]
                            X gives the X dimension of the image, i.e. the
                            number of pixels in an image row.
                            Y gives the Y dimension of the image, i.e. the
                            number of pixels in an image column.
                            Z gives the volume Z dimension, i.e. the number of
                            slices in a volume.
                            T indicates the time points, i.e. the number of
                            volumes in the dataset.
                            Dimensions vector only returns non-zero entries.
    
        VoxelUnits        Spatial units of measure for a voxel.
    
        CalibrationUnits  Name of the calibration unit.
    
        ImgDataType       Data type of the IMG file.
    
        PixelDimensions   A vector providing information on the pixel
                            dimensions. PixelDimensions is parallel to the
                            Dimensions field, providing real world
                            measurements in mm. The vector is of the form
                                [Xp Yp Zp Tp]
                            Xp provides the voxel width in mm.
                            Yp provides the voxel height in mm.
                            Zp provides the slice thickness in mm.
                            Tp provides the time points in ms.
                            PixelDimensions vector only returns non-zero
                            entries.
    
        VoxelOffset       The byte offset in the image file at which voxels
                            start. This value may be negative to specify that
                            the absolute value is applied for every image in
                            the file.
    
        CalibrationMax    Maximum Calibration value.
    
        CalibrationMin    Minimum Calibration value.
    
        GlobalMax         Global Maximum. The maximum pixel values for the
                            entire dataset.
    
        GlobalMin         Global Minimum. The minimum pixel values for the
                            entire dataset.
    
        Descriptor        Data description.
    
        Orientation       Slice orientation for the dataset.
    
    Use struct.unpack to format byte data, with format strings
        c = char 1
        b = signed char (int8) 1
        B = unsigned char (uint8) 1
        h = short (int16) 2
        i = int (int32) 4
        l = long (int64) 8
        f = float (single) 4
        d = double 8

    Based on analyze75info copyright 2005-2017 The MathWorks, Inc.
    Rewritten for python by Mike Berks
    '''
    # Check byte order value
    user_supplied = False
    if byte_order is not None:
        user_supplied = True
        
        if byte_order not in ['ieee-le', 'ieee-be']:
            raise ValueError(f'{byte_order} not recognised, must be ieee-le or ieee-be')  
    
    #Add hdr extension (in case .img file given as input)
    filename = os.path.splitext(filename)[0] + ".hdr"  
    
    # Remember if we have already warned the user about truncated header file.
    already_warned = False
    
    #Read binary contents of header file into buffer
    with open(filename, 'br') as f:
        buffer = f.read()

    #--------------------------------------------------------------------------------
    #Set up metadata
    hdr_data = SimpleNamespace()
    hdr_data.Filename = filename
    s = os.stat(filename)
    hdr_data.FileModDate = datetime.fromtimestamp(s.st_mtime).strftime(
        '%d-%b-%Y %H:%M:%S')
        
    # Image File Size will be obtained below after constructing the
    # Image filename from the header filename.
    hdr_data.Format = 'Analyze'
    hdr_data.FormatVersion = '7.5'
    hdr_data.Width = []
    hdr_data.Height = []
    hdr_data.BitDepth = []
    hdr_data.ColorType = 'unknown'
    
    # Construct Image filename using Filename obtained from
    # Metadata struct.
    img_filename = os.path.splitext(filename)[0] + '.img'
                
    # Obtain Image file size.
    try:
        s = os.stat(img_filename)
        hdr_data.ImgFileSize = s.st_size
    except OSError:
        hdr_data.ImgFileSize = []
        warnings.warn(f'{filename} missing image part {img_filename}')

    #-------------------------------------------------------------------------
    #Check whether buffer is little or big-endian
    #Read headerSize with both little and big endian
    headerSize_le = struct.unpack_from('<i', buffer, 0)[0]
    headerSize_be = struct.unpack_from('>i', buffer, 0)[0]
    
    # Possible exted header size
    min_size = 348 
    max_size = 2000
    
    # headerSize should be within the extedRange. Use that to check
    # if incorrect ByteOrder was used to open and read the file.
    if min_size <= headerSize_le <= max_size:
        endian = '<'
        hdr_data.HdrFileSize = headerSize_le
        hdr_data.ByteOrder = 'ieee-le'
        
    elif min_size <= headerSize_be <= max_size:
        endian = '>'
        hdr_data.HdrFileSize = headerSize_be
        hdr_data.ByteOrder = 'ieee-be'
    
    else:        
        # We have tried reading the file with both ByteOrder
        # formats. Generate error
        raise RuntimeError('Analyze header size wrong, is this really an hdr file')
                        
    # Generate warning if incorrect ByteOrder was provided by user.
    if user_supplied and byte_order != hdr_data.ByteOrder:
        warnings.warn(f'User supplied {byte_order} was incorrect, using {hdr_data.ByteOrder}')
    #------------------------------------------------------------------------------------
    
    #Initialise buffer offset to 4 (we've already read header size)
    offset = 4
    #------------------------------------------------------------------------
    ###
    ### Define nested helper function unpackVerified
    ###
    def unpackVerified(count, precision):
        # This function reads the specified number of bytes using unpack and
        # checks for premature EOF. In that case, a warning is generated the
        # first time this is encountered in the file.
        nonlocal offset, already_warned
        try:
            out = struct.unpack_from(
                    endian+str(count)+precision, buffer, offset)
            offset += count*struct.calcsize(precision)
            
            #Assume we want join chars together into a string
            if precision == 's':
                try:
                    out = out[0].decode('utf-8').strip().strip('\x00')
                except UnicodeDecodeError:
                    warnings.warn('Unable to decode characters from byte stream')
                    out = []

            #And that for other datatypes, if we only wanted one, get rid of the
            #outer tuple
            elif count == 1:
                out = out[0]

        except struct.error:
            out = []
            if not already_warned:
                warnings.warn('Truncated header file')
                already_warned = True
        
        return out
    #------------------------------------------------------------------------

    # Read the HeaderKey information from HDR file.
    # Read all information in the HeaderKey structure.
    hdr_data.HdrDataType  = unpackVerified(10, 's')
    hdr_data.DatabaseName = unpackVerified(18, 's')
    hdr_data.Extents      = unpackVerified(1,  'i')
    hdr_data.SessionError = unpackVerified(1,  'h')
    hdr_data.Regular      = unpackVerified(1,  's')  == 'r'

    #Advance one position for an unused character.
    offset += 1 
    
    # Read the ImgDimension information from HDR file.
    dims = unpackVerified(8, 'h')
    
    # Return useful dimension information.
    hdr_data.Dimensions = [d for d in dims[1:] if d]
    hdr_data.Width = hdr_data.Dimensions[0]
    hdr_data.Height= hdr_data.Dimensions[1]
    hdr_data.VoxelUnits  = unpackVerified(4, 's')
    hdr_data.CalibrationUnits = unpackVerified(8, 's')

    # Advance 2 positions for an unused field.
    offset += 2

    # Parse image data type
    ImgDataType = unpackVerified(1, 'h')

    #Check if sparse
    #Our sparse format adds 5 to the supported data types,
    #(this means we can sparsify masks, if we added 1, then
    # a sparse BINARY type would have code 2 == UNSIGNED_CHAR)
    hdr_data.Sparse = False
    if ImgDataType == 6 or (ImgDataType % 2):
        hdr_data.Sparse = True
        ImgDataType -= 5

    if ImgDataType == 0:
        hdr_data.ImgDataType = 'DT_UNKNOWN'
    elif ImgDataType == 1:
        hdr_data.ImgDataType = 'DT_BINARY'
        hdr_data.ColorType = 'grayscale'
    elif ImgDataType == 2:
        hdr_data.ImgDataType = 'DT_UNSIGNED_CHAR'
        hdr_data.ColorType = 'grayscale'
    elif ImgDataType == 4:
        hdr_data.ImgDataType = 'DT_SIGNED_SHORT'
        hdr_data.ColorType = 'grayscale'
    elif ImgDataType == 8:
        hdr_data.ImgDataType = 'DT_SIGNED_INT'
        hdr_data.ColorType = 'grayscale'
    elif ImgDataType == 16:
        hdr_data.ImgDataType = 'DT_FLOAT'
        hdr_data.ColorType = 'grayscale'
    elif ImgDataType == 32:
        hdr_data.ImgDataType = 'DT_COMPLEX'
        hdr_data.ColorType = 'grayscale'
    elif ImgDataType == 64:
        hdr_data.ImgDataType = 'DT_DOUBLE'
        hdr_data.ColorType = 'grayscale'
    elif ImgDataType == 128:
        hdr_data.ImgDataType = 'DT_RGB'
        hdr_data.ColorType = 'truecolor'
    elif ImgDataType == 255:
        hdr_data.ImgDataType = 'DT_ALL'

    #Read bit depth
    hdr_data.BitDepth   = unpackVerified(1, 'h')

    #Advance 2 positions for an unused field.
    offset += 2
    pix_dims   = unpackVerified(8, 'f')
    hdr_data.PixelDimensions = [p for p in pix_dims if p]
    hdr_data.VoxelOffset     = unpackVerified(1, 'f')

    #Advance 12 positions for an unused field.
    offset += 12
    hdr_data.CalibrationMax = unpackVerified(1, 'f')
    hdr_data.CalibrationMin = unpackVerified(1, 'f')
    hdr_data.Compressed     = unpackVerified(1, 'f')
    hdr_data.Verified       = unpackVerified(1, 'f')
    hdr_data.GlobalMax      = unpackVerified(1, 'i')
    hdr_data.GlobalMin      = unpackVerified(1, 'i') 
    
    # Read the DataHistory information from HDR file.
    hdr_data.Descriptor   = unpackVerified(80, 's')
    hdr_data.AuxFile      = unpackVerified(24, 's')
    Orientation           = unpackVerified(1, 'B')
    if Orientation == 0:
        hdr_data.Orientation = 'Transverse unflipped'
    elif Orientation == 1:
        hdr_data.Orientation = 'Coronal unflipped'
    elif Orientation == 2:
        hdr_data.Orientation = 'Sagittal unflipped'
    elif Orientation == 3:
        hdr_data.Orientation = 'Transverse flipped'
    elif Orientation == 4:
        hdr_data.Orientation = 'Coronal flipped'
    elif Orientation == 5:
        hdr_data.Orientation = 'Sagittal flipped'
    else:
        hdr_data.Orientation = 'Orientation unavailable'

    # Various scanner generated fields
    hdr_data.Originator   = unpackVerified(10, 's')
    hdr_data.Generated    = unpackVerified(10, 's')
    hdr_data.Scannumber   = unpackVerified(10, 's')
    hdr_data.PatientID    = unpackVerified(10, 's')
    hdr_data.ExposureDate = unpackVerified(10, 's')
    hdr_data.ExposureTime = unpackVerified(10, 's')

    #Advance 3 positions for an unused field.
    offset += 3
    hdr_data.Views          = unpackVerified(1, 'i')
    hdr_data.VolumesAdded   = unpackVerified(1, 'i')
    hdr_data.StartField     = unpackVerified(1, 'i')
    hdr_data.FieldSkip      = unpackVerified(1, 'i')
    hdr_data.OMax           = unpackVerified(1, 'i')
    hdr_data.OMin           = unpackVerified(1, 'i')
    hdr_data.SMax           = unpackVerified(1, 'i')
    hdr_data.SMin           = unpackVerified(1, 'i') 
    
    #We're done, return header data
    return hdr_data

##-----------------------------------------------------------------------------
##-----------------------------------------------------------------------------
def write_analyze(img_data: np.array, filename: str,
    scale:float = 1.0, swap_axes:bool = True,
    flip_x: bool = False, flip_y: bool = True,
    voxel_size=[1,1,1], dtype=None, sparse=False):
    '''
    Wrapper to write_analyze_img for writing array to analyze 75 format image on disk,
    with some extra options for scaling and transforming data

    Parameters:
        img_data: np.array, 
            Input image array to write
        filename: str,
            Path of file to write to. If filename extension is not hdr or img
            .img will be appended

        scale:float default 1.0, 
            Value by which data is multiplied before writing.      
        swap_axes:bool = True,
            Swap x and y axes before writing binary data
        flip_y : bool default True,
            If true, flips the output image about array axis 0 (vertical flip)
        flip_x : bool default False,
            If true, flips the output image about array axis 1 (horizontal flip)
        voxel_size : tuple/list default [1,1,1]

        dtype=None
    '''
    
    #Check filename extension, 
    # if it's .img or .hdr leave alone
    # if it's anything else, append img
    file_ext = os.path.splitext(filename)[1]
    if file_ext != '.hdr' and file_ext != '.img':
        filename = filename + ".img"

    #Finally, the QBI lab often used to save scaled output, we may want
    #to scale that during loading
    if scale != 1.0:
      img_data *= scale 

    #images also get loaded in with the y-axis (after swapping!) reversed
    #(eg upside down). Use the flip_y flag to correct this
    if flip_y:
        img_data = np.flip(img_data, 0)

    if flip_x:
        img_data = np.flip(img_data, 1)

    if swap_axes:
        img_data = np.swapaxes(img_data, 0, 1)

    #Call write analyze
    write_analyze_img(img_data, filename, 
        voxel_size=voxel_size, dtype=dtype, sparse=sparse)

    #Undo the scaling - not the most efficient, but saves taking a deep
    #copy
    if scale != 1.0:
      img_data /= scale
    
###-----------------------------------------------------------------
###-----------------------------------------------------------------
def write_analyze_img(data, filename, data_size=None, voxel_size=[1,1,1], 
    dtype=None, index=None, default_scaling='pos', sparse=False):
    '''     
    Write array to analyze 75 format image on disk. Also writes matching
    header data file

    Parameters:
        data : np.array, 
            Data can be (3d,4d) array or vector. If data is a 
            vector then data_size must be included. The data can also
            be indexed, meaning that each point corresponds to an
            indexed point in the 3d volume. In this case both the
            index vector and data_size must be included.
            ( Note: 3d_data(index) = indexed_data 
                for 4d data the same index is used for each time
                point )
        filename: str, 
        data_size : tuple/list default None, 
            size of data [X,Y,Z] or [X,Y,Z,T], if None, uses size of input data array
        voxel_size : tuple/list default [1,1,1],
            size of voxels [X,Y,Z] in mm 
        dtype : np.dtype/type/str default None, 
            data type of image array ('uint8','int16','double',etc...)
            If None, data type is the same as the input data
            matrix. Use this parameter to set the data type to be
            something different. Note if you want float to mean single,
            explicity use float32, as float may cast to float64 (double)
        index : np.array default None,
            1D-array index vector (see indexed data above) 
        default_scaling='pos'

    Adapted by Mike Berks from Matlab code Written by Colin Humphries
        Feb, 2000
        University of California, Irvine
        colin@alumni.caltech.edu
    '''
    # Strip final file extension in filename
    filename = os.path.splitext(filename)[0]
    img_filename = filename + '.img'
    hdr_filename = filename + '.hdr'

    #We have write_analyze_hdr as an accessible function, so no need
    #to have the write header only option here, just use this as the 
    #dedicated function for writing image data
    
    # Get data_size and dtype from the data
    if data_size is None:
        data_size = list(data.shape)
    
    if dtype is None:
        dtype = data.dtype
    else:
        #Forcing dtype through np.dtype here clears up ambiguity
        #between a) whether given as string or np.dtype and b)
        #the precision of ints and floats which may vary between
        # platforms, numpy versions etc.
        dtype = np.dtype(dtype)

    # If the type is logical then use uint8
    if dtype == np.bool:
        dtype = np.uint8
    #int64 not supported by analyze
    elif dtype == np.int64:
        dtype = np.int32

    #If scaling not set, get from data
    if default_scaling is None:
        default_scaling = [np.min(data), np.max(data)]
    
    elif type(default_scaling) is str:
        if default_scaling == 'auto':
            if np.any(data < 0):
                default_scaling = 'posneg'
            else:
                default_scaling = 'pos'

    #Pack the data into a bytes buffer, either in sparse or full form
    if sparse:
        buf = sparse_analyze_buffer(data, dtype, index, data_size)
    else:
        buf = full_analyze_buffer(data, dtype, index, data_size)

    #Write the array data
    with open(img_filename, "wb") as fid:
        if buf:
            fid.write(buf)

    # Write header file
    write_analyze_hdr(
        hdr_filename,data_size,voxel_size,dtype,default_scaling,sparse)    


def full_analyze_buffer(data, dtype, index=None, data_size=None):
    ''''
    Pack data into byte array for writing, using all data
    '''
    if data_size is None:
        data_size = data.shape

    if index is None:
        ndata = data

    else:

        if len(data_size) == 3:
            # Output 3-d data
            ndata = np.zeros( np.prod(data_size) )
            ndata[index] = data

        else:
            # Output 4-d data 
            # The 4d data is treated separately here because the indexing is
            # different.
            #To file write in row-major form, so create N-Data as (N4 x (N1*N2*N3))
            ndata = np.zeros( (data_size[3], np.prod(data_size[0:3])) )
            for ii in range(data_size[3]):
                ndata[ii,index] = data[ ii*len(index):(ii+1)*len(index) ]

    #Now convert array to bytes buffer
    buf = ndata.astype(dtype).tobytes('F')
    return buf

def sparse_analyze_buffer(data, dtype, index=None, data_size=None, endian="@"):
    '''
    Pack data into byte array for writing, using list of non-zero indices and values
    '''
    if data_size is None:
        data_size = data.shape

    if index is None:
        index = np.ravel_multi_index(
            np.nonzero(data), data_size, order='C'
        )
    n_idx = len(index)
    if not n_idx:
        return []

    #Get precision from data type
    precision = format_str_numpy_to_struct(dtype)

    #Create an appropriately sized buffer
    idx_buf_sz = struct.calcsize(str(n_idx)+"i")
    data_buf_sz = struct.calcsize(str(n_idx)+precision)
    buf = bytearray(idx_buf_sz+data_buf_sz)

    #Convert C-order to Fortran order idx to save
    idx_F = np.ravel_multi_index(
        np.unravel_index(index, data_size, order='C'),
        data_size, order='F'
    )

    #Pack the data and then the indices
    values = data.flat[index]
    struct.pack_into(endian+str(n_idx)+precision, buf, 0, *values)
    struct.pack_into(endian+str(n_idx)+'i', buf, data_buf_sz, *idx_F)
    return buf


def write_analyze_hdr(filename, data_size, voxel_size, dtype, scaling, sparse=False):
    '''
    Write analyze format header file containing metadata for the image
    '''
    #Need to be careful, Matlab ints default to i32, my python to i64 (but may depend on environment?)
    #and likewise for float. In write_analyze_img we force ints to be i32 and floats to be f32,
    #so if this header function is called separately, we should interpret int/float as a int/float32
    #data type, even though that may not be how those types are implemented in python
    dtype = np.dtype(dtype)
    if dtype == np.uint8:
        datatype = 2  
        bitpix = 8
    elif dtype == np.int16: #Also short
        datatype = 4
        bitpix = 16
    elif dtype  == np.int32: #Also int
        datatype = 8
        bitpix = 32
    elif dtype == np.float32: #Also single 
        datatype = 16
        bitpix = 32
    elif dtype == np.float64: #Also float, double 
        datatype = 64
        bitpix = 64
    else:
        raise ValueError(f'Unsupported datatype {dtype}.')

    #For our sparse format, add 1 to the supported data types
    datatype += 5*sparse

    # Set the scaling values.
    if type(scaling) is str:
        if scaling in ['default','pos']:
            if bitpix == 8:
                scaling = [0, 255]
            elif bitpix == 16:
                scaling = [0, 32767]
            else:
            # Note: currently only 8 and 16 bit integers are set correctly.
            # because AIR only uses those datatypes.
                scaling = [0, 32767]
        
        elif scaling == 'posneg':
            if bitpix == 8:
                scaling = [0, 255]
            elif bitpix  == 16:
                scaling = [-32767, 32767]
            else:
        # Note: currently only 8 and 16 bit integers are set correctly.
                scaling = [-32767, 32767]

    #We'll do this a bit differently to the Matlab implementation.
    #In the Matlab version an array of 348 uint8 zeros is written to
    #the file, creating the total buffer size
    #A combination of fseek and fwrite commands are then used to overwrite
    #specific parts of this buffer, before closing the file
    #Instead, we'll create the 348 byte-array, the use struct.pack_into
    #to fill this at specific offsets and formats, then we simply
    #write this array to file
    #For comparison, the Matlab version commands are shown in comments

    #Create empty byte array - fwrite(fid,zeros(1,348),'uint8')
    hdr_array = bytearray(348)

    #fseek(fid,0) 
    #fwrite(fid,348,'int16')
    struct.pack_into('h', hdr_array, 0, 348)

    #fseek(fid,32)
    #fwrite(fid,16384,'int16')
    struct.pack_into('h', hdr_array, 32, 16384)

    #fseek(fid,38)
    #fwrite(fid,'r','char')
    struct.pack_into('c', hdr_array, 38, b'r')

    # This is the convention used for data_size that I've seen in other Analyze
    # files. You could also use this convention (eg for 3d data
    # fwrite(fid,[3,data_size(1),data_size(2),data_size(3)],'int16') ) and it should
    # still be valid.
    ndims = len(data_size)
    if ndims > 4:
        raise ValueError(f'Max supported dimensions is 4, data_size has length {ndims}.')

    #Add 4 to start of data_size and post-fill with 1s so always len(data_size)==5
    data_size = [4]+data_size+(4-ndims)*[1]

    #fseek(fid,40+0)
    #fwrite(fid,[4,data_size(1),data_size(2),1,1],'int16')
    struct.pack_into('5h', hdr_array, 40, *data_size)

    #fseek(fid,40+30)
    #fwrite(fid,datatype,'int16')
    struct.pack_into('h', hdr_array, 40+30, datatype)

    #fseek(fid,40+32)
    #fwrite(fid,bitpix,'int16')
    struct.pack_into('h', hdr_array, 40+32, bitpix)

    #fseek(fid,40+36)
    #fwrite(fid,[0,voxel_size],'float32')
    voxel_size = [float(f) for f in list(voxel_size)]
    struct.pack_into(f'{len(voxel_size)+1}f', hdr_array, 40+36, 0.0, *voxel_size)

    #fseek(fid,40+100)
    #fwrite(fid,[scaling(2), scaling(1)],'int32')
    struct.pack_into('2i', hdr_array, 40+100, scaling[1], scaling[0])

    # Note the following two fields are used by SPM and are not part of the
    # Analyze format.
    # Scale factor = 1
    #fseek(fid,40+72)
    #fwrite(fid,1,'int32')
    struct.pack_into('i', hdr_array, 40+72, 1)

    # DC offset = 0
    #fseek(fid,40+76)
    #fwrite(fid,0,'int32')
    struct.pack_into('i', hdr_array, 40+76, 1)

    # Open header file	
    with open(filename, "wb") as fid:
        fid.write(hdr_array)

    



