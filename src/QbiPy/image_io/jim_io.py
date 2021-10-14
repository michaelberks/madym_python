'''
Functions for reading and writing Jim ROI files

Jim is a cross-platform MRI analysis tool the QBI lab has long used
to annotate MRI images and generate ROI masks.

See http://www.xinapse.com/j-im-8-software/ for details.

This module provides functions for:

* Interpreting the contents of ROI files created in Jim, reading the information into python data structures
* Creating mask files (as numpy binary arrays) from ROI files
* Writing Jim ROI files given an input binary mask or list of ROI contour co-ordinates
'''
import numpy as np
import datetime
import os
from skimage import measure
import warnings
import re

#----------------------------------------------------------------------------
def in_poly(shape, xy, res = 10):
    yx = np.flip(xy, 1)
    roi_mask_lrg1 = measure.grid_points_in_poly( 
    (res*shape[0],res*shape[1]), res*yx)

    yx2 = np.empty_like(yx)
    yx2[:,0] = shape[0]-yx[:,0]
    yx2[:,1] = shape[1]-yx[:,1]

    roi_mask_lrg2 = np.flip(np.flip(measure.grid_points_in_poly( 
        (res*shape[0],res*shape[1]), res*yx2), 1), 0)

    res2 = res*res
    roi_frac1 = np.empty(shape)
    roi_frac2 = np.empty(shape)
    for row in range(shape[0]):
        for col in range(shape[1]):
            roi_frac1[row,col] = np.sum(
            roi_mask_lrg1[(row*res):((row+1)*res), (col*res):((col+1)*res)]
            ) / res2
            roi_frac2[row,col] = np.sum(
            roi_mask_lrg2[(row*res):((row+1)*res), (col*res):((col+1)*res)]
            ) / res2
    roi_mask = (roi_frac1 + roi_frac2) >= 1
    return roi_mask

#-----------------------------------------------------------------------------
def read_jim_roi(roi_path, roi_dims, vox_size, make_mask = False, mask_res = 1):
    '''READ_JIM_ROI read Jim format ROI file into a Matlab binary array
       [roi, slice_info] = read_jim_roi(roi_pth)
    
     Parameters:
        roi_path - path to Jim file
        roi_dims - dimensions of the ROI, annoyingly not included in Jim
                    given as (ny, nx, nz)
        vox_size - real world scaling of the dimensions
                    given as (dy, dx, dz)
        make_mask - flag to make binary mask

        mask_res - set the resolution at which we subdivide pixels to get
                an accurate rendering relative to Jim. Higher numbers produce more
                accurate masks, but take longer
    
     Returns:
        slice_info - structure containing meta information related to ROI
    
        roi - binary mask (3D array), only created if make_mask = true
    
    '''

    # contents of ROI file
    roi_lines = []
    with open(roi_path, "rt") as roi_file:
        for line in roi_file:
            roi_lines.append(line.strip('\n').strip(' '))

    slice_start_tag = 'Begin Irregular ROI'
    slice_end_tag = 'End Irregular ROI'

    slice_starts = [i for i, e in enumerate(roi_lines) 
        if e == slice_start_tag]
    slice_ends = [i for i, e in enumerate(roi_lines) 
        if e == slice_end_tag]
    
    num_slices = len(slice_starts)

    #Raise error if length of starts and ends don't match
    assert(num_slices == len(slice_ends))

    #Pre-allocate ROI
    roi = np.full(roi_dims, False)

    #Set up x, y co-ordinates for each slice
    scale = (1/vox_size[0], 1/vox_size[1])
    offset = ((roi_dims[1])/2, (roi_dims[0])/2)
    
    slice_info = list()
    for i_slice in range(num_slices):
        #Get lines defining this slice
        slice_lines = roi_lines[
            slice_starts[i_slice]+1:slice_ends[i_slice]
        ]
        
        #Get slice info and shape contour
        slice_info_i, slice_contour = get_slice(
            slice_lines, scale, offset)
        slice_info_i['roi_xy'] = slice_contour
        colour = slice_info_i['colour']
        slice_info.append(slice_info_i)

        #Don't mask if slice deleted!
        if make_mask and 'deleted' not in slice_info_i:
            #Make ROI slice mask from the contour
            i_z = slice_info_i['slice']

            #The lines below were used to make a mask froma polygon
            #in the first version of this function. It didn't align
            #very well with Jim, we now use a custom in_poly function
            #based on skimage.measure.grid_in_polygon. Code kept here
            #for posterity. If you want to run these lines you need to
            #import Image, ImageDraw and PIL
            #roi_img = Image.new('L', (roi_dims[0], roi_dims[1]), 0)
            #ImageDraw.Draw(roi_img).polygon(slice_contour, outline=1, fill=colour)
            #roi_i = np.array(roi_img).astype('bool')

            #Call in in_poly to make mask from polygon
            roi_i = in_poly((roi_dims[0], roi_dims[1]), 
                slice_contour, res=mask_res)

            #Negate the ROI if colour is 1
            if colour:
                roi_i = ~roi_i

            #Add ROI slice to ROI volume, and add slice_info to main list
            roi[:,:,i_z] = roi[:,:,i_z] | roi_i
                   
    
    return slice_info, roi 

#-----------------------------------------------------------------------------
def get_slice(s, scale, offset):
    shape_starts = s.index('Begin Shape')
    shape_ends = s.index('End Shape')
    #shape_starts = [i for i, e in enumerate(s) 
    #    if e == s.find('Begin Shape')>=0]
    #shape_ends = [i for i, e in enumerate(s) 
    #    if e == s.find('End Shape')>=0]
    
    slice_info_lines = s[:shape_starts]
    slice_contour_lines = s[shape_starts+1:shape_ends]
    slice_info = get_slice_info(slice_info_lines)
    slice_contour = get_slice_contour(slice_contour_lines, scale, offset)
    return slice_info, slice_contour
        
#-----------------------------------------------------------------------------        
def get_slice_info(s):
    '''
    Possible tags:
    Build version="6.0_16"
    Annotation=""
    Colour=0
    Image source="Q:dataMBGrace_4T1BALBC34955Visit2pred_805704T2Ws_unet_isensee3D_"
    Slice=4
    Created  "Unknown date/time" by Operator ID="AutoGenerated"
    Modified "14 Oct 2019 11:21:17.045 British Summer Time" by Operator ID="ross"
    Deleted  "14 Oct 2019 11:10:12.069 British Summer Time" by Operator ID="ross"
    Statistics: Area=0; Mean=0; Std Dev=0; Min=0; Max=0
  '''
    slice_info = dict()
    for info in s:
        tag = re.split(' |: |=', info)[0].lower()
        if tag == 'build':
            slice_info['version'] = info.partition('version=')[2].strip('"')

        elif tag == 'annotation':
            slice_info[''] = info.partition('=')[2].strip('"')

        elif tag == 'colour':
            slice_info['colour'] = int(info.partition('=')[2].strip('"'))

        elif tag == 'image':
            slice_info['source'] = info.partition('source=')[2].strip('"')

        elif tag == 'slice':
            slice_info['slice'] = int(info.partition('Slice=')[2])-1

        elif tag == 'created':
            slice_info['created'] = info.partition(' ')[2]

        elif tag == 'modified':
            slice_info['modified'] = info.partition(' ')[2]

        elif tag == 'deleted':
            slice_info['deleted'] = info.partition(' ')[2]

        elif tag == 'statistics':
            slice_info['statistics'] = info.partition(': ')[2]

        else:
            warnings.warn(f'Slice tag {tag} not recognised', Warning)
    
    return slice_info
#-----------------------------------------------------------------------------
def get_slice_contour(s, scale, offset):
    num_pts = int(s[0].partition('Points=')[2])
    
    slice_contour = list()
    for sxy in s[1:]: 
        slice_contour.append(get_xy(sxy, scale, offset))

    assert(len(slice_contour) == num_pts)
        #error('Slice shape doesn''t match listed number of points')       
    return slice_contour
#-----------------------------------------------------------------------------
def get_xy(s, scale, offset):
    #Extract values for x and y from co-ordinate string
    xy = [
        float(ss.partition('=')[2]) for ss in s.split(';')]
    
    #Scale and offset each co-ord then wrap in a tuple and return
    return tuple([p*s + o for (p,s, o) in zip(xy, scale, offset)])

#-----------------------------------------------------------------------------
def write_jim_roi(roi, thresh, vox_size, jim_path, min_contour=0,
    version_str="", image_src="", operator="AutoGenerated"): 
    '''
    Write a an ROI file that can read in the annotation software Jim from a 3D ROI image

    Parameters:
        roi : np.array
            Image or binary mask from which to generate ROI
        
        thresh: float
            Value in roi image used to generate contours. 
            For binary masks use 0
        
        vox_size: 3 element tuple/list
            Voxel size (in mm), required to write contour values as 
            real world units in jim roi file
        
        jim_path: str
            Output path for ROI file
        
        min_contour=0 : int
            Minimum number of points needed in contour to write as a shape
            in ROI file. Contours below this size are ignored

        version_str="" : str
            Completes version string field in Jim ROI file meta info
        
        image_src="": str
            Completes image source field in Jim ROI file meta info

        operator="AutoGenerated"
            Completes operator field in Jim ROI file meta info

    Returns:
        None

    '''   
    #Open file for writing
    os.makedirs(os.path.dirname(jim_path),
        exist_ok=True)
    
    with open(jim_path, "w") as jim_file:
        write_slices(jim_file, roi, thresh, vox_size, 
            min_contour, version_str, image_src, operator)

#-----------------------------------------------------------------------------        
def write_jim_roi_from_list(contour_list, jim_path,
    version_str="", image_src="", operator="AutoGenerated", 
    scale=(1,1), offset=(0,0)):
    '''
    Write a Jim ROI file from a list of 2D x,y contours

    Parameters:
        contour_list: list 
            list of 2 element tuple, each of which must contain
            as first element a 2D arrays defining a contour (in
            real world (x,y) mm units) and
            second element specifying slice number 

        jim_path: str
            Output path for ROI file
        
        version_str="" : str
            Completes version string field in Jim ROI file meta info
        
        image_src="": str
            Completes image source field in Jim ROI file meta info

        operator="AutoGenerated"
            Completes operator field in Jim ROI file meta info

        scale-(1,1): 2 element tuple
            Additional scaling applied to (x,y) co-ordinates in each slice contour

        offset=(0,0): 2 element tuple
            Additional offset applied to (x,y) co-ordinates in each slice contour

    Returns:
    
    '''       
    #Open file for writing
    os.makedirs(os.path.dirname(jim_path),
        exist_ok=True)
    
    with open(jim_path, "w") as jim_file:
        write_slices_from_list(jim_file, contour_list,
                               version_str, image_src, operator,
                               scale, offset)

#-----------------------------------------------------------------------------        
def write_jim_roi_from_slice_info(slice_info, jim_path,
    operator="AutoGenerated", scale=(1,1), offset=(0,0)):
    '''
    Write a Jim ROI file from a slice_info list, as returned by read_jim_roi

    Parameters:
        slice_info: list of dict
            List of slice information data-structures, as return by read_jim_roi

        jim_path: str
            Output path for ROI file
        
        operator="AutoGenerated"
            Completes operator field in Jim ROI file meta info

        scale-(1,1): 2 element tuple
            Additional scaling applied to (x,y) co-ordinates in each slice contour

        offset=(0,0): 2 element tuple
            Additional offset applied to (x,y) co-ordinates in each slice contour

    Returns:
    
    '''       
    #Open file for writing
    os.makedirs(os.path.dirname(jim_path),
        exist_ok=True)
    
    with open(jim_path, "w") as jim_file:
        write_slices_from_slice_info(jim_file, slice_info, operator, scale, offset)

#-----------------------------------------------------------------------------
def write_slices(fid, roi, thresh, vox_size, min_contour=0,
    version_str="", image_src="", operator="AutoGenerated"):
    '''
    Write slices of ROI mask to given fid, in JIm format

    Parameters:
        fid : file indentifier
            Open file identifier of ROI files to which slice info written

        roi : np.array
            Mask image to generate slices from

        thresh : float
            Threshold at which contours in mask generated. Use 0 for binary masks

        min_contour : int = 0
            Minimum required number of points to include a contour in the ROI file. 
            Contours below this size are ignored

        version_str="" : str
            Completes version string field in Jim ROI file meta info
        
        image_src="": str
            Completes image source field in Jim ROI file meta info

        operator="AutoGenerated"
            Completes operator field in Jim ROI file meta info    

    Returns:
        None
    
    '''     
    #Go through each roi slice, find contours of any regions and write
    #these as a slice in the jim roi file
    n_y, n_x, n_z = roi.shape
    offset = (n_x/2, n_y/2)

    for i_z in range(n_z):
        contours = measure.find_contours(roi[:,:,i_z].astype(float), thresh)

        internal = list()
        for i,c1 in enumerate(contours):
            internal_i = False
            for c2 in contours[0:i]+contours[i+1:]:
                if np.all(measure.points_in_poly(c1, c2)):
                    #print('Internal contour in slice ', i_z)
                    internal_i = True
                    break
            internal.append(int(internal_i))

        for contour, internal in zip(contours,internal):
            if contour.shape[0] > min_contour:
                #Contours are returned as [row,col] not [x,y]
                contour = np.flip(contour,1)
                write_slice(fid, contour, i_z+1, internal, 
                    version_str, image_src, operator, vox_size, offset)

#-----------------------------------------------------------------------------                
def write_slices_from_list(fid, contour_list,
    version_str="", image_src="", operator="AutoGenerated", scale=(1,1), offset=(0,0)):
    '''
    Write slices of contour list to given fid, in JIm format

    Parameters:

    Returns:
    
    '''    

    for contour in contour_list:
        if len(contour)<3:
            internal = False
        else:
            internal = contour[2]
        write_slice(fid, contour[0], contour[1], int(internal), 
                    version_str, image_src, operator, scale, offset)

#-----------------------------------------------------------------------------                
def write_slices_from_slice_info(fid, slice_info,
    operator="AutoGenerated", scale=(1,1), offset=(0,0)):
    '''
    Write slices of contour list to given fid, in JIm format

    Parameters:

    Returns:
    
    '''    

    for slice in slice_info:
        
        write_slice(fid, slice['roi_xy'], slice['slice']+1, slice['colour'], 
            slice['version'], slice['source'], operator, scale, offset)

#-----------------------------------------------------------------------------
def write_slice(fid, contour, slice_no, internal, 
    version_str="", image_src="", operator="AutoGenerated", scale=(1,1), offset=(0,0)):
    print(f"Begin Irregular ROI", file=fid)
    write_slice_info(fid, contour, slice_no, internal,
        version_str, image_src, operator)
    write_slice_pts(fid, contour, scale, offset)
    print(f"End Irregular ROI", file=fid)

#-----------------------------------------------------------------------------
def write_slice_info(fid, contour, slice_no, internal,
    version_str="", image_src="", operator="AutoGenerated"):

    slice_time = datetime.datetime.now().strftime("%d %b %Y %H%M%S") 
    area,mean,std,min_v,max_v = get_contour_stats(contour)

    print(f"  Build version=\"{version_str}\"", file=fid)
    print(f"  Annotation=\"\"", file=fid)
    print(f"  Colour={internal}", file=fid)
    print(f"  Image source=\"{image_src}\"", file=fid)
    print(f"  Slice={slice_no}", file=fid)
    print(f"  Created  \"{slice_time} Greenwich Mean Time\" by Operator ID=\"{operator}\"", file=fid)
    print(f"  Statistics: Area={area}; Mean={mean}; Std Dev={std}; Min={min_v}; Max={max_v}", file=fid)

#-----------------------------------------------------------------------------
def write_slice_pts(fid, contour, scale=(1,1), offset=(0,0)):
    print("  Begin Shape", file=fid)
    print(f"    Points={len(contour)}", file=fid)
    for xy in contour:
        print(f"    X={(xy[0]-offset[0])*scale[0]:4.3f}; Y={(xy[1]-offset[1])*scale[1]:4.3f}", 
            file=fid)
    print("  End Shape", file=fid)

#-----------------------------------------------------------------------------
def get_contour_stats(contour):
    area = 0
    mean = 0
    std = 0
    min_v = 0
    max_v = 0
    return area,mean,std,min_v,max_v


if __name__ == '__main__':
    roi, slice_info = read_jim_roi(
        'Q:\\data\\MB\\Grace_4T1\\BALBC\\111111\\Visit1\\T2W_yw.roi', 
        (128,128,20), (1,1,1))

    for si in slice_info:
        print(si)

    print('Num voxels in ROI = ', np.sum(roi))


# Begin Irregular ROI
#   Build version="5.0_23"
#   Annotation=""
#   Colour=0
#   Image source="/qubi/projects2/preclinical_hypoxia/015519/Visit1/T2W.img"
#   Slice=6
#   Created  "10 Apr 2015 15:42:27.525 British Summer Time" by Operator ID="yvonne"
#   Statistics: Area=41.59375 Mean=12669.984753 Std Dev=2378.163528 Min=4657 Max=17214
#   Begin Shape
#     Points=15
#     X=3.5 Y=-2.5
#     X=3.75 Y=-1
#     X=3 Y=0.75
#     X=2 Y=2
#     X=1 Y=2.5
#     X=-0.5 Y=2.25
#     X=-2 Y=2.25
#     X=-3 Y=1
#     X=-3.25 Y=-1.25
#     X=-2.5 Y=-3.75
#     X=-1.5 Y=-5
#     X=0 Y=-5.25
#     X=1.25 Y=-5
#     X=1.5 Y=-5
#     X=2.5 Y=-3.75
#   End Shape
# End Irregular ROI
# Begin Irregular ROI
#   Build version="5.0_23"
#   Annotation=""
#   Colour=0
#   Image source="/qubi/projects2/preclinical_hypoxia/015519/Visit1/T2W.img"
#   Slice=7
#   Created  "10 Apr 2015 13:49:15.963 British Summer Time" by Operator ID="yvonne"
#   Statistics: Area=60.125 Mean=17284.787383 Std Dev=2072.68631 Min=6205 Max=23751
#   Begin Shape
#     Points=21
#     X=-1 Y=3.5
#     X=-2.75 Y=2.75
#     X=-2.75 Y=2.5
#     X=-4.25 Y=1.25
#     X=-4.25 Y=-0.75
#     X=-4.25 Y=-1
#     X=-3.5 Y=-2.75
#     X=-2.75 Y=-4
#     X=-1.75 Y=-5.25
#     X=-1.25 Y=-5.5
#     X=1.5 Y=-5.5
#     X=2.5 Y=-4.75
#     X=3.5 Y=-3.5
#     X=4.5 Y=-2.5
#     X=4.75 Y=-1.25
#     X=4.5 Y=0.25
#     X=3.75 Y=1.25
#     X=2.5 Y=2.25
#     X=1 Y=2.75
#     X=0.25 Y=3.25
#     X=-0.5 Y=3.25
#   End Shape
# End Irregular ROI
# Begin Irregular ROI
#   Build version="5.0_23"
#   Annotation=""
#   Colour=0
#   Image source="/qubi/projects2/preclinical_hypoxia/015519/Visit1/T2W.img"
#   Slice=8
#   Created  "10 Apr 2015 13:48:19.871 British Summer Time" by Operator ID="yvonne"
#   Statistics: Area=57.09375 Mean=18406.836146 Std Dev=1832.548143 Min=5852 Max=23173
#   Begin Shape
#     Points=17
#     X=4.5 Y=-1.5
#     X=4.25 Y=-0.25
#     X=3.5 Y=1.25
#     X=2.5 Y=2.25
#     X=0.5 Y=3
#     X=-1.25 Y=2.75
#     X=-2.75 Y=2
#     X=-3.75 Y=1.25
#     X=-3.75 Y=1
#     X=-4.25 Y=0
#     X=-4 Y=-2
#     X=-3.25 Y=-3.75
#     X=-1.25 Y=-5.25
#     X=0.25 Y=-5.75
#     X=1.75 Y=-5.5
#     X=3.5 Y=-4
#     X=4.25 Y=-2.75
#   End Shape
# End Irregular ROI
# Begin Irregular ROI
#   Build version="5.0_23"
#   Annotation=""
#   Colour=0
#   Image source="/qubi/projects2/preclinical_hypoxia/015519/Visit1/T2W.img"
#   Slice=9
#   Created  "10 Apr 2015 13:47:35.167 British Summer Time" by Operator ID="yvonne"
#   Statistics: Area=56.6875 Mean=21317.828256 Std Dev=1721.060488 Min=5527 Max=25124
#   Begin Shape
#     Points=26
#     X=-2 Y=2.5
#     X=-2.75 Y=1.75
#     X=-3.5 Y=1.25
#     X=-4.25 Y=0.5
#     X=-4 Y=-1
#     X=-3.5 Y=-1.75
#     X=-3.5 Y=-2.75
#     X=-3.25 Y=-2.75
#     X=-3 Y=-4
#     X=-3 Y=-4.25
#     X=-1.75 Y=-5
#     X=-0.5 Y=-5.5
#     X=1.5 Y=-5.5
#     X=2.75 Y=-4.75
#     X=3.75 Y=-4
#     X=4.25 Y=-2.75
#     X=4.5 Y=-2.25
#     X=4.5 Y=-1.25
#     X=4.5 Y=-0.5
#     X=4.25 Y=0.5
#     X=3.5 Y=1.5
#     X=2.75 Y=2
#     X=1.5 Y=2.75
#     X=0.25 Y=2.75
#     X=-0.75 Y=2.75
#     X=-1.5 Y=2.75
#   End Shape
# End Irregular ROI
# Begin Irregular ROI
#   Build version="5.0_23"
#   Annotation=""
#   Colour=0
#   Image source="/qubi/projects2/preclinical_hypoxia/015519/Visit1/T2W.img"
#   Slice=10
#   Created  "10 Apr 2015 13:46:44.610 British Summer Time" by Operator ID="yvonne"
#   Statistics: Area=47.21875 Mean=21511.267552 Std Dev=1563.171731 Min=11499 Max=24941
#   Begin Shape
#     Points=21
#     X=4 Y=-2.5
#     X=4 Y=-1.25
#     X=4 Y=0
#     X=3.5 Y=1
#     X=2.5 Y=2
#     X=2.25 Y=2
#     X=0.75 Y=2.5
#     X=-0.75 Y=2.75
#     X=-2.25 Y=2
#     X=-3 Y=1.5
#     X=-3.75 Y=0.25
#     X=-3.5 Y=-1
#     X=-3.5 Y=-1.25
#     X=-3 Y=-3
#     X=-2.25 Y=-4
#     X=-1.25 Y=-4.5
#     X=0 Y=-5
#     X=1 Y=-5
#     X=2.5 Y=-4.75
#     X=3.5 Y=-3.75
#     X=3.75 Y=-3.25
#   End Shape
# End Irregular ROI
# Begin Irregular ROI
#   Build version="5.0_23"
#   Annotation=""
#   Colour=0
#   Image source="/qubi/projects2/preclinical_hypoxia/015519/Visit1/T2W.img"
#   Slice=11
#   Created  "10 Apr 2015 13:46:03.691 British Summer Time" by Operator ID="yvonne"
#   Statistics: Area=29.875 Mean=23424.306226 Std Dev=2229.218318 Min=13341 Max=27320
#   Begin Shape
#     Points=24
#     X=-1.5 Y=1.75
#     X=-2.5 Y=0.75
#     X=-2.75 Y=-0.5
#     X=-2.75 Y=-0.75
#     X=-2 Y=-2.5
#     X=-0.75 Y=-3.75
#     X=0.25 Y=-3.75
#     X=1 Y=-4.5
#     X=1.75 Y=-4.75
#     X=2 Y=-4.75
#     X=2.75 Y=-4
#     X=3 Y=-3.25
#     X=3.25 Y=-2.25
#     X=2.75 Y=-2.25
#     X=3 Y=-1.5
#     X=3 Y=-0.5
#     X=2.75 Y=-0.25
#     X=3 Y=0.5
#     X=2.75 Y=1.25
#     X=2.25 Y=1.5
#     X=2 Y=1.75
#     X=0.75 Y=2
#     X=0 Y=2
#     X=-0.75 Y=1.75
#   End Shape
# End Irregular ROI
