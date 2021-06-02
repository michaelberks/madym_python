# Created: 29-Mar-2017
# Author: Michael Berks 
# Email : michael.berks@manchester.ac.uk 
# Phone : +44 (0)161 275 7669 
# Copyright: (C) University of Manchester

import numpy as np

from QbiPy.image_io.analyze_format import read_analyze_img, read_analyze_hdr, read_analyze_xtr

#-----------------------------------------------------------------------------------
def get_dyn_vals(root_path, num_vols, roi, index_fmt = '01d'):
    '''GET_DYN_VALS given directory of volumes and ROI mask, get array of
    time-series for voxels in the ROI
    [times] = get_dyn_vals(root_path, num_vols, roi, index_fmt)

    Inputs:
        root_path - folder + filename root where volumes are, or 4D array of loaded volumes

        num_vols - number of volumes to load

        roi - mask volume, must have same dims as dynamic volumes

        index_fmt ('01d') - format that converts indexes into volume suffix


    Outputs:
        dyn_signals - N_vox x N_times array of time-series for each voxel
      '''

    #If ROI is a path, load it from disk
    if type(roi) == str:
        roi = read_analyze_img(roi) > 0


    num_pts = np.count_nonzero(roi)
    dyn_signals = np.empty((num_pts, num_vols))

    load_vols = type(root_path) == str
    for i_vol in range(num_vols):
        if load_vols:
            vol_path = f"{root_path}{i_vol+1:{index_fmt}}.hdr"
            vol = read_analyze_img(vol_path)
        else:
            vol = root_path[:,:,:,i_vol]
        
        dyn_signals[:,i_vol] = vol[roi]

    return dyn_signals

#-----------------------------------------------------------------------------------
def get_dyn_vols(root_path, num_vols, apply_smoothing=False, 
    index_fmt = '01d', load_headers=False):
    '''#GET_DYN_TIMES *Insert a one line summary here*
    [times] = get_dyn_times(root_path, index_fmt, num_vols)

    Inputs:
        root_path - path to each volume to be loaded, so that the full path
        to the i-th volume is [root_path sprintf(index_fmt, i) '.hdr']

        index_fmt - defines format for indexing of volumes. In most QBi data
        this is just 01d (equivalent to num2str, with no modifiers)

        num_vols - number of volumes to load

        apply_smoothing - flag to apply tangential smoothing to each volume


    Outputs:
        dyn_vols - (Ny, Nx, Nz, Nvols) array containing each dynamic volume

        dyn_headers - header information for each volume'''
    dyn_headers = []
    dyn_vols = []
    for i_vol in range(num_vols):
        
        vol_path = f"{root_path}{i_vol+1:{index_fmt}}.hdr"
        d = read_analyze_img(vol_path)
        
        
        if i_vol == 0:
            [n_y, n_x, n_z] = d.shape
            dyn_vols = np.empty((n_y, n_x, n_z, num_vols))
        
        if apply_smoothing:
            pass
            #d = tangential_smoothing_vol(d)
        
        dyn_vols[:,:,:,i_vol] = d
        
        if load_headers:
            dyn_header = read_analyze_hdr(vol_path)
            dyn_headers.append(dyn_header)
    
    return dyn_vols, dyn_headers

#-----------------------------------------------------------------------------------
def get_dyn_xtr_data(root_path, num_vols, index_fmt = '01d'):
    '''GET_DYN_TIMES get meta information (scan time, FA, TR) from xtr files for 
    folder of dynamic volumes
    [times] = get_dyn_times(root_path, index_fmt, num_vols)

    Inputs:
        root_path - folder + filename root where volumes are

        num_vols - number of volumes to load

        index_fmt ('01d') - format that converts indexes into volume suffix


    Outputs:
        dyn_times - *Insert description of input variable here*'''

    dyn_times = np.zeros(num_vols)
    dyn_TR = np.zeros(num_vols)
    dyn_FA = np.zeros(num_vols)
    dyn_noise = np.zeros(num_vols)

    for i_vol in range(num_vols):
        vol_path = f"{root_path}{i_vol+1:{index_fmt}}.xtr"
        xtr_data = read_analyze_xtr(vol_path)
        dyn_times[i_vol] = xtr_data['TimeStamp']
        dyn_TR[i_vol] = xtr_data['FlipAngle']
        dyn_FA[i_vol] = xtr_data['TR']
        dyn_noise[i_vol] = xtr_data['NoiseSigma']
