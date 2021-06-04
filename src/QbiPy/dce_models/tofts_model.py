import numpy as np
from scipy.interpolate import interp1d
import QbiPy.dce_models.dce_aif as dce_aif

#
#---------------------------------------------------------------------------------
def concentration_from_model(aif:dce_aif.Aif, 
    Ktrans: np.array, Ve: np.array, Vp: np.array, tau_a: np.array)->np.array:
    #EXTENDED_KETY_MODEL *Insert a one line summary here*
    #   [model_signals] = extended_kety_model(dyn_times, aif, Ktrans, Vp, Ve)
    #
    # Inputs:
    #   aif (Aif object, num_times): object to store and resample arterial input function values (1 for each time point)
    #
    #   Ktrans (1D numpy array, num_voxels): Ktrans values, 1 for each voxel
    #
    #   Vp (1D numpy array, num_voxels): Vp values, 1 for each voxel
    #
    #   Ve (1D numpy array, num_voxels): Ve values, 1 for each voxel
    #
    #   tau_a (1D numpy array, num_voxels): Ve values, 1 for each voxel
    #
    #
    # Outputs:
    #   C_model (2D numpy array, num_times x num_voxels) - Model concentrations at each time point for each 
    #   voxel computed from model paramaters
    #
    # Example:
    #
    # Notes:
    #
    # See also:
    #
    # Created: 29-Mar-2017
    # Author: Michael Berks 
    # Email : michael.berks@manchester.ac.uk 
    # Phone : +44 (0)161 275 7669 
    # Copyright: (C) University of Manchester

    #We allow the mdoel paramaters to be scalar, whilst also accepting higher dimension arrays
    #so call at least 1d to make sure the scalars are treated as arrays (otherwise calling size etc won't work)
    Ktrans = np.atleast_1d(Ktrans)
    Ve = np.atleast_1d(Ve)
    Vp = np.atleast_1d(Vp)
    tau_a = np.atleast_1d(tau_a)

    num_times = aif.times_.size

    #Check dimensions of parameters match each other
    num_voxels = Ktrans.size
    if Ktrans.ndim > 1:
        Ktrans = Ktrans.reshape(num_voxels)

    if Ve.size > 1:
        if num_voxels == 1:
            num_voxels = Ve.size
        elif Ve.size != num_voxels:
            print('Error, size of Ve (%d) does not match the size of the Ktrans (%d)',
            Ve.size, num_voxels)
            return np.empty(0)

        if Ve.ndim > 1:
            Ve = Ve.reshape(num_voxels)

    if Vp.size > 1:
        if num_voxels == 1:
            num_voxels = Vp.size
        elif Vp.size != num_voxels:
            print('Error, size of Vp (%d) does not match the size of the other parameter maps (%d)',
            Vp.size, num_voxels)
            return np.empty(0)

        if Vp.ndim > 1:
            Vp = Vp.reshape(num_voxels)

    if tau_a.size > 1:
        if num_voxels == 1:
            num_voxels = tau_a.size
        elif tau_a.size != num_voxels:
            print('Error, size of tau_a (%d) does not match the size of the other parameter maps (%d)',
            tau_a.size, num_voxels)
            return np.empty(0)

        tau_a = tau_a.reshape(num_voxels)

    #precompute exponential
    k_ep = Ktrans / Ve

    #Make time relative to first scan, and compute time intervals
    t = aif.times_

    #create container for running integral sum
    #integral_sum = np.zeros(num_voxels) #1d nv

    #Resample the AIF
    aif_offset = aif.resample_AIF(tau_a) #nv x nt
    
    #Create container for model concentrations
    C_model = np.zeros([num_voxels, num_times])

    e_i = 0
    for i_t in range(1, num_times):
        
        #Get current time, and time change
        t1 = t[i_t] #scalar
        delta_t = t1 - t[i_t-1] #scalar
        
        #Compute (tau_a) combined arterial and vascular input for this time
        Ca_t0 = aif_offset[:,i_t-1]#1d n_v
        Ca_t1 = aif_offset[:,i_t]#1d n_v
        
        #Update the exponentials for the transfer terms in the two compartments
        e_delta = np.exp(-delta_t * k_ep) #1d n_v
        
        #Combine the two compartments with the rate constant to get the final
        #concentration at this time point
        A = delta_t * 0.5 * (Ca_t1 + Ca_t0*e_delta)

        e_i = e_i * e_delta + A
        C_model[:,i_t] = Vp * Ca_t1 + Ktrans * e_i
        
        
    '''
    e0 = np.exp(k_ep*t[0]) # 1d n_v
    for i_t in range(1, num_times):
        e1 = np.exp(k_ep*t[i_t]) #1d n_v
        aif_t0 = aif_offset[i_t-1,:]#1d n_v
        aif_t1 = aif_offset[np.newaxis,i_t,:]#1d n_v
        
        a_i = delta_t[i_t-1] * 0.5 * (aif_t1*e1 + aif_t0*e0)
        
        integral_sum = (e0 * integral_sum + Ktrans * a_i) / e1
        C_model[i_t,:] = Vp*aif_t1 + integral_sum
        e0 = e1'''

    return C_model
