'''
Implementation of the extended-Tofts model
'''

import numpy as np
from scipy.interpolate import interp1d
from QbiPy.dce_models import dce_aif 
from QbiPy import helpers

#
#---------------------------------------------------------------------------------
def concentration_from_model(aif:dce_aif.Aif, 
    Ktrans: np.array, v_e: np.array, v_p: np.array, tau_a: np.array)->np.array:
    '''
    Compute concentration time-series of extended-Tofts model from input
    paramaters
    
     Parameters:
       aif (Aif object, num_times): object to store and resample arterial input function values (1 for each time point)
    
       Ktrans (1D numpy array, num_voxels): Ktrans values, 1 for each voxel
    
       v_p (1D numpy array, num_voxels): v_p values, 1 for each voxel
    
       v_e (1D numpy array, num_voxels): v_e values, 1 for each voxel
    
       tau_a (1D numpy array, num_voxels): v_e values, 1 for each voxel
    
    
     Returns:
       C_model (2D numpy array, num_times x num_voxels) - Model concentrations at each time point for each 
       voxel computed from model paramaters
    '''

    #We allow the mdoel paramaters to be scalar, whilst also accepting higher dimension arrays
    num_voxels,Ktrans, v_e, v_p, tau_a = helpers.check_param_size(
        Ktrans=Ktrans,v_e=v_e,v_p=v_p, tau_a=tau_a
    )

    #precompute exponential
    k_ep = Ktrans / v_e

    #Make time relative to first scan, and compute time intervals
    num_times = aif.times_.size
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
        C_model[:,i_t] = v_p * Ca_t1 + Ktrans * e_i
        
        
    '''
    e0 = np.exp(k_ep*t[0]) # 1d n_v
    for i_t in range(1, num_times):
        e1 = np.exp(k_ep*t[i_t]) #1d n_v
        aif_t0 = aif_offset[i_t-1,:]#1d n_v
        aif_t1 = aif_offset[np.newaxis,i_t,:]#1d n_v
        
        a_i = delta_t[i_t-1] * 0.5 * (aif_t1*e1 + aif_t0*e0)
        
        integral_sum = (e0 * integral_sum + Ktrans * a_i) / e1
        C_model[i_t,:] = v_p*aif_t1 + integral_sum
        e0 = e1'''

    return C_model
