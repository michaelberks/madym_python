'''
Implementation of the two compartment exchange model
'''

import numpy as np
from QbiPy.dce_models import dce_aif, dibem

#
#---------------------------------------------------------------------------------
def concentration_from_model(aif:dce_aif.Aif, 
    Fp: np.array, PS: np.array, Ve: np.array, Vp: np.array, tau_a: np.array)->np.array:
    ''' 
    Compute concentration time-series of 2CXM from input
    paramaters. Note instead of re-implementing a bi-exponential
    model here, we call the DIBEM module to convert the 2CXM
    params to the bi-exponential parameters, and then call
    DIBEM's concentration_from_model

     Parameters:
       aif (Aif object, n_t): object to store and resample arterial input function values (1 for each time point)
    
     Parameters:
          Fp - flow plasma rate
    
          PS - extraction flow
    
          v_e - extravascular, extracellular volume
    
          v_p - plasma volume
    
          tau_a - tau_a times of arrival for conccentraion for Ca_t
    
     Returns:
       C_model (2D numpy array, n_t x n_vox) - Model concentrations at each time point for each 
       voxel computed from model paramaters
    
    '''

    #We derive the params in a standalone function now, this takes care of
    #checks on FP, PS to choose the best form of derived parameters
    F_pos, F_neg, K_pos, K_neg  = dibem.params_2CXM_to_DIBEM(
        Fp, PS, Ve, Vp)

    C_t = dibem.concentration_from_model(
        aif, F_pos, F_neg, K_pos, K_neg, 1.0, tau_a, 0)

    return C_t
    
