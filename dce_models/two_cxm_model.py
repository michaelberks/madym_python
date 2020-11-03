import numpy as np
from scipy.interpolate import interp1d
import dce_models.dce_aif as dce_aif


#
#---------------------------------------------------------------------------------
def params_phys_to_model(F_p, PS, v_e, v_p, using_Fp=False):
    K_pos, K_neg, F_pos, F_neg = [], [], [], []
    return K_pos, K_neg, F_pos, F_neg

#
#---------------------------------------------------------------------------------
def params_model_to_phys(K_pos, K_neg, F_pos, F_neg, using_Fp=False):
    Fp, PS, Ve, Vp = [],[],[],[]
    return Fp, PS, Ve, Vp

#
#---------------------------------------------------------------------------------
def concentration_from_model(aif:dce_aif.Aif, 
    Fp: np.array, PS: np.array, Ve: np.array, Vp: np.array, offset: np.array)->np.array:
    #EXTENDED_KETY_MODEL *Insert a one line summary here*
    #   [model_signals] = extended_kety_model(dyn_times, aif, Fp, Vp, Ve)
    #
    # Inputs:
    #   aif (Aif object, n_t): object to store and resample arterial input function values (1 for each time point)
    #
    # Inputs:
    #      Fp - flow plasma rate
    #
    #      PS - extraction flow
    #
    #      v_e - extravascular, extracellular volume
    #
    #      v_p - plasma volume
    #
    #      offset - offset times of arrival for conccentraion for Ca_t
    #
    #
    #
    # Outputs:
    #   C_model (2D numpy array, n_t x n_vox) - Model concentrations at each time point for each 
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
    Fp = np.atleast_1d(Fp)
    PS = np.atleast_1d(PS)
    Ve = np.atleast_1d(Ve)
    Vp = np.atleast_1d(Vp)
    offset = np.atleast_1d(offset)

    n_t = aif.times_.size

    K_max = 1e9

    #Check dimensions of parameters match each other
    n_vox = max([p.size for p in
        [Fp, PS, Ve, Vp, offset]])

    if Fp.size > 1 and Fp.size != n_vox:
        print('Error, size of Fp (#d) does not match the size of the other parameters (#d)',
        Fp.size, n_vox)
        return np.empty(0)
    if Fp.ndim > 1:
        Ve = Ve.reshape(n_vox)

    if Ve.size > 1 and Ve.size != n_vox:
        print('Error, size of Ve (#d) does not match the size of the other parameters (#d)',
        Ve.size, n_vox)
        return np.empty(0)

    if Ve.ndim > 1:
        Ve = Ve.reshape(n_vox)

    if Vp.size > 1 and Vp.size != n_vox:
        print('Error, size of Vp (#d) does not match the size of the other parameters (#d)',
        Vp.size, n_vox)
        return np.empty(0)

    if Vp.ndim > 1:
        Vp = Vp.reshape(n_vox)

    if offset.size > 1 and offset.size != n_vox:
        print('Error, size of offset (#d) does not match the size of the other parameters (#d)',
        offset.size, n_vox)
        return np.empty(0)

    offset = offset.reshape(1,n_vox)

    #Make time relative to first scan, and compute time intervals
    t = aif.times_
    delta_t = np.diff(t)

    #Resample the AIF
    Ca_t = aif.resample_AIF(offset) #nt x nv

    #We derive the params in a standalone function now, this takes care of
    #checks on FP, PS to choose the best form of derived parameters
    K_pos, K_neg, F_pos, F_neg = params_phys_to_model(
        Fp, PS, Ve, Vp)

    #Irf is of form: I(t) = F_pos.exp(-tK_pos) + F_neg.exp(-tK_neg)
    #C(t) = I(t) ** Ca(t)
    C_t = np.zeros((n_vox,n_t)) 
    Ca_t0 = Ca_t[0]
    Ft_pos = 0
    Ft_neg = 0
    for i_t in range(1, n_t):
        
        #Compute (offset) combined arterial and vascular input for this time
        Ca_ti = Ca_t[i_t]
        
        #Update the exponentials for the transfer terms in the two compartments        
        et_pos = np.exp(-delta_t[i_t] * K_pos)
        et_neg = np.exp(-delta_t[i_t] * K_neg)
            
        #Use iterative trick to update the convolutions of transfers with the
        #input function. This only works when the exponent is finite, otherwise
        #the exponential is zero, and the iterative solution is not valid. For
        #these voxels, set A_pos/neg to zero
        A_pos = delta_t * 0.5 * (Ca_ti + Ca_t0*et_pos)
        A_pos[K_pos > K_max] = 0    
        
        A_neg = delta_t * 0.5 * (Ca_ti + Ca_t0*et_neg)
        A_neg[K_neg > K_max] = 0
        
        Ft_pos = Ft_pos*et_pos + A_pos
        Ft_neg = Ft_neg*et_neg + A_neg
        
        #Combine the two compartments with the rate constant to get the final
        #concentration at this time point
        C_t[:,i_t] = F_pos*Ft_pos + F_neg*Ft_neg
        Ca_t0 = Ca_ti
    
