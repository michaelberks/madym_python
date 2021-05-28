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
    K_pos: np.array, K_neg: np.array, F_pos: np.array, F_neg: np.array, 
    f_a: np.array, aoffset: np.array, voffset: np.array)->np.array:
    #
    # Inputs:
    #   aif (Aif object, n_t): object to store and resample arterial input function values (1 for each time point)
    #
    # Inputs:
    #      K_pos, K_neg, F_pos, F_neg - bi-exponetial IRF parameters
    #
    #      aoffset - offset times of arrival for conccentraion for Ca_t
    #
    #      voffset - offset times of arrival for conccentraion for Cv_t
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
    K_pos = np.atleast_1d(K_pos)
    K_neg = np.atleast_1d(K_neg)
    F_pos = np.atleast_1d(F_pos)
    F_neg = np.atleast_1d(F_neg)
    f_a = np.atleast_1d(f_a)
    aoffset = np.atleast_1d(aoffset)
    voffset = np.atleast_1d(voffset)

    n_t = aif.times_.size

    K_max = 1e9

    #Check dimensions of parameters match each other
    n_vox = max([p.size for p in
        [K_pos,K_neg,F_pos,F_neg,f_a,aoffset,voffset]])

    if K_pos.size > 1 and K_pos.size != n_vox:
        print('Error, size of K_pos (#d) does not match the size of the other parameters (#d)',
        K_pos.size, n_vox)
        return np.empty(0)
    if K_pos.ndim > 1:
        K_pos.shape = n_vox

    if K_neg.size > 1 and K_neg.size != n_vox:
        print('Error, size of K_neg (#d) does not match the size of the other parameters (#d)',
        K_neg.size, n_vox)
        return np.empty(0)

    if K_neg.ndim > 1:
        K_neg.shape = n_vox

    if F_pos.size > 1 and F_pos.size != n_vox:
        print('Error, size of F_pos (#d) does not match the size of the other parameters (#d)',
        F_pos.size, n_vox)
        return np.empty(0)

    if F_pos.ndim > 1:
        F_pos.shape = n_vox

    if F_neg.size > 1 and F_neg.size != n_vox:
        print('Error, size of F_neg (#d) does not match the size of the other parameters (#d)',
        F_neg.size, n_vox)
        return np.empty(0)

    if F_neg.ndim > 1:
        F_neg.shape = n_vox

    if aoffset.size > 1 and aoffset.size != n_vox:
        print('Error, size of aoffset (#d) does not match the size of the other parameters (#d)',
        aoffset.size, n_vox)
        return np.empty(0)

    if aoffset.ndim > 1:
        aoffset.shape = n_vox

    if voffset.size > 1 and voffset.size != n_vox:
        print('Error, size of voffset (#d) does not match the size of the other parameters (#d)',
        voffset.size, n_vox)
        return np.empty(0)

    if voffset.ndim > 1:
        voffset.shape = n_vox
    f_v = 1 - f_a

    #Get AIF and PIF, labelled in model equation as Ca_t and Cv_t
    #Resample AIF and get AIF times
    #Make time relative to first scan, and compute time intervals
    t = aif.times_

    #Resample the AIF
    Ca_t = aif.resample_AIF(aoffset) #nv x nt

    resample_AIF = np.any(f_a)
    if resample_AIF:
        Ca_t = aif.resample_AIF(aoffset)
    else:
        Ca_t = np.zeros((n_vox,n_t))

    if np.any(f_v):
        Cv_t = aif.resample_PIF(voffset, ~resample_AIF, True)
    else:
        Cv_t = np.zeros((n_vox,n_t))

    #Irf is of form: I(t) = F_pos.exp(-tK_pos) + F_neg.exp(-tK_neg)
    #C(t) = I(t) ** Ca(t)
    C_t = np.zeros((n_vox,n_t)) 
    Ft_pos = 0
    Ft_neg = 0

    Cp_t0 = f_a*Ca_t[:,0] + f_v * Cv_t[:,0]

    for i_t in range(1, n_t):
        delta_t = t[i_t] - t[i_t-1]

        #Compute combined arterial and vascular input for this time
        Cp_t1 = f_a*Ca_t[:,i_t] + f_v * Cv_t[:,i_t] #n_v,1

        #Update the exponentials for the transfer terms in the two compartments
        e_delta_pos = np.exp(-delta_t * K_pos)
        e_delta_neg = np.exp(-delta_t * K_neg)

        #Use iterative trick to update the convolutions of transfers with the
        #input function
        A_pos = delta_t * 0.5 * (Cp_t1 + Cp_t0 * e_delta_pos)
        A_pos[K_pos > K_max] = 0    
            
        A_neg = delta_t * 0.5 * (Cp_t1 + Cp_t0 * e_delta_neg)
        A_neg[K_neg > K_max] = 0

        Ft_pos = Ft_pos * e_delta_pos + A_pos
        Ft_neg = Ft_neg * e_delta_neg + A_neg

        #Combine the two compartments with the rate constant to get the final
        #concentration at this time point
        C = F_neg * Ft_neg + F_pos * Ft_pos
        C[np.isnan(C)] = 0

        C_t[:,i_t] = C
        Cp_t0 = Cp_t1
    return C_t
    
