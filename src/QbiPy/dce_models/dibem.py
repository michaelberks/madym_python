'''
Functions for working with generic Dual-input bi-exponential models (DIBEM)
including functions to convert to/from the two-compartment exchange model
(2CXM) and the active-uptake and efflux model (AUEM)

---------------------- AUEM conversions ----------------------------------
Concentration model equation
   Cl_t = F_p.(E_i.exp(-t/Ti) / (1 - T_e/T_i) + (1 - E_i/(1 - T_e / T_i)).exp(-t/Te)) * Cp_t

 Where
   Cp_t = (f_a.Ca_t + f_v.Cv_t) / (1 - Hct)

   F_p - flow plasma rate
   T_e = v_ecs / (F_p + k_i) - extracellular mean transit time
   T_i = vi / kef - intracellular mean transit time
   E_i = ki / (Fp + ki) - the hepatic uptake fraction
   f_a - the arterial fraction
   f_v = 1 - fa - estimate of hepatic portal venous fraction
   v_i = 1 - v_ecs - estimate of intracellular volume
 
 See paper: Invest Radiol. 2017 Feb52(2):111-119. doi: 10.1097/RLI.0000000000000316.
   "Quantitative Assessment of Liver Function Using Gadoxetate-Enhanced Magnetic Resonance Imaging: 
   Monitoring Transporter-Mediated Processes in Healthy Volunteers"
   Georgiou L1, Penny J, Nicholls G, Woodhouse N, Bl FX, Hubbard Cristinacce PL, Naish JH.

---------------------- 2CXM conversions ----------------------------------
 2CXM model is bi-exponential, with  concentration computed as
   C(t) = [ F_pos.exp(-t.K_pos) + F_neg.exp(-t.K_neg) ] ** Ca(t)

 Where
   K_pos = K_sum + K_root
   K_neg = K_sum - K_root
 
   E_pos = (T_neg - Kb) / (T_neg + T_pos)
   F_pos = F_p.E_pos
   F_neg = F_p.(1 - E_pos)

 Derived from

   Kp = (F_p + PS) / v_p
   Ke = PS / v_ecs
   Kb = F_p / v_p
   K_sum = 0.5*(Kp + Ke)
   K_root = 0.5* sqrt( (Kp + Ke).^2 - 4*Ke *Kb)

 Where

   F_p - flow plasma rate
   PS = extraction flow
   v_e - extra cellular extra vascular volume
   v_p - plasma vlume
 
 See paper: Phys Med Bio. 201055:6431-6643
   "Error estimation for perfusion parameters obtained using the 
   two-compartment exchange model in dynamic contrast-enhanced MRI: a simulation study"
   R Luypaert, S Sourbron, S Makkat and J de Mey.

Created: 01-Feb-2019
Author: Michael Berks 
Email : michael.berks@manchester.ac.uk 
Phone : +44 (0)161 275 7669 
Copyright: (C) University of Manchester

'''

import warnings
import numpy as np
from scipy.interpolate import interp1d
import QbiPy.dce_models.dce_aif as dce_aif

#
#-------------------------------------------------------------------------------
def params_2CXM_to_DIBEM(F_p, PS, v_e, v_p, using_Fp=False):
    '''
    compute the derived parameters for the 2CXM
    model given input physiological parameters
    [K_pos, K_neg, F_pos, F_neg] = two_cx_params_phys_to_model(F_p, PS, v_e, v_p)

    Inputs:
        F_p - flow plasma rate

        PS - extraction flow

        v_e - extravascular, extracellular volume

        v_p - plasma volume

    Outputs:
        F_pos, F_neg - scalars in model IRF

        K_pos, K_neg - exponents in model IRF

    Notes:

    We can derive the params in a couple of ways, which remain stable under
    different conditions of ve, vp, PS and FP

    The first way is as derived in the Sourbron 2011 MRM paper, which is valid
    except when PS = 0 or FP = 0. The second method is as derived in Luypaert 
    paper 2010 paper. It works when PS or FP = 0, but doesn't like ve or vp = 0
    '''
    F_p = np.array(F_p)
    PS = np.array(PS)
    v_e = np.array(v_e)
    v_p = np.array(v_p)

    method1 = (PS > 0) & (F_p > 0) & ((v_e + v_p) > 0)
    method2 = ~method1

    #We're assuming all params have been passed in the same size, not doing any
    #error checks here
    dims_sz = F_p.shape
    K_pos = np.zeros(dims_sz)
    K_neg = np.zeros(dims_sz)
    E_pos = np.zeros(dims_sz)

    ## Method 1: Sourbron 2011
    #First derive the secondary parameters from the input Pk parameters
    E = PS[method1] / (PS[method1] + F_p[method1]) #Extraction fraction
    e = v_e[method1] / (v_p[method1] + v_e[method1]) #Extractcellular fraction

    tau = (E - E*e + e) / (2*E)
    tau_root = np.sqrt(1 - 4*(E*e*(1-E)*(1-e)) / ((E - E*e + e)**2) )
    tau_pos = tau * (1 + tau_root)
    tau_neg = tau * (1 - tau_root)

    K_pos[method1] = F_p[method1] / ((v_p[method1] + v_e[method1])*tau_neg)
    K_neg[method1] = F_p[method1] / ((v_p[method1] + v_e[method1])*tau_pos)

    E_pos[method1] = (tau_pos - 1) / (tau_pos - tau_neg)

    ## Method 2
    Kp = (F_p[method2] + PS[method2]) / v_p[method2]
    Ke = PS[method2] / v_e[method2]
    Kb = F_p[method2] / v_p[method2]

    K_sum = 0.5*(Kp + Ke)
    K_root = 0.5* np.sqrt( (Kp + Ke)**2 - 4*Ke *Kb)
    K_pos[method2] = K_sum - K_root
    K_neg[method2] = K_sum + K_root

    E_pos[method2] = (K_neg[method2] - Kb) / (K_neg[method2] - K_pos[method2]) 
    ##
    if using_Fp:
        F_pos = F_p
        F_neg = E_pos
    else:
        F_pos = F_p*E_pos
        F_neg = F_p*(1 - E_pos)

    return F_pos, F_neg, K_pos, K_neg 

#
#-------------------------------------------------------------------------------
def params_DIBEM_to_2CXM(F_pos, F_neg, K_pos, K_neg, using_Fp=False):
    '''
    Starting with the derived parameters fitted in
    the 2CXM model, convert to the physiological parameters F_p, PS, ve and vep
    model given input physiological parameters
    [F_p, PS, v_e, v_p] = two_cx_params_model_to_phys(K_pos, K_neg, F_pos, F_neg)

    Inputs:
        F_pos, F_neg - scalars in 2CXM model IRF

        K_pos, K_neg - exponents in 2CXM model IRF

    Outputs:
        F_p - flow plasma rate

        PS - extraction flow

        v_e - extravascular, extracellular volume

        v_p - plasma volume
    '''
    F_pos = np.array(F_pos)
    F_neg = np.array(F_neg)
    K_pos = np.array(K_pos)
    K_neg = np.array(K_neg)

    #We derive the params based on 2009 Sourbron paper
    if not using_Fp:
        F_p = F_pos + F_neg
        E_neg = F_neg / F_p
    else:
        F_p = F_pos
        E_neg = (1 - F_neg)

    T_B = 1 / (K_pos - E_neg * (K_pos - K_neg))
    T_E = 1 / (T_B * K_pos * K_neg)
    T_P_inv = K_pos + K_neg - 1 / T_E

    v_p = F_p * T_B
    PS = F_p * (T_B * T_P_inv - 1)
    v_e = PS * T_E 

    apply_tm = (K_pos==0) & (F_pos==0)
    if np.any(apply_tm):
        PS[apply_tm] = np.NaN
        v_p[apply_tm] = 0
        v_e[apply_tm] = F_p[apply_tm] / K_neg[apply_tm]
    
    return F_p, PS, v_e, v_p

#
#-------------------------------------------------------------------------------
def params_AUEM_to_DIBEM(F_p, v_ecs, k_i, k_ef, using_Fp=False):
    '''compute the derived parameters for the AUEM given input physiological parameters
   [K_pos, K_neg, F_pos, F_neg] = active_params_phys_to_model(F_p, v_e, k_i, k_ef)

    Inputs:
        F_p - flow plasma rate

        v_ecs - extra-cellular space (v_i = 1 - v_ecs)

        k_i - active-uptake rate

        k_ef - efflux rate

    Outputs:
        F_pos, F_neg - scalars in model IRF
    
        K_pos, K_neg - exponents in model IRF
    '''
    F_p = np.array(F_p)
    v_ecs = np.array(v_ecs)
    k_i = np.array(k_i)
    k_ef = np.array(k_ef)

    #Compute derived parameters from input parameters
    T_e = v_ecs / (F_p + k_i) # extracellular mean transit time
    v_i = 1 - v_ecs # - etsimate of intracellular volume
    T_i = v_i / k_ef # intracellular mean transit time
    E_i = k_i / (F_p + k_i) # the hepatic uptake fraction

    #This can also be precomputed
    E_pos = E_i / (1 - T_e/T_i)

    K_neg = 1 / T_e
    K_pos = 1 / T_i

    if using_Fp:
        F_pos = F_p
        F_neg = E_pos
    else:
        F_pos = F_p*E_pos
        F_neg = F_p*(1 - E_pos)

    return F_pos, F_neg, K_pos, K_neg, 

#
#-------------------------------------------------------------------------------
def params_DIBEM_to_AUEM(F_pos, F_neg, K_pos, K_neg, 
    using_Fp=False, warn_mode = 'warn'):
    '''
    Starting with the derived parameters fitted in
    the IRF-3 model, convert to the physiological parameters F_p, v_ecs, k_i
    and k_ef
    model given input physiological parameters
    [F_p, v_ecs, k_i, k_ef] = active_params_model_to_phys(K_pos, K_neg, F_pos, F_neg)

    Inputs:
        F_pos, F_neg - scalars in 2CXM model IRF

        K_pos, K_neg - exponents in 2CXM model IRF

    Outputs:
        F_p - flow plasma rate

        v_ecs - extra-cellular space (v_i = 1 - v_ecs)

        k_i - active-uptake rate

        k_ef - efflux rate

    Concentration model equation
    Cl_t = F_p.(E_i.exp(-t/Ti) / (1 - T_e/T_i) + (1 - E_i/(1 - T_e / T_i)).exp(-t/Te)) * Cp_t

    Where
    Cp_t = (f_a.Ca_t + f_v.Cv_t) / (1 - Hct)

    F_p - flow plasma rate
    T_e = v_ecs / (F_p + k_i) - extracellular mean transit time
    T_i = vi / kef - intracellular mean transit time
    E_i = ki / (Fp + ki) - the hepatic uptake fraction
    f_a - the arterial fraction
    f_v = 1 - fa - estimate of hepatic portal venous fraction
    v_i = 1 - v_ecs - estimate of intracellular volume
    
    See paper: Invest Radiol. 2017 Feb52(2):111-119. doi: 10.1097/RLI.0000000000000316.
    "Quantitative Assessment of Liver Function Using Gadoxetate-Enhanced Magnetic Resonance Imaging:"
    Georgiou L1, Penny J, Nicholls G, Woodhouse N, Bl FX, Hubbard Cristinacce PL, Naish JH.'''
    F_pos = np.array(F_pos)
    F_neg = np.array(F_neg)
    K_pos = np.array(K_pos)
    K_neg = np.array(K_neg)

    #First get F_p from F_pos and F_neg
    if not using_Fp:
        F_p = F_pos + F_neg
        E_pos = F_pos / F_p
    else:
        F_p = F_pos
        E_pos = F_neg

    #Derivation is only valid for K_pos < K_neg. If not, the swapping
    #F_pos, K_pos for F_neg, K_neg will generate valid active parameters (and
    #an indentical concentration time series due to symmetry of the
    #bi-exponential). User defines whether swap with warning, quietly or force
    #an error if invalid voxels found
    swap_idx = K_pos > K_neg
    if np.any(swap_idx):
        if warn_mode == 'warn':
            warnings.warn(
                f'K_pos > K_neg for {np.sum(swap_idx)} of {swap_idx.size} voxels. Switching these voxels')
        elif warn_mode == 'error':
            raise RuntimeError(
                f'K_pos > K_neg for {np.sum(swap_idx)} of {swap_idx.size} voxels. ' 
                'Run with warn_mode = ''quiet'' or ''warn to switch these voxels.')
        elif warn_mode == 'quiet':
            #do nothing
            pass
        else:
            raise ValueError('Warn mode {warn_mode} not recognised. Must be ''warn'', ''quiet'' or ''error''')
        
        if not using_Fp:
            #F_p doesn't change it is the sum of F_pos and F_neg
            #E_pos needs to remade from F_neg for the swapped indices
            E_pos[swap_idx] = F_neg[swap_idx] / F_p[swap_idx]
        else:
            #F_p doesn't change, E_pos needs negating
            E_pos[swap_idx] = 1 - E_pos[swap_idx]
        
        #K_pos and K_neg are just a straight swap
        K_pos_swap = K_pos[swap_idx]
        K_pos[swap_idx] = K_neg[swap_idx]   
        K_neg[swap_idx] = K_pos_swap

    #Now derive Te, Ti and Ei
    Te = 1 / K_neg
    Ti = 1 / K_pos
    Ei = E_pos * (1 - Te / Ti)

    #Can solve for k_i in terms of F_p and Ei
    k_i = Ei * F_p / (1 - Ei)

    #Solve for v_ecs in terms of Te, F_p and K-i
    v_ecs = Te * (F_p + k_i)

    #Finally solve for k_ef in terms of v_ecs and Ti
    k_ef = (1 - v_ecs) / Ti
    return F_p, v_ecs, k_i, k_ef

#
#-------------------------------------------------------------------------------
def concentration_from_model(aif:dce_aif.Aif, 
    K_pos: np.array, K_neg: np.array, F_pos: np.array, F_neg: np.array, 
    f_a: np.array, tau_a: np.array, tau_v: np.array)->np.array:
    '''
    Compute concentration time-series from model parameters
    Inputs:
        aif (Aif object, n_t): object to store and resample arterial input function values (1 for each time point)
    
        K_pos, K_neg, F_pos, F_neg - bi-exponetial IRF parameters

        tau_a - offset times of arrival for conccentraion for Ca_t

        tau_v - offset times of arrival for conccentraion for Cv_t
    
     Outputs:
       C_model (2D numpy array, n_t x n_vox) - Model concentrations at each time point for each 
       voxel computed from model paramaters
    
     '''
    
    #We allow the model paramaters to be scalar, whilst also accepting higher dimension arrays
    #so call at least 1d to make sure the scalars are treated as arrays (otherwise calling size etc won't work)
    K_pos = np.atleast_1d(K_pos)
    K_neg = np.atleast_1d(K_neg)
    F_pos = np.atleast_1d(F_pos)
    F_neg = np.atleast_1d(F_neg)
    f_a = np.atleast_1d(f_a)
    tau_a = np.atleast_1d(tau_a)
    tau_v = np.atleast_1d(tau_v)

    n_t = aif.times_.size

    K_max = 1e9

    #Check dimensions of parameters match each other
    n_vox = max([p.size for p in
        [K_pos,K_neg,F_pos,F_neg,f_a,tau_a,tau_v]])

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

    if tau_a.size > 1 and tau_a.size != n_vox:
        print('Error, size of tau_a (#d) does not match the size of the other parameters (#d)',
        tau_a.size, n_vox)
        return np.empty(0)

    if tau_a.ndim > 1:
        tau_a.shape = n_vox

    if tau_v.size > 1 and tau_v.size != n_vox:
        print('Error, size of tau_v (#d) does not match the size of the other parameters (#d)',
        tau_v.size, n_vox)
        return np.empty(0)

    if tau_v.ndim > 1:
        tau_v.shape = n_vox
    f_v = 1 - f_a

    #Get AIF and PIF, labelled in model equation as Ca_t and Cv_t
    #Resample AIF and get AIF times
    #Make time relative to first scan, and compute time intervals
    t = aif.times_

    #Resample the AIF
    Ca_t = aif.resample_AIF(tau_a) #nv x nt

    resample_AIF = np.any(f_a)
    if resample_AIF:
        Ca_t = aif.resample_AIF(tau_a)
    else:
        Ca_t = np.zeros((n_vox,n_t))

    if np.any(f_v):
        Cv_t = aif.resample_PIF(tau_v, ~resample_AIF, True)
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
    
