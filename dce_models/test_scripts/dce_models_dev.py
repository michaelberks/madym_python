#%%
DEBUG = True
import os
from image_io.analyze_format import read_analyze_img
import matplotlib.pyplot as plt
import numpy as np
import projects.patient_data as patient_data

import dce_models.dce_aif as dce_aif
import dce_models.tofts_model as tm
import dce_models.tissue_concentration as tc

import utils.utils as utils

#%%
if DEBUG:
    import importlib
    importlib.reload(dce_aif)
    importlib.reload(tm)

#%%
visit_dir = 'V:/isbe/qbi/data/travastin/A002_AS/visit1/'
analysis_dir = visit_dir + 'madym_python_auto_v1.24/'
params = ['Ktrans', 'Ve', 'Vp', 'offset', 'ERR', 'T1']
param_maps = []
for param in params:
    param_maps.append(read_analyze_img(analysis_dir + param))
ROI = read_analyze_img(analysis_dir + 'ROI')>0
ROI = ROI & np.isfinite(param_maps[0])

T1_0 = param_maps[5][ROI]

dyn_signals = utils.get_series_roi_vals(visit_dir + 'dynamic/dyn_', ROI)
dyn_ca = tc.signal_to_concentration(dyn_signals, T1_0)
#%%

aif_slice = 12
aif_path = visit_dir + 'slice_' + str(aif_slice) + '_Auto_AIF.txt'
auto_aif = dce_aif.Aif(dce_aif.AifType.FILE, filename=aif_path)
model_ca = tm.concentration_from_model(
        auto_aif, 
        param_maps[0][ROI],
        param_maps[1][ROI],
        param_maps[2][ROI],
        param_maps[3][ROI])

print(model_ca.shape)
utils.display_model_fits(dyn_ca[:,:30], model_ca[:,:30,])
