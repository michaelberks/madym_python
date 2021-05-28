#**************************************************************************
# Interactive script used in devloping python wrapper functions to the main
# Madym C++ toolkit.
#--------------------------------------------------------------------------
#%%
%load_ext autoreload

%autoreload 2
import numpy as np
# %%
from madym_interface import madym_DCE_lite

madym_DCE_lite.run()
# %%
from madym_interface import madym_DCE
madym_DCE.run()
# %%
from madym_interface import madym_T1
madym_T1.run()
# %%
from madym_interface import utils
v1 = utils.local_madym_version()
v2 = utils.latest_madym_version()
utils.check_madym_updates()
# %%
from madym_interface import install_madym
install_madym.install_madym()
# %%
from madym_interface import run_madym_tests
run_madym_tests.run_madym_tests()
#%%
from dce_models.dce_aif import Aif
from dce_models.tissue_concentration import signal_to_concentration, concentration_to_signal
from dce_models import tofts_model
import matplotlib.pyplot as plt

from madym_interface import madym_DCE, madym_DCE_lite, madym_T1
#%%
ktrans = 0.25
ve = 0.2
vp = 0.1
tau = 0
injection_img = 8
t = np.linspace(0, 5, 100)
aif = Aif(times=t, prebolus=injection_img, hct=0.42)
C_t = tofts_model.concentration_from_model(aif, ktrans, ve, vp, tau)

#Add some noise and rescale so baseline mean is 0
C_tn = C_t + np.random.randn(1,100)/10
C_tn = C_tn - np.mean(C_tn[:,0:injection_img])
#%%
aif = Aif(times=t, prebolus=injection_img, hct=0.42)
with open ('C:\\isbe\\aif_py.txt', 'w') as aif_file:
    for t_i, c_i in zip(t, aif.base_aif_[0,:]):
        print(f'{t_i:5.4f} {c_i:5.4f}', file=aif_file)

#%%
#Use madym_DCE lite to fit this data
model_params_C, model_fit_C, _, _, CmC_t,_ = madym_DCE_lite.run(
    model='ETM', input_data=C_tn, dyn_times=t)

#Convert the concentrations to signals with some notional T1 values and
#refit using signals as input
FA = 20
TR = 3.5
T1_0 = 1000
r1_const = 3.4
S_t0 = 100
S_tn = concentration_to_signal(
    C_tn, T1_0, S_t0, FA, TR, r1_const, injection_img)

model_params_S, model_fit_S, _,_,CmS_t,Cm_t = madym_DCE_lite.run(
    model='ETM',
    input_data=S_tn,
    dyn_times=t,
    input_Ct=0,
    T1=T1_0,
    TR=TR,
    FA=FA,
    r1_const=r1_const,
    injection_image=injection_img)

#Convert the modelled concentrations back to signal space
Sm_t = concentration_to_signal(
    CmS_t, T1_0, S_t0, FA, TR, r1_const, injection_img)
#%%
#Display plots of the fit
plt.figure(figsize=(16,8))
plt.suptitle('madym_DCE_lite test applied')
plt.subplot(2,2,(1 ,3))
plt.plot(t, C_tn.reshape(-1,1))
plt.plot(t, CmC_t.reshape(-1,1))
plt.legend(['C(t)', 'ETM model fit'])
plt.xlabel('Time (mins)')
plt.ylabel('Voxel concentration')
plt.title(f'Input C(t): Model fit SSE = {model_fit_C[0]:4.3f}')

plt.subplot(2,2,2)
plt.plot(t, C_tn.reshape(-1,1))
plt.plot(t, Cm_t.reshape(-1,1), '--')
plt.plot(t, CmS_t.reshape(-1,1))
plt.legend(['C(t)', 'C(t) (output from MaDym)', 'ETM model fit'])
plt.xlabel('Time (mins)')
plt.ylabel('Voxel concentration')
plt.title(f'Input S_t: Model fit SSE = {model_fit_S[0]:4.3f}')

plt.subplot(2,2,4)
plt.plot(t, S_tn.reshape(-1,1))
plt.plot(t, Sm_t.reshape(-1,1))
plt.legend(['S(t)', 'ETM model fit - converted to signal'])
plt.xlabel('Time (mins)')
plt.ylabel('Voxel signal')
plt.title(f'Input S(t): Signal SSE = {np.sum((S_tn-Sm_t)**2):4.3f}')
plt.show()

print(f'Parameter estimation (actual, fitted concentration, fitted signal)')
print(f'Ktrans: ({ktrans:3.2f}, {model_params_C[0,0]:3.2f}, {model_params_S[0,0]:3.2f})')
print(f'Ve: ({ve:3.2f}, {model_params_C[0,1]:3.2f}, {model_params_S[0,1]:3.2f})')
print(f'Vp: ({vp:3.2f}, {model_params_C[0,2]:3.2f}, {model_params_S[0,2]:3.2f})')
print(f'Tau: ({tau:3.2f}, {model_params_C[0,3]:3.2f}, {model_params_S[0,3]:3.2f})')

# %%
madym_DCE_lite.test()


# %%
from image_io.write_xtr_file import write_xtr_file
write_xtr_file('C:\\isbe\\temp.xtr', TR=2.4, FlipAngle=20.0, TimeStamp=12345)

# %%
write_xtr_file('C:\\isbe\\temp.xtr', 
    TR=2.4, FlipAngle=20.0, TimeStamp=(12345,2))
write_xtr_file('C:\\isbe\\temp.xtr', append=True,
    TimeStamp=3)


# %%
madym_DCE.test()

# %%
from t1_mapping.signal_from_T1 import signal_from_T1
signal_from_T1([1000.1,1200.1], 500, [5, 10, 20], 4)

# %%
madym_T1.test()

# %%
from madym_interface import install_madym
#%%
install_madym.install_madym(madym_root='C:\\isbe\mdm_tools')

# %%
import tkinter as tk
from tkinter import messagebox, filedialog
#%%
root = tk.Tk()
root.overrideredirect(1)
root.withdraw()
root.lift()
messagebox.showinfo(title='Step 1: locate shared drive', message='Use finder to select root of QBI share drive')
qbi_share_path = filedialog.askdirectory(
    title='Find shared drive'
)

root.destroy()
print(qbi_share_path)


