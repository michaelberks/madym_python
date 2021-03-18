
#%%
%load_ext autoreload

%autoreload 2
#%%
%run ./simple_3D_viewer_tool.py C:/isbe/qbi/madym_testing/examples/test_dataset/madym_output/T1
# %%
import numpy as np

x = np.repeat(np.expand_dims(np.arange(5),0), 3, 0)
print(x)
# %%
spec = '03d'
a = 1
print(f"{a+1:{spec}}")
# %%
from dce_models.data_io import get_dyn_vals
from image_io.analyze_format import read_analyze_img
#%%
data_dir = 'C:/isbe/qbi/madym_testing/exciting_new_study/subject001/madym_output/ETM_auto/' 
roi = read_analyze_img(data_dir+"ROI.hdr") > 0
Ct_s = get_dyn_vals(data_dir + "Ct_sig", 75, roi)
Ct_m = get_dyn_vals(data_dir + "Ct_mod", 75, roi)


# %%
import matplotlib.pyplot as plt
#%%
plt.figure()
for i_vox in range(15):
    plt.subplot(3,5,i_vox+1)
    plt.plot(Ct_s[i_vox,:])
    plt.plot(Ct_m[i_vox,:], 'r')

plt.show()
# %%

# %%
print(f'{3.1417634:g2}')
# %%
