
#%%

%load_ext autoreload
%autoreload 2

import struct
from image_io.analyze_format import read_analyze_img, read_analyze_hdr
import numpy as np
import matplotlib.pyplot as plt
#%%
img_file = 'C:/isbe/qbi/data/milano_primovist/PRIMDCE_1/visit1/dynamic/dyn_20.img'
hdr_file = 'C:/isbe/qbi/data/milano_primovist/PRIMDCE_1/visit1/dynamic/dyn_20.hdr'

img = read_analyze_img(img_file)
hdr = read_analyze_hdr(hdr_file)

with open(img_file, 'rb') as f:
    buffer = f.read()
#%%
img_data = struct.unpack_from('<h512000', buffer, 0)
#%%
img2 = np.array(img_data).reshape((40,100,128))

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img[:,:,20])
plt.subplot(1,2,2)
plt.imshow(img2[20,:,:])

# %%
%load_ext autoreload
%autoreload 2
from image_io.analyze_format import read_analyze_hdr, read_analyze_img, read_analyze
img_file = 'C:/isbe/qbi/data/travastin/A002_AS/visit1/dynamic/dyn_20.img'
hdr_file = 'C:/isbe/qbi/data/travastin/A002_AS/visit1/dynamic/dyn_20.hdr'
#%%
hdr = read_analyze_hdr(hdr_file)
img = read_analyze_img(img_file, flip_y=True, output_type=None)
#%%
img2,hdr2 = read_analyze(img_file, flip_y=True, output_type=None)

# %%
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img[:,:,20])
plt.subplot(1,2,2)
plt.imshow(img2[:,:,20])

# %%
from image_io.analyze_format import write_analyze, write_analyze_img

write_analyze(img, 'C:/isbe/temp.hdr')
write_analyze_img(img, 'C:/isbe/temp2.hdr')
#%%
img[:,:,1:] = 0
write_analyze_img(img, 'C:/isbe/temp3.hdr', sparse=True, swap_axes=False)
#%%
img3 = read_analyze_img('C:/isbe/temp3.hdr')
#%%
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img[:,:,1])
plt.subplot(1,2,2)
plt.imshow(img3[:,:,1])
#%%
from image_io.analyze_format import read_analyze_xtr

xtr_file_old = 'C:/isbe/qbi/data/travastin/A002_AS/visit1/dynamic/dyn_20.xtr'
xtr_file_new = 'C:/isbe/qbi/data/milano_primovist/PRIMDCE_1/visit1/dynamic/dyn_20.xtr'
# %%
xtr_old = read_analyze_xtr(xtr_file_old)
xtr_new = read_analyze_xtr(xtr_file_new)

# %%
n_idx = 512000
buf_sz = struct.calcsize(str(n_idx)+'i')
buf = bytearray(buf_sz)
#struct.pack_into(str(n_idx)+'i', buf, 0, *list(range(n_idx)))
struct.pack_into(str(n_idx)+'i', buf, 0, *np.arange(n_idx))
out = np.array(struct.unpack_from(str(n_idx)+'i',buf, 0))

# %%
matches = [
    ('a', 'A', 'aa'),
    ('b', 'B', 'bb'),
    ('c', 'C', 'cc'),
]
#%%
def lookup(type_in, type_out, val):
    idx = [m[type_in] for m in matches].index(val)
    return matches[idx][type_out]


# %%
