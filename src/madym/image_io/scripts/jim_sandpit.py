#%%

%load_ext autoreload

%autoreload 2

import numpy as np
from image_io.jim_io import write_jim_roi_from_list, read_jim_roi
from image_io.analyze_format import write_analyze, read_analyze_img, read_analyze_hdr
import matplotlib.pyplot as plt
from skimage import measure

#%%
roi = [
       np.array(
        [ [0,0],
          [0,1],
          [1,0]]),
       np.array(
        [ [0,0],
          [0,2],
          [1,0]]),
       np.array(
        [ [0,0],
          [0,1],
          [1,1],
          [1,0]]),
       np.array(
        [ [0,0],
          [0,2],
          [1,1],
          [1,0]]),
       np.array(
        [ [0,0],
          [0,3],
          [1,0]])]

offsets = [0, 0.25, 1/3, 0.5, 0.75]

#%%
roi_slices = []
slice_num = 1
for slice in roi:
    for offset in offsets:
        roi_slices += [(slice+offset, slice_num)]
        slice_num+=1

#%%
write_jim_roi_from_list(roi_slices, 'Users/qbiuser/qbi/test.roi')
#%%
write_analyze(np.random.rand(8,8,25), 'testing/test.hdr', voxel_size=[1,1,1])
write_analyze(np.random.rand(9,9,25), 'testing/test1.hdr', voxel_size=[1,1,1])
#%%
jim_dir = 'Q:/data/MB/testing_jim/'
mask_py = read_jim_roi(jim_dir + 'test.roi', (8,8,25), [1,1,1])[0]
mask1_py = read_jim_roi(jim_dir + 'test.roi', (9,9,25), [1,1,1])[0]
mask_jim = read_analyze_img(jim_dir + 'test_roi.hdr')
mask1_jim = read_analyze_img(jim_dir + 'test1_roi.hdr')
#%%
mask_jim100 = read_analyze_img(jim_dir + 'test_roi100.hdr')
#mask1_jim = read_analyze_img(jim_dir + 'test1_roi.hdr')
#%%
mask_jim0 = read_analyze_img(jim_dir + 'test_roi0.hdr')
#mask1_jim = read_analyze_img(jim_dir + 'test1_roi.hdr')

# %%
for i in range(25):
  plt.figure()
  plt.subplot(1,2,1)
  plt.imshow(mask_jim0[:,:,i]>0)
  plt.plot(roi_slices[i][0][:,0]+3.5,roi_slices[i][0][:,1]+3.5)

  plt.subplot(1,2,2)
  plt.imshow(mask_py[:,:,i])
  plt.plot(roi_slices[i][0][:,0]+3.5,roi_slices[i][0][:,1]+3.5)
  plt.show()

#%%
for i in range(25):
  plt.figure()
  plt.subplot(1,2,1)
  plt.imshow(mask_jim[:,:,i]>0)
  plt.plot(roi_slices[i][0][:,0]+3.5,roi_slices[i][0][:,1]+3.5)

  plt.subplot(1,2,2)
  plt.imshow(mask_py[:,:,i])
  plt.plot(roi_slices[i][0][:,0]+3.5,roi_slices[i][0][:,1]+3.5)
  plt.show()

# %%

#%%
for roi_slice in roi_slices:
  contour = roi_slice[0]
  roi_mask = in_poly( (3,4), contour, res = 10)
  plt.figure()
  plt.imshow(roi_mask)
  for x in [0.5, 1.5, 2.5, 3.5]:
    plt.plot([x, x], [-0.5, 2.5], 'c')
  for y in [0.5, 1.5]:
    plt.plot([-0.5, 3.5], [y, y], 'c')

  for x in range(4):
    for y in range(3):
      plt.plot(x,y,'r*')

  plt.plot(
    np.concatenate((contour[:,1], contour[0:1,1]),0)-0.5,
    np.concatenate((contour[:,0], contour[0:1,0]),0)-0.5, 'r')
  plt.plot(contour[:,1]-0.5, contour[:,0]-0.5, 'g.')
  plt.colorbar()
  plt.show()

#%%
for roi_slice in roi_slices:
  contour = 10*roi_slice[0]
  roi_mask = measure.grid_points_in_poly( 
    (30,40), contour)
  plt.figure()
  plt.imshow(roi_mask)

  plt.plot(
    np.concatenate((contour[:,1], contour[0:1,1]),0),
    np.concatenate((contour[:,0], contour[0:1,0]),0), 'r')
  plt.plot(contour[:,1]-0.5, contour[:,0], 'g.')
  #plt.colorbar()
  plt.show()

# %%
def in_poly(shape, yx, res = 8):
  roi_mask_lrg1 = measure.grid_points_in_poly( 
    (res*shape[0],res*shape[1]), res*yx)

  yx2 = np.empty_like(yx)
  yx2[:,0] = shape[0]-yx[:,0]
  yx2[:,1] = shape[1]-yx[:,1]

  roi_mask_lrg2 = np.flip(np.flip(measure.grid_points_in_poly( 
    (res*shape[0],res*shape[1]), res*yx2), 1), 0)

  roi_mask_lrg = roi_mask_lrg1 | roi_mask_lrg2

  res2 = res*res
  roi_frac = np.empty(shape)
  for row in range(shape[0]):
    for col in range(shape[1]):
        roi_frac[row,col] = np.sum(
          roi_mask_lrg[(row*res):((row+1)*res), (col*res):((col+1)*res)]
        ) / res2
  roi_mask = roi_frac >= 0.5
  return roi_mask

# %%
for i in range(20):
  if np.any(mask_jim_rt[:,:,i]) or np.any(mask_py_rt[:,:,i]):
    diff = mask_jim_rt[:,:,i].astype(int) - mask_py_rt[:,:,i].astype(int)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(mask_jim_rt[:,:,i])
    plt.subplot(1,2,2)
    plt.imshow(diff)

# %%
contour = np.array([[-.5,-.5], [-.5, .5], [.5, .5], [.5, -.5]])

for offset in [0, 0.25, 0.5, 0.75, 1.0]:
  c = np.concatenate((contour[:,0:1], contour[:,1:]+offset),1)
  c = np.flip(c,0)
  mask = measure.grid_points_in_poly((1,2), c)

  plt.figure()
  plt.imshow(mask)
  plt.plot(c[:,1],c[:,0], 'r')

# %%
contour = np.array([[-1,-1], [0, -1], [0, 0], [-1, 0]])

roi_slices = []
slice_num = 1
for offset in [0, 0.25, 0.5, 0.75, 1.0]:
  c = np.concatenate((contour[:,0:1]+offset, contour[:,1:]),1)
  roi_slices += [(c, slice_num)]
  slice_num+=1
  c = np.concatenate((contour[:,0:1], contour[:,1:]+offset),1)
  roi_slices += [(c, slice_num)]
  slice_num+=1
  c = np.concatenate((contour[:,0:1]+offset, contour[:,1:]+offset),1)
  roi_slices += [(c, slice_num)]
  slice_num+=1
#%%        
write_jim_roi_from_list(roi_slices, jim_dir + 'rec_xy.roi') 
#%%
write_analyze(10*np.random.rand(2,1,5), jim_dir + 'rec_x.hdr')
write_analyze(10*np.random.rand(2,2,15), jim_dir + 'rec_xy.hdr')     


# %%
mask_jim_x = read_analyze_img(jim_dir + 'rec_x_mask.hdr')
mask_jim_xy = read_analyze_img(jim_dir + 'rec_xy_mask.hdr')

# %%
for i in range(5):
  plt.figure()
  plt.imshow(mask_jim_x[:,:,i]>0)
  plt.plot(roi_slices[3*i][0][:,0]+.5,roi_slices[3*i][0][:,1]+.5)

# %%
for i in range(15):
  cx = roi_slices[i][0][:,0]+1
  cy = roi_slices[i][0][:,1]+1

  cx2 = 1 - cx
  cy2 = 1 - cy

  #mask = measure.grid_points_in_poly((2,2), 
  #  np.concatenate( (cy.reshape(4,1), cx.reshape(4,1)) ,1))

  mask2 = measure.grid_points_in_poly((2,2), 
    np.concatenate( (cy2.reshape(4,1), cx2.reshape(4,1)) ,1))

  mask2 = np.flip(np.flip(mask2,1),0)

  mask = in_poly((2,2), 
    np.concatenate( (cy.reshape(4,1), cx.reshape(4,1)) ,1))

  plt.figure(figsize=(12,4))
  plt.subplot(1,3,1)
  plt.imshow(mask_jim_xy[:,:,i]>0)
  plt.plot(cx,cy)
  plt.plot(cx,cy, 'r*')

  plt.subplot(1,3,2)
  plt.imshow(mask)
  plt.plot(cx,cy)
  plt.plot(cx,cy, 'r*')

  plt.subplot(1,3,3)
  plt.imshow(mask2)
  plt.plot(cx,cy)
  plt.plot(cx,cy, 'r*')

# %%
mask_jim_rt = np.abs(read_analyze_img(
  'Q:/data/MB/Grace_RT/BALBC/010876/Visit3/T2W_mask_yw.hdr')) > 0

rt_hdr = read_analyze_hdr(
  'Q:/data/MB/Grace_RT/BALBC/010876/Visit3/T2W_mask_yw.hdr')

mask_py_rt = read_jim_roi(
  'Q:/data/MB/Grace_RT/BALBC/010876/Visit3/T2W_yw.roi',
  mask_jim_rt.shape, rt_hdr.spacing)[0]

union = mask_jim_rt | mask_py_rt
intersect = mask_jim_rt & mask_py_rt
dice = np.sum(intersect) / np.sum(union)
print(f'Num voxels Jim {np.sum(mask_jim_rt)}, py {np.sum(mask_py_rt)}'
  f', Dice overlap = {dice}')
# %%
for i in range(20):
  m1 = mask_jim_rt[:,:,i]
  m2 = mask_py_rt[:,:,i]
  if np.any(m1) or np.any(m2):
    xy1 = np.nonzero(m1 & ~m2)
    xy2 = np.nonzero(~m1 & m2)
    
    print(xy1)
    print(xy2)
    plt.figure()
    plt.imshow(m1)
    plt.plot(xy1[1], xy1[0], 'g*')
    plt.plot(xy2[1], xy2[0], 'r*')

# %%
mask_jim_rt = np.abs(read_analyze_img(
  'C:/isbe/qbi/data/milano_primovist/PRIMDCE_1/visit1/all_masks/dyn_mean_66-75_liver_mask_yw.hdr')) > 0

rt_hdr = read_analyze_hdr(
  'C:/isbe/qbi/data/milano_primovist/PRIMDCE_1/visit1/all_masks/dyn_mean_66-75_liver_mask_yw.hdr')

mask_py_rt = read_jim_roi(
  'C:/isbe/qbi/data/milano_primovist/PRIMDCE_1/visit1/roi_files/dyn_mean_66_75_liver_yw.roi',
  mask_jim_rt.shape, rt_hdr.spacing)[0]

union = mask_jim_rt | mask_py_rt
intersect = mask_jim_rt & mask_py_rt
dice = np.sum(intersect) / np.sum(union)
print(f'Num voxels Jim {np.sum(mask_jim_rt)}, py {np.sum(mask_py_rt)}'
  f', Dice overlap = {dice}')

# %%
for i in range(40):
  m1 = mask_jim_rt[:,:,i]
  m2 = mask_py_rt[:,:,i]
  if np.any(m1) or np.any(m2):
    xy1 = np.nonzero(m1 & ~m2)
    xy2 = np.nonzero(~m1 & m2)
    
    plt.figure()
    plt.imshow(m1)
    plt.plot(xy1[1], xy1[0], 'g*')
    plt.plot(xy2[1], xy2[0], 'r*')

# %%
