# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp

def make_data_loader(cfg, is_train, is_distributed=False, start_iter=0):
  scale = cfg.SPARSE3D.VOXEL_SCALE
  full_scale=cfg.SPARSE3D.VOXEL_FULL_SCALE
  val_reps = cfg.SPARSE3D.VAL_REPS
  batch_size = cfg.SOLVER.IMS_PER_BATCH
  dimension=3

  # VALID_CLAS_IDS have been mapped to the range {0,1,...,19}
  VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

  def get_files(split):
    import os
    cur_path = os.path.dirname(os.path.abspath(__file__))
    dset_path = f'{cur_path}/ScanNetTorch'
    with open(f'{cur_path}/Benchmark_Small/scannetv1_{split}.txt') as f:
      scene_names = [l.strip() for l in f.readlines()]
    files = [f'{dset_path}/{scene}/{scene}_vh_clean_2.pth' for scene in scene_names]
    return files

  train,val=[],[]
  for x in torch.utils.data.DataLoader(
        get_files('train'),
          collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
      train.append(x)
  for x in torch.utils.data.DataLoader(
        get_files('val'),
          collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
      val.append(x)
  print('Training examples:', len(train))
  print('Validation examples:', len(val))

  #Elastic distortion
  blur0=np.ones((3,1,1)).astype('float32')/3
  blur1=np.ones((1,3,1)).astype('float32')/3
  blur2=np.ones((1,1,3)).astype('float32')/3
  def elastic(x,gran,mag):
      bb=np.abs(x).max(0).astype(np.int32)//gran+3
      noise=[np.random.randn(bb[0],bb[1],bb[2]).astype('float32') for _ in range(3)]
      noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
      noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
      noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
      noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
      noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
      noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
      ax=[np.linspace(-(b-1)*gran,(b-1)*gran,b) for b in bb]
      interp=[scipy.interpolate.RegularGridInterpolator(ax,n,bounds_error=0,fill_value=0) for n in noise]
      def g(x_):
          return np.hstack([i(x_)[:,None] for i in interp])
      return x+g(x)*mag

  def trainMerge(tbl):
      locs=[]
      feats=[]
      labels=[]
      for idx,i in enumerate(tbl):
          a,b,c=train[i] # a:xyz  b:color c:label
          m=np.eye(3)+np.random.randn(3,3)*0.1 # aug: position distortion
          m[0][0]*=np.random.randint(0,2)*2-1  # aug: x flip
          m*=scale
          theta=np.random.rand()*2*math.pi # rotation aug
          m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
          a=np.matmul(a,m)
          a=elastic(a,6*scale//50,40*scale/50)
          a=elastic(a,20*scale//50,160*scale/50)
          m=a.min(0)
          M=a.max(0)
          q=M-m
          # aug: the centroid between [0,full_scale]
          offset = -m + np.clip(full_scale-M+m-0.001, 0, None) * np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
          a+=offset
          idxs=(a.min(1)>=0)*(a.max(1)<full_scale)
          assert np.all(idxs), "some points are missed in train"
          a=a[idxs]
          b=b[idxs]
          c=c[idxs]
          a=torch.from_numpy(a).long()
          locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
          feats.append(torch.from_numpy(b)+torch.randn(3)*0.1)
          labels.append(torch.from_numpy(c))
      locs=torch.cat(locs,0)
      feats=torch.cat(feats,0)
      labels=torch.cat(labels,0)

      #batch_scopes(locs, scale)
      return {'x': [locs,feats], 'y': labels.long(), 'id': tbl}
  train_data_loader = torch.utils.data.DataLoader(
      list(range(len(train))),batch_size=batch_size, collate_fn=trainMerge, num_workers=20*0, shuffle=True)


  valOffsets=[0]
  valLabels=[]
  for idx,x in enumerate(val):
      valOffsets.append(valOffsets[-1]+x[2].size)
      valLabels.append(x[2].astype(np.int32))
  valLabels=np.hstack(valLabels)

  def valMerge(tbl):
      locs=[]
      feats=[]
      labels=[]
      point_ids=[]
      for idx,i in enumerate(tbl):
          a,b,c=val[i]
          m=np.eye(3)
          m[0][0]*=np.random.randint(0,2)*2-1
          m*=scale
          theta=np.random.rand()*2*math.pi
          m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
          a=np.matmul(a,m)+full_scale/2+np.random.uniform(-2,2,3)
          m=a.min(0)
          M=a.max(0)
          q=M-m
          offset=-m+np.clip(full_scale-M+m-0.001,0,None)*np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
          a+=offset
          idxs=(a.min(1)>=0)*(a.max(1)<full_scale)
          assert np.all(idxs), "some points are missed in val"
          a=a[idxs]
          b=b[idxs]
          c=c[idxs]
          a=torch.from_numpy(a).long()
          locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
          feats.append(torch.from_numpy(b))
          labels.append(torch.from_numpy(c))
          point_ids.append(torch.from_numpy(np.nonzero(idxs)[0]+valOffsets[i]))
      locs=torch.cat(locs,0)
      feats=torch.cat(feats,0)
      labels=torch.cat(labels,0)
      point_ids=torch.cat(point_ids,0)
      return {'x': [locs,feats], 'y': labels.long(), 'id': tbl, 'point_ids': point_ids}
  val_data_loader = torch.utils.data.DataLoader(
      list(range(len(val))),batch_size=batch_size, collate_fn=valMerge, num_workers=20,shuffle=True)

  if is_train:
    return train_data_loader
  else:
    return val_data_loader


def locations_to_position(locations, voxel_scale):
  return [location_to_position(loc, voxel_scale) for loc in locations]

def location_to_position(location, voxel_scale):
  assert location.shape[1] == 4
  return location[:,0:3].float() / voxel_scale

def batch_scopes(location, voxel_scale):
  batch_size = torch.max(location[:,3])+1
  s = 0
  e = 0
  scopes = []
  for i in range(batch_size):
    e += torch.sum(location[:,3]==i)
    xyz = location[s:e,0:3].float() / voxel_scale
    s = e.clone()
    xyz_max = xyz.max(0)[0]
    xyz_min = xyz.min(0)[0]
    xyz_scope = xyz_max - xyz_min
    print(f"min:{xyz_min}  max:{xyz_max} scope:{xyz_scope}")
    scopes.append(xyz_scope)
  scopes = torch.cat(scopes, 0)
  return scopes



