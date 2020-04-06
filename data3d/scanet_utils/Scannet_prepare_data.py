# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob, plyfile, numpy as np, multiprocessing as mp, torch, os

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper=np.ones(150)*(-100)
for i,x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]):
    remapper[x]=i

root_path = '/DT/SCANNET'
save_path = '/DS/ScanNet/SparseCNN'
files=sorted(glob.glob('%s/*/*_vh_clean_2.ply'%(root_path)))
files2=sorted(glob.glob('%s/*/*_vh_clean_2.labels.ply'%(root_path)))
assert len(files) == len(files2)

def f(fn):
    fn2 = fn[:-3]+'labels.ply'
    a=plyfile.PlyData().read(fn)
    v=np.array([list(x) for x in a.elements[0]]) # (81369, 7)
    coords=np.ascontiguousarray(v[:,:3]-v[:,:3].mean(0)) # (81369, 3)
    colors=np.ascontiguousarray(v[:,3:6])/127.5-1 #  colors
    a=plyfile.PlyData().read(fn2)
    label = np.array(a.elements[0]['label'])
    label = np.minimum(label, 149)
    w=remapper[label]
    fn_3 = (fn[:-4]+'.pth').replace(root_path, save_path)
    dir_3 = os.path.dirname(fn_3)
    if not os.path.exists(dir_3):
      os.makedirs(dir_3)
    torch.save((coords,colors,w),fn_3)
    print(fn, fn2, fn_3)

#f(files[0])

p = mp.Pool(processes=mp.cpu_count())
p.map(f,files)
p.close()
p.join()
