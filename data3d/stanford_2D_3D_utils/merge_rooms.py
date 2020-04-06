import os
import h5py
import numpy as np
import torch

STANFORD_PATH = '/DS/2D-3D-Semantics_Stanford/2D-3D-Semantics_Pth/'

if __name__ == "__main__":

    for AREA in [1,2,3,4,'5a','5b',6]:
      PTH_PATH = f'{STANFORD_PATH}/area_{AREA}'
      whole_area_path = f'{STANFORD_PATH}/whole_areas'
      if not os.path.exists(whole_area_path):
        os.makedirs(whole_area_path)
      whole_area_fn = f'{whole_area_path}/area_{AREA}.pth'

      room_fns = os.listdir(PTH_PATH)
      pcl = []
      bboxes = {}
      for rn in room_fns:
        fn = os.path.join(PTH_PATH, rn)
        pcl_room, bboxes_room = torch.load(fn)
        pcl.append(pcl_room)
        for obj in bboxes_room:
          if obj not in bboxes:
            bboxes[obj] = bboxes_room[obj]
          else:
            bboxes[obj] = np.concatenate([bboxes[obj], bboxes_room[obj]], 0)
      pcl = np.concatenate(pcl, 0)

      torch.save( (pcl,bboxes), whole_area_fn )
      print(f'area {AREA} save ok: \n\t{whole_area_fn}')



