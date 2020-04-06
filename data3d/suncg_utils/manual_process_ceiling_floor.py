import os, sys
import numpy as np
from data3d.suncg_utils.suncg_preprocess import read_summary
from data3d.suncg_utils.celing_floor_room_preprocessing import preprocess_cfr

PARSED_DIR = '/DS/SUNCG/suncg_v1/parsed'

def rename_file():
  file_names = os.listdir(PARSED_DIR)
  file_names.sort()
  num = len(file_names)
  for i in range(num):
    fn = file_names[i]
    print(i)
    print(fn)
    path = os.path.join(PARSED_DIR, fn)
    summary = read_summary(path)
    if len(summary ) == 0:
      continue
    if summary['level_num'] != 1:
      continue

    box_path = os.path.join(path, 'object_bbox')
    src =os.path.join(box_path, 'ceiling.txt')
    if os.path.exists(src):
      dst =os.path.join(box_path, 'ceiling_raw.txt')
      os.rename(src, dst)
      print(f'{src}\n->\n{dst}\n\n')
    src =os.path.join(box_path, 'floor.txt')
    if os.path.exists(src):
      dst =os.path.join(box_path, 'floor_raw.txt')
      os.rename(src, dst)
      print(f'{src}\n->\n{dst}\n\n')
    pass

def process_ceiling_floor():
  file_names = os.listdir(PARSED_DIR)
  #file_names = ['0058113bdc8bee5f387bb5ad316d7b28']
  file_names.sort()
  num = len(file_names)
  for i in range(0,num):
      fn = file_names[i]
      print(i)
      print(fn)
      path = os.path.join(PARSED_DIR, fn)
      summary = read_summary(path)
      if len(summary ) == 0:
        continue
      if summary['level_num'] != 1:
        continue

      box_path = os.path.join(path, 'object_bbox')
      ceiling_fn =os.path.join(box_path, 'ceiling_raw.txt')
      floor_fn =os.path.join(box_path, 'floor_raw.txt')
      wall_fn =os.path.join(box_path, 'wall.txt')
      ceiling_raw = np.loadtxt(ceiling_fn).reshape([-1,7])
      floor_raw = np.loadtxt(floor_fn).reshape([-1,7])
      wall = np.loadtxt(wall_fn).reshape([-1,7])

      ceiling_new = preprocess_cfr(ceiling_raw, wall, 'ceiling')
      floor_new = preprocess_cfr(floor_raw, wall, 'floor')

      ceiling_fn =os.path.join(box_path, 'ceiling.txt')
      floor_fn =os.path.join(box_path, 'floor.txt')

      np.savetxt(ceiling_fn, ceiling_new)
      np.savetxt(floor_fn, floor_new)
      print(f'Save  {ceiling_fn}')
      pass



if __name__ == '__main__':
  #rename_file()
  process_ceiling_floor()
