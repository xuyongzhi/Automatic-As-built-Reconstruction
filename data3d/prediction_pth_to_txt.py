
from utils3d.bbox3d_ops import Bbox3D
import torch, os
import numpy as np

def pth_to_txt(pth_fn):
  '''
    class_2_label0 = {'background':0,'wall':1, 'window':2, 'door':3,
                    'ceiling':5, 'floor': 4, 'room':6}
  '''
  boxlist = torch.load(pth_fn)
  boxlist = [b.remove_low('scores',0.5) for b in boxlist]
  bboxes = [r.bbox3d for r in boxlist]
  labels = [r.get_field('labels') for r in boxlist]
  n = len(bboxes)
  path = os.path.dirname(pth_fn) + '/bims'
  if not os.path.exists(path):
    os.makedirs(path)
  for i in range(n):
    txt_fn = os.path.join(path,f'house_{i}.txt')
    if bboxes[i].shape[0] == 0:
      continue
    bboxes_revit = Bbox3D.gen_object_for_revit(bboxes[i].data.numpy(), is_yx_zb=True, labels=labels[i].reshape(-1,1) )
    np.savetxt(txt_fn, bboxes_revit)
    print(f'save {txt_fn} ok')
  pass

if __name__ == '__main__':
  path = '/home/z/Research/Detection_3D/RES/res_2g4c_Fpn4321_bs1_lr5_Tr50_CA/inference_3d/suncg_test_50_iou_3_augth_2'
  pth_fn = f'{path}/predictions.pth'
  pth_to_txt(pth_fn)
