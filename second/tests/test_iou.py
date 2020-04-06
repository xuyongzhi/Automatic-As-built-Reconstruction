from second.core.non_max_suppression.nms_gpu import rotate_iou_gpu, rotate_iou_gpu_eval
import numpy as np
from utils3d.bbox3d_ops import Bbox3D


def test_iou():
  angle = 0.1 * np.pi/180.0
  anchors_3d = [[0,0,0, 2, 6 ,4, angle],]
                #[1.3,0,0, 2,8,4,0],
                #[0,4,0, 2,8,4,0]]
  anchors_3d = np.array(anchors_3d, dtype=np.float32)
  #anchors_3d = Bbox3D.convert_to_yx_zb_boxes(anchors_3d)

  angle_1 = 0 * np.pi/180.0
  targets_3d = [[0,0,0, 2,8,4, angle_1]]
  targets_3d = np.array(targets_3d, dtype=np.float32)
  #targets_3d = Bbox3D.convert_to_yx_zb_boxes(targets_3d)


  anchors_2d = anchors_3d[:,[0,1,3,4,6]]
  targets_2d = targets_3d[:,[0,1,3,4,6]]
  # criterion=0: use anchors_2d as reference
  # criterion=1: use targets_2d as reference
  #ious = rotate_iou_gpu_eval(targets_2d, anchors_2d, criterion=2)
  #ious = rotate_iou_gpu_eval(anchors_2d, targets_2d, criterion=2)

  ious = rotate_iou_gpu_eval(anchors_2d.copy(), anchors_2d.copy(), criterion=2)

  print(anchors_2d)
  print(f'ious: {ious}')
  return

  boxes_3d = np.concatenate([anchors_3d, targets_3d], 0)
  labels = np.array([0]*anchors_3d.shape[0] + [1]*targets_3d.shape[0])
  Bbox3D.draw_bboxes(boxes_3d, up_axis='Z', is_yx_zb=False, labels=labels)
  pass


def test_iou2():
  angle0 = 0 * np.pi/180.0
  box0 = np.array([[0,0, 2, 6 , angle0]])
  angle1 = 0.1 * np.pi/180.0
  box1 = np.array([[0,0, 2, 6 , angle1]])

  #ious = rotate_iou_gpu_eval(box0, box0, criterion=2)
  ious = rotate_iou_gpu_eval(box1, box1, criterion=2)
  print(f'ious: {ious}')


def test_iou1():
  box0 = np.array([[ 4.38500402e+01, -4.00173668e+01,  1.36749997e+00,  2.58104093e+00,
                9.47311396e-02,  2.73499994e+00,  1.59986348e-02 ]])
  box1 = np.array([[ 4.38299566e+01, -4.00226317e+01,  1.36749997e+00,  2.62101683e+00,
                9.47311399e-02,  2.73499994e+00,  1.57467297e-02 ]])
  boxes = np.concatenate([box0, box1], 0)

  boxes = Bbox3D.convert_to_yx_zb_boxes(boxes)

  box0 =  boxes[0:1,[0,1,3,4,6]]
  box1 =  boxes[1:2,[0,1,3,4,6]]

  #box1[:,0:3] = 0
  #box1[:,3:6] = 1
  box1[:,-1] = 0.6

  ious = rotate_iou_gpu_eval( box1, box1 , criterion=-1)

  print(ious)
  Bbox3D.draw_bboxes(boxes, 'Z', True)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

if __name__ == '__main__':
    test_iou2()

