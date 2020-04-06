# xyz Oct 2019
import torch, math
from utils3d.geometric_torch import angle_with_x
from utils3d.geometric_torch import OBJ_DEF

_cx,_cy,_cz, _sx,_sy,_sz, _yaw = range(7)

def adjust_corner_order(boxes_2corners):
  corners0 = boxes_2corners[:,0:2]
  corners1 = boxes_2corners[:,2:4]
  centroids = (corners0 + corners1)/2
  aim0_dir = torch.ones(1,2, dtype=torch.float32, device=boxes_2corners.device)
  mask = ((corners0-centroids) * aim0_dir).sum(dim=1) > 0
  mask = mask.view(-1,1).to(torch.float32)
  corners0_ = corners0 * mask + corners1 * (1-mask)
  corners1_ = corners1 * mask + corners0 * (1-mask)
  boxes_2corners[:,0:2] = corners0_
  boxes_2corners[:,2:4] = corners1_
  return boxes_2corners

class Box3D_Torch():
  '''
      bbox yx_zb  : [xc, yc, z_bot, y_size, x_size, z_size, yaw-0.5pi]
      bbox standard: [xc, yc, zc, x_size, y_size, z_size, yaw]
      bbox corner : [x0, y0, x1, y1, z0, z1, thickness]
      y_size is thickness
  '''
  @staticmethod
  def from_2corners_to_yxzb(boxes, debug=0):
    assert boxes.dim() == 2
    assert boxes.shape[-1] % 7 == 0
    boxes = boxes.clone()
    k = -1
    if boxes.shape[-1] != 7:
      n = boxes.shape[0]
      k = boxes.shape[1] // 7
      boxes = boxes.view([-1,7])
    boxes_stand = Box3D_Torch.corner_box_to_standard(boxes, debug)
    boxes_yxzb = Box3D_Torch.convert_to_yx_zb_boxes(boxes_stand, debug)
    if k>0:
      boxes_yxzb = boxes_yxzb.view([n,k*7])
    return boxes_yxzb

  @staticmethod
  def corner_box_to_standard(boxes, debug=1):
    assert boxes.shape[1] == 7
    centroid = (boxes[:,[0,1,4]] + boxes[:,[2,3,5]])/2.0
    box_direction = boxes[:,[2,3]] - boxes[:,[0,1]]
    xsize = torch.norm( box_direction, dim=1 )
    xsize = xsize.view([-1,1])
    ysize = boxes[:,6:7]
    zsize = (boxes[:,5] - boxes[:,4]).view([-1, 1])
    yaw = angle_with_x( box_direction, scope_id=0, debug=debug ).view([-1,1])
    boxes_stand = torch.cat([centroid, xsize, ysize, zsize, yaw], 1)
    return boxes_stand

  @staticmethod
  def convert_from_yx_zb_boxes(boxes):
    '''
    Input
      bbox yx_zb  : [xc, yc, z_bot, y_size, x_size, z_size, yaw-0.5pi]
    Output
      bbox standard: [xc, yc, zc, x_size, y_size, z_size, yaw]

    The input is kitti lidar bbox format used in SECOND: x,y,z,w,l,h,orientation
      orientation=0: positive x of camera/car = negative lidar y -> car front face neg lidar y
      orientation = -pi/2: car face pos x of world -> clock wise rotation is positive
      orientation : (-pi,0]


    In my standard definition, bbox frame is same as world -> yaw=0. Also clock wise is positive.
    yaw = pi/2 is the orientation=0 status for yx_zb format of SECOND.
    yaw: (-pi/2,pi/2]

    yaw = orientation + pi/2

    The output format is the standard format I used in Bbox3D
    '''
    assert boxes.dim() == 2
    assert boxes.shape[1] == 7
    if boxes.shape[0] == 0:
      return boxes
    boxes = boxes.clone()
    boxes[:,2] += boxes[:,5]*0.5
    boxes = boxes[:,[0,1,2,4,3,5,6]]
    boxes[:,-1] += math.pi*0.5
    # limit in [-pi/2, pi/2]
    boxes[:,_yaw] = OBJ_DEF.limit_yaw(boxes[:,_yaw], False)
    OBJ_DEF.check_bboxes(boxes, False)
    return boxes

  @staticmethod
  def convert_to_yx_zb_boxes(boxes_standard, debug=0):
    '''
    Input
      bbox standard
    Output
      bbox yx_zb
    '''
    assert boxes_standard.dim() == 2
    assert boxes_standard.shape[1] == 7

    # This should be implemented in data prepration. For ceiling, floor, room,
    # temporaly performed here.
    #boxes = Bbox3D.define_walls_direction(boxes, 'Z', yx_zb=False, check_thickness=False)

    boxes = boxes_standard[:,[0,1,2,4,3,5,6]]
    boxes[:,2] = boxes[:,2] - boxes[:,5]*0.5
    boxes[:,-1] -= math.pi*0.5
    boxes[:,_yaw] = OBJ_DEF.limit_yaw(boxes[:,_yaw], True)
    OBJ_DEF.check_bboxes(boxes, True)
    return boxes

  @staticmethod
  def from_yxzb_to_2corners(boxes_in):
    from utils3d.bbox3d_ops import Bbox3D
    if boxes_in.shape[0] == 0:
      return boxes_in
    device = boxes_in.device
    boxes = boxes_in.clone().detach()
    boxes0 = boxes.cpu().data.numpy()
    zneg_corners, zpos_corners = Bbox3D.bboxes_corners_xz_central_surface(boxes0, is_yx_zb = True)
    zneg_corners = torch.from_numpy(zneg_corners).to(device)
    zpos_corners = torch.from_numpy(zpos_corners).to(device)
    c0x  = zneg_corners[:,0,0].view([-1,1])
    c0y  = zneg_corners[:,0,1].view([-1,1])
    c1x  = zneg_corners[:,1,0].view([-1,1])
    c1y  = zneg_corners[:,1,1].view([-1,1])

    z0  = zneg_corners[:,0,2].view([-1,1])
    z1  = zpos_corners[:,0,2].view([-1,1])
    th = boxes[:,3].view([-1,1])
    boxes_2corners = torch.cat([c0x, c0y, c1x, c1y, z0, z1, th], 1).to(device)
    boxes_2corners = adjust_corner_order(boxes_2corners)
    return boxes_2corners

def show_boxes_corners_boxes(boxes, is_yx_zb, boxes_corners):
  from utils3d.bbox3d_ops import Bbox3D
  corners0 = boxes_corners[:,[0,1,4]]
  corners1 = boxes_corners[:,[2,3,4]]
  corners = torch.cat([corners0, corners1], 0).data.numpy()
  Bbox3D.draw_points_bboxes(corners, boxes.data.numpy(), 'Z', is_yx_zb)

def box_dif(boxes0, boxes1):
  assert boxes0.dim() == boxes1.dim()
  boxes_dif = torch.abs( boxes1 - boxes0 ).view([-1,7])
  boxes_dif[:,-1] = torch.abs( OBJ_DEF.limit_yaw(boxes_dif[:,-1], yx_zb=True) )
  difv_s1 = torch.max(boxes_dif)
  assert difv_s1 < 1e-5
  return difv_s1


def test1():
  boxes_yxzb0 = torch.tensor( [[0.0780, 6.1881, 0.0496, 0.0947, 4.8724, 2.7350, 0.0000]] )
  boxes_2corners = Box3D_Torch.from_yxzb_to_2corners(boxes_yxzb0, True)
  boxes_yxzb_1 = Box3D_Torch.from_2corners_to_yxzb(boxes_2corners)
  print('Y1', box_dif( boxes_yxzb0, boxes_yxzb_1 ) )
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

def test():
  boxes_standard0 = torch.tensor( [[0.0,0,0, 8,1,3, math.pi * 0.15]] )
  #Bbox3D.show_bboxes_with_corners(boxes_standard0.data.numpy(), 'Z', False)

  boxes_standard0 = torch.rand( [4, 42] )
  boxes_standard0[:,[3]]  += 2
  boxes_standard0[:,[4]]  += 0.5
  boxes_standard0[:,[6]]  = (boxes_standard0[:,[6]]) * math.pi

  #boxes_standard0 =  \
  #    torch.tensor([[0.6371, 0.9343, 0.6529, 2.1351, 1.2135, 0.7307, 2.9768],
  #                  [0.5465, 0.9131, 0.3274, 2.0307, 0.6773, 0.7028, 0.7707]])
  #boxes_standard0 = boxes_standard0[1:]

  if boxes_standard0.shape[1] == 14:
    boxes_standard0[:,[10]]  += 2
    boxes_standard0[:,[13]]  = (boxes_standard0[:,[13]]-0.5) * math.pi
    boxes_standard0[:,[11]]  += 0.5

  n = boxes_standard0.shape[0]

  boxes_yxzb0 = Box3D_Torch.convert_to_yx_zb_boxes(boxes_standard0.view([-1, 7]))
  boxes_standard1 = Box3D_Torch.convert_from_yx_zb_boxes(boxes_yxzb0).view([n,-1])


  print('S1', box_dif( boxes_standard1, boxes_standard0 ) )

  boxes_2corners = Box3D_Torch.from_yxzb_to_2corners(boxes_yxzb0, is_yx_zb=True)
  #show_boxes_corners_boxes(boxes_yxzb0, True, boxes_2corners)
  boxes_standard2 = Box3D_Torch.corner_box_to_standard(boxes_2corners,debug=1).view([n,-1])
  #show_boxes_corners_boxes(boxes_standard2, False, boxes_2corners)

  boxes_yxzb_1 = Box3D_Torch.from_2corners_to_yxzb(boxes_2corners)
  boxes_standard3 = Box3D_Torch.corner_box_to_standard(boxes_2corners).view([n,-1])

  print('S2', box_dif( boxes_standard2, boxes_standard0 ) )
  print('S3', box_dif( boxes_standard3, boxes_standard0 ) )
  print('Y1', box_dif( boxes_yxzb0, boxes_yxzb_1 ) )

  #show_boxes_corners_boxes(boxes_standard3, False, boxes_2corners)

  #Bbox3D.draw_bboxes(boxes_standard0.view([-1,7]).data.numpy(), 'Z', False)
  #Bbox3D.draw_bboxes(boxes_standard2.view([-1,7]).data.numpy(), 'Z', False)

  #Bbox3D.draw_bboxes(boxes_yxzb0.view([-1,7]).data.numpy(), 'Z', True)
  #Bbox3D.draw_bboxes(boxes_yxzb_1.view([-1,7]).data.numpy(), 'Z', True)

  #print('standard\t', boxes_standard0)
  #print('yxzb\t\t', boxes_yxzb)
  #print('2corners\t', boxes_2corners)
  #print('yzxb 1\t', boxes_yxzb_1)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

if __name__ == '__main__':
  test()
