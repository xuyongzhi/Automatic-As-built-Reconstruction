# xyz Nov 2018
import numpy as np
import os,sys
import open3d
import numba
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from open3d_util import  draw_cus, gen_animation

from geometric_util import Rz as geo_Rz, angle_of_2lines, OBJ_DEF, angle_with_x

DEBUG = True

FRAME_SHOW = 1
POINTS_KEEP_RATE = 1.0
POINTS_SAMPLE_RATE = 1.0
BOX_XSURFACE_COLOR_DIF = False
CYLINDER_RADIUS = 0.02   # paper: 0.04

_cx,_cy,_cz, _sx,_sy,_sz, _yaw = range(7)
SameAngleThs = 0.01 * 6 # 0.01 rad = 0.6 degree
SameDisThs = 1e-3 * 50 # 0.1 mm


def same_a(x,y, threshold=SameDisThs):
  same0 = abs(x-y) < threshold
  return same0

def dif_rate(v0, v1):
  max_v =  max(abs(v0), abs(v1))
  if max_v==0:
    return 0
  return 1.0* abs(v1-v0) / max_v

def points_in_scope(points, scope):
  # point:[n,3]
  # scope:[2,3]
  c0 = points >= scope[0:1]
  c1 = points <= scope[1:2]
  inside = c0.all(1) * c1.all(1)
  return inside

def rotate_iou_gpu_eval_standbox(boxes, query_boxes, criterion=-1, device_id=0):
  '''
    The standard box, need to be converted as yx_zb before apply rotate_iou_gpu_eval
  '''
  from second.core.non_max_suppression.nms_gpu import rotate_iou_gpu, rotate_iou_gpu_eval
  boxes = Bbox3D.convert_to_yx_zb_boxes(boxes)
  query_boxes = Bbox3D.convert_to_yx_zb_boxes(query_boxes)
  return rotate_iou_gpu_eval(boxes, query_boxes, criterion, device_id)

def corners4_to_mesh2(corners, color=[255,0,0]):
  # corners: [n,4,3]
  assert corners.ndim == 3
  assert corners.shape[1:] == (4,3)
  n = corners.shape[0]
  triangles = np.array([[[0,1,2], [2,3,0]]]) # [1,2,3]
  triangles = np.tile(triangles, [n, 1, 1]) # [n,2,3]
  for i in range(n):
    triangles[i] += i*4
  corners = corners.reshape([-1,3])
  triangles = triangles.reshape([-1,3])

  mesh = open3d.geometry.TriangleMesh()
  mesh.vertices = open3d.utility.Vector3dVector(corners)
  mesh.triangles = open3d.utility.Vector3iVector(triangles)

  if color == 'random':
    color = np.random.sample(3)
  mesh.paint_uniform_color(color)
  return mesh


def down_sample_points(points0, sample_rate):
  n = points0.shape[0]
  indices = np.random.choice(n, int(n*sample_rate), replace=False)
  points = points0[indices]
  return points

def cut_points_roof(points, keep_rate=POINTS_KEEP_RATE, sample_rate=POINTS_SAMPLE_RATE):
  if points.shape[0] == 0:
    return points
  z_min = np.min(points[:,2])
  z_max = np.max(points[:,2])
  threshold = z_min + (z_max - z_min) * keep_rate
  mask = points[:,2] <= threshold
  points_cutted = points[mask]
  points_cutted = down_sample_points(points_cutted, sample_rate)
  return points_cutted

class Bbox3D():
  '''
      bbox standard: [xc, yc, zc, x_size, y_size, z_size, yaw]
      bbox yx_zb  : [xc, yc, z_bot, y_size, x_size, z_size, yaw-0.5pi]

      All the approaches here (In data generation) are designed for standard boxes, up_axis='Z'.
      The original up_axis from SUNCG is 'Y' (cam frame), but is already converted to 'Z' here.
      The boxes feed into networ is yx_zb, up_axis='Z'.
  '''
  _corners_tmp = np.array([ [0,0,0],[1,0,0],[0,1,0],[1,1,0],
                            [0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=np.float)
  #
  _xneg_vs = [0,2,6,4]
  _xpos_vs = [1,3,7,5]
  _yneg_vs = [0,1,5,4]
  _ypos_vs = [2,3,7,6]
  _zneg_vs = [0,1,3,2]
  _zpos_vs = [4,5,7,6]
  _face_vidxs = np.array([_xneg_vs, _xpos_vs, _yneg_vs, _ypos_vs, _zneg_vs, _zpos_vs])

  _lines_vids = np.array([[0,1],[0,2],[1,3],[2,3],
                        [4,5],[4,6],[5,7],[6,7],
                        [0,4],[1,5],[2,6],[3,7]] )
  _lines_z0_vids = np.array([[0,1],[0,2],[1,3],[2,3]])
  # positive x face lines:
  _x_pos_lines = [2,6,9,11]
  # positive z face lines
  _triangles_tmp = np.array( [[0,1,4],[0,2,4],[1,3,5],[2,3,6],
             [4,5,0],[4,6,2],[5,7,3],[6,7,2],
             [0,4,6],[1,5,7],[2,6,4],[3,7,1]] )


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
    boxes = boxes.copy().reshape([-1,7])
    if boxes.shape[0] == 0:
      return boxes
    boxes[:,2] += boxes[:,5]*0.5
    boxes = boxes[:,[0,1,2,4,3,5,6]]
    boxes[:,-1] += np.pi*0.5
    # limit in [-pi/2, pi/2]
    boxes[:,_yaw] = OBJ_DEF.limit_yaw(boxes[:,_yaw], False)
    OBJ_DEF.check_bboxes(boxes, False)
    return boxes

  @staticmethod
  def convert_to_yx_zb_boxes(boxes):
    '''
    Input
      bbox standard
    Output
      bbox yx_zb
    '''
    assert boxes.copy().shape[1] == 7

    # This should be implemented in data prepration. For ceiling, floor, room,
    # temporaly performed here.
    #boxes = Bbox3D.define_walls_direction(boxes, 'Z', yx_zb=False, check_thickness=False)

    boxes = boxes[:,[0,1,2,4,3,5,6]]
    boxes[:,2] = boxes[:,2] - boxes[:,5]*0.5
    boxes[:,-1] -= np.pi*0.5
    boxes[:,_yaw] = OBJ_DEF.limit_yaw(boxes[:,_yaw], True)
    OBJ_DEF.check_bboxes(boxes, True)
    return boxes

  @staticmethod
  def set_yaw_zero(boxes):
    '''
    For object like ceiling, floor, room, which are symmetry about both x_b and y_b.
    Always use yaw==0, length = size along x_r, thickness = size along x_y
    yaw is times of pi/2
    '''
    if boxes.shape[0] == 0:
      return boxes
    yaws = boxes[:,-1]
    assert np.mod( yaws, np.pi/2 ).max() < 0.01
    switch_lt = np.abs(yaws / (np.pi/2)).astype(np.int)
    size_y = boxes[:,3] * (1-switch_lt) + boxes[:,4] * (switch_lt)
    size_x = boxes[:,4] * (1-switch_lt) + boxes[:,3] * (switch_lt)
    boxes[:,3] = size_y
    boxes[:,4] = size_x
    boxes[:,-1] = 0
    return boxes

  @staticmethod
  def boxes_size(boxes, up_axis='Z'):
      corners = Bbox3D.bboxes_corners(boxes, up_axis).reshape([-1,3])
      xyz_max = corners.max(0)
      xyz_min = corners.min(0)
      xyz_size = xyz_max - xyz_min
      return xyz_size

  @staticmethod
  def video(pcds):
      def rotate_view(vis):
          ctr = vis.get_view_control()
          ctr.rotate(10, 0)
          return False

      open3d.visualization.draw_geometries_with_animation_callback(pcds,
                                                              rotate_view)

  @staticmethod
  def draw_points_open3d(points, color=[0,1,1], show=False, points_keep_rate=POINTS_KEEP_RATE, points_sample_rate=POINTS_SAMPLE_RATE):
    points = cut_points_roof(points, points_keep_rate, points_sample_rate)
    pcl = open3d.geometry.PointCloud()
    pcl.points = open3d.utility.Vector3dVector(points[:,0:3])
    if points.shape[1] >= 6:
      pcl.colors = open3d.utility.Vector3dVector(points[:,3:6])
    else:
      pcl.paint_uniform_color(color)
    if points.shape[1] >= 9:
      pcl.normals = open3d.utility.Vector3dVector(points[:,6:9])
    if show:
      #open3d.draw_geometries([pcl])
      draw_cus([pcl])
    return pcl

  @staticmethod
  def draw_points(points, color=[0,1,1], points_keep_rate=POINTS_KEEP_RATE, points_sample_rate=POINTS_SAMPLE_RATE, animation_fn=None, ani_size=None):
    pcds = Bbox3D.draw_points_open3d(points, color, show=True, points_keep_rate=points_keep_rate, points_sample_rate=points_sample_rate)
    if animation_fn is not None:
      gen_animation([pcds], animation_fn, ani_size)

  @staticmethod
  def draw_points_bboxes(points, gt_boxes0, up_axis, is_yx_zb, labels=None, names=None, lines=None, random_color=True,
                         points_keep_rate=POINTS_KEEP_RATE, points_sample_rate=POINTS_SAMPLE_RATE, animation_fn=None, ani_size=None, box_colors=None):
    '''
    points, gt_boxes0, up_axis, is_yx_zb, labels=None, names=None, lines=None)
    '''
    if points is not None:
      pcl = Bbox3D.draw_points_open3d(points, points_keep_rate=points_keep_rate, points_sample_rate=points_sample_rate)

    bboxes_lineset_ls = Bbox3D.bboxes_lineset(gt_boxes0, up_axis, is_yx_zb, labels, names, random_color, box_colors)

    if lines is not None:
      lineset = [Bbox3D.draw_lines_open3d(lines)]
    else:
      lineset = []
    if points is not None:
      #open3d.draw_geometries(bboxes_lineset_ls + [pcl] + lineset)
      pcds = bboxes_lineset_ls + [pcl] + lineset
    else:
      #open3d.draw_geometries(bboxes_lineset_ls + lineset)
      pcds = bboxes_lineset_ls + lineset

    draw_cus(pcds)

    if animation_fn is not None:
      gen_animation(pcds, animation_fn, ani_size)

  @staticmethod
  def draw_points_bboxes_mesh(points, gt_boxes0, up_axis, is_yx_zb, labels=None, names=None, lines=None,
                              points_keep_rate=POINTS_KEEP_RATE, points_sample_rate=POINTS_SAMPLE_RATE, animation_fn=None, ani_size=None, random_color=False, box_colors=None):
    mesh = Bbox3D.bboxes_mesh(gt_boxes0, up_axis, is_yx_zb, labels, names, random_color=random_color, box_colors=box_colors)
    #Bbox3D.video(mesh)
    if points is not None:
      pcl = Bbox3D.draw_points_open3d(points, points_keep_rate=points_keep_rate, points_sample_rate=points_sample_rate)
      mesh.append(pcl)
    draw_cus(mesh)
    if animation_fn is not None:
      gen_animation(mesh, animation_fn, ani_size)

  @staticmethod
  def draw_bboxes(gt_boxes0, up_axis, is_yx_zb, labels=None, names=None, random_color=True, highlight_ids=None):
    if highlight_ids is not None:
        assert labels is None
        labels = np.ones([gt_boxes0.shape[0]], dtype=np.int32)
        labels[highlight_ids] = 0
    bboxes_lineset_ls = Bbox3D.bboxes_lineset(gt_boxes0, up_axis, is_yx_zb, labels, names, random_color=random_color)
    draw_cus(bboxes_lineset_ls)

  @staticmethod
  def bboxes_lineset(gt_boxes0, up_axis, is_yx_zb, labels=None, names=None, random_color=True, colors=None):
    from color_list import COLOR_LIST
    gt_boxes0 = gt_boxes0.reshape([-1,7])
    #gt_boxes0 = np.array([gtb for gtb in gt_boxes0 if gtb[3]>=0])
    if colors is not None:
      assert colors.shape[0] == gt_boxes0.shape[0]
    #else:
    #  if labels is not None:
    #    ml = labels.max() + 1
    #    colors  = COLOR_LIST[0:ml]
    #    print('colors used: {colors}')

    gt_boxes1 = gt_boxes0.copy()
    if is_yx_zb:
      gt_boxes1 = Bbox3D.convert_from_yx_zb_boxes(gt_boxes1)
    bn = gt_boxes1.shape[0]

    bbox_meshes = []
    if bn > COLOR_LIST.shape[0]:
      print(f'bn={bn} > {COLOR_LIST.shape[0]}')
      random_color = False
    for i in range(bn):
        box = gt_boxes1[i]
        if colors is not None:
          color = colors[i]
        else:
          if random_color:
            color = COLOR_LIST[i]
          else:
            color = [1,0,0]
            if labels is not None:
                color = COLOR_LIST[labels[i]]
        bbox_meshes.append( Bbox3D.get_one_bbox(box, up_axis, color=color) )

    if bn > 0:
      bboxes_lineset = bbox_meshes[0]
      for i in range(1, bn):
        bboxes_lineset += bbox_meshes[i]
    else:
      bboxes_lineset = None

    #if names is not None:
    #  print("names:", names)
    #print('boxes:\n', gt_boxes1)

    if bn > 0:
      out = [bboxes_lineset]
      if FRAME_SHOW == 1:
        mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.6, origin = [0,0,0])
        out = out + [mesh_frame]
      elif FRAME_SHOW == 2:
        mesh_frame = open3d.create_mesh_coordinate_frame(size = 0.6, origin = gt_boxes1[0,0:3])
        out = out + [mesh_frame]
      return out
    else:
      return []


  @staticmethod
  def bboxes_mesh(boxes0, up_axis, is_yx_zb, labels=None, names=None, random_color=False, box_colors=None):
    from color_list import COLOR_LIST
    assert boxes0.ndim == 2
    if boxes0.shape[0] == 0:
      return []
    corners = Bbox3D.bboxes_corners(boxes0, up_axis, is_yx_zb)
    faces_corners = np.take(corners, Bbox3D._face_vidxs, axis=1)
    n = boxes0.shape[0]
    mesh = []
    if box_colors is not None:
      colors = box_colors
    else:
      if labels is None or random_color:
          colors = COLOR_LIST[0:n]
      else:
          colors = COLOR_LIST[labels]
    for i in range(n):
      mesh_i = corners4_to_mesh2(faces_corners[i].reshape([-1,4,3]), colors[i])
      mesh.append( mesh_i)
    return mesh

  @staticmethod
  def draw_bboxes_mesh(boxes0, up_axis, is_yx_zb, labels=None, names=None):
    mesh = Bbox3D.bboxes_mesh(boxes0, up_axis, is_yx_zb, labels, names)
    draw_cus(mesh)

  @staticmethod
  def draw_points_lines(points, lines, color=[0,0,0], show=False):
    pcl = Bbox3D.draw_points_open3d(points)
    line_set = Bbox3D.draw_lines_open3d(lines, color)
    draw_cus([pcl, line_set])

  @staticmethod
  def draw_lines_open3d(lines, color=[0,0,0], show=False):
    '''
    lines: [n,2,3]
    '''
    assert lines.ndim == 3
    nl = lines.shape[0]
    assert lines.shape[1] == 2
    if lines.shape[2] == 2:
      lines = np.concatenate([lines, np.zeros([nl,2,1])], 2)
    assert lines.shape[2]==3
    lines = lines.reshape([-1,3])
    line_set = open3d.LineSet()
    line_set.points = open3d.utility.Vector3dVector(lines)
    lines_vids = np.arange(lines.shape[0]).reshape([-1,2])
    line_set.lines = open3d.Vector2iVector(lines_vids)

    colors = [color for i in range(nl)]
    line_set.colors = open3d.utility.Vector3dVector(colors)

    if show:
      draw_cus([line_set])
      #open3d.draw_geometries([line_set])
    return line_set

  @staticmethod
  def get_one_bbox(bbox, up_axis, plyfn=None, color=[1,0,0]):
    return Bbox3D.get_1bbox_mesh(bbox, up_axis, plyfn, color)
    #return Bbox3D.get_1bbox_lineset(bbox, up_axis, plyfn, color)

  @staticmethod
  def get_1bbox_mesh(bbox, up_axis, plyfn=None, color=[1,0,0], radius=CYLINDER_RADIUS):
    assert bbox.shape == (7,)
    corners = Bbox3D.bbox_corners(bbox, up_axis)
    lines = np.take(corners, Bbox3D._lines_vids, axis=0)
    centroids = lines.mean(1)
    #angles = angle_with_x(directions[:,:2])
    directions = lines[:,1,:]-lines[:,0,:]
    heights = np.linalg.norm(directions,axis=1)
    directions = directions/heights.reshape([-1,1])
    mesh = []
    for i in range(12):
        cylinder_i = open3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=heights[i])
        cylinder_i.paint_uniform_color(color)
        transformation = np.identity(4)
        transformation[:3,3] = centroids[i]
        transformation[:3,2] = directions[i]
        cylinder_i.transform(transformation)
        mesh.append(cylinder_i)

    cm = mesh[0]
    for i in range(1,12):
      cm += mesh[i]
    return cm

  def get_1bbox_lineset(bbox, up_axis, plyfn=None, color=[1,0,0]):
    '''
    only one box
    '''
    assert bbox.shape == (7,)
    corners = Bbox3D.bbox_corners(bbox, up_axis)
    colors = [color for i in range(len(Bbox3D._lines_vids))]
    if BOX_XSURFACE_COLOR_DIF:
        for i in Bbox3D._x_pos_lines:
            colors[i] = [0,0,1]
    line_set = open3d.LineSet()
    line_set.points = open3d.utility.Vector3dVector(corners)
    line_set.lines = open3d.Vector2iVector(Bbox3D._lines_vids)
    line_set.colors = open3d.utility.Vector3dVector(colors)

    if plyfn!=None:
      Bbox3D.save_bbox_ply(plyfn, bbox, up_axis, color)

    #print('bbox:\n',bbox)
    #print('corners:\n',corners)
    #mesh_frame = open3d.create_mesh_coordinate_frame(size = 1.0, origin = np.mean(corners, 0))
    #open3d.draw_geometries([line_set, mesh_frame])
    return line_set

  @staticmethod
  def save_bboxes_ply(plyfn, bboxes, up_axis, color=[1,0,0]):
    for i,box in enumerate(bboxes):
      tmp1, tmp2 = os.path.splitext(plyfn)
      plyfn_i = '%s_%d%s'%(tmp1, i, tmp2)
      Bbox3D.save_bbox_ply(plyfn_i, box, up_axis, color)
  @staticmethod
  def save_bbox_ply(plyfn, bbox, up_axis, color=[1,0,0]):
    from plyfile import PlyData, PlyElement
    #*************************************
    corners = Bbox3D.bbox_corners(bbox, up_axis)
    lines = np.array(Bbox3D._lines_vids)
    colors = np.array([color for i in range(lines.shape[0])])
    #*************************************
    num_vertex = corners.shape[0]
    vertex = np.zeros( shape=(num_vertex) ).astype([('x', 'f8'), ('y', 'f8'),('z', 'f8')])
    for i in range(num_vertex):
        vertex[i] = ( corners[i,0], corners[i,1], corners[i,2] )

    el_vertex = PlyElement.describe(vertex,'vertex')

    #*************************************
    edge = np.zeros( shape=(lines.shape[0]) ).astype(
                    dtype=[('vertex1', 'i4'), ('vertex2','i4'),
                           ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    num_line = lines.shape[0]
    for i in range(num_line):
        edge[i] = ( lines[i,0], lines[i,1], colors[i,0], colors[i,1], colors[i,2] )
    el_edge = PlyElement.describe(edge,'edge')

    dirname = os.path.dirname(plyfn)
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    PlyData([el_vertex, el_edge],text=True).write(plyfn)
    print('write %s ok'%(plyfn))

  @staticmethod
  def Unused_draw_bbox_open3d_mesh(bbox, up_axis, color=[1,0,0], plyfn=None):
    '''
      box_min: [3]
      box_max: [3]
    '''
    assert bbox.shape == (7,)
    corners = Bbox3D.bbox_corners(bbox, up_axis)

    mesh = open3d.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(corners)
    mesh.triangles = open3d.Vector3iVector(Bbox3D._triangles_tmp)
    mesh.paint_uniform_color(color)
    #open3d.draw_geometries([mesh])

    if plyfn is not None:
      open3d.write_triangle_mesh(plyfn, mesh,write_ascii=True)
    return mesh

  @staticmethod
  def bbox_from_minmax(bbox_min_max):
    bmin = np.array(bbox_min_max['min'])
    bmax = np.array(bbox_min_max['max'])
    centroid = (bmin + bmax) / 2.0
    lwh = bmax - bmin
    rotation = np.array([0])
    bbox = np.concatenate([centroid, lwh, rotation])
    return bbox

  @staticmethod
  def bbox_corners(bbox, up_axis):
    '''
    for yaw, clock wise is positive.
    In Geou.Rz, anticlock wise is positive. But by:
      corners = (np.matmul(R, (corners-bsize*0.5).T )).T + bsize*0.5
    do not use R.transpose(), it is changed to clock wise.
    '''
    assert bbox.shape == (7,)
    cx,cy,cz, sx,sy,sz, yaw = bbox
    centroid = bbox[0:3]
    bsize = bbox[3:6]

    ## corners aligned
    corners = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                [0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=np.float)
    corners[:,0] *= bsize[0]
    corners[:,1] *= bsize[1]
    corners[:,2] *= bsize[2]

    # rotate corners
    if yaw!=0:
      R = Bbox3D.get_yaw_R(bbox, up_axis)
      corners = (np.matmul(R, (corners-bsize*0.5).T )).T + bsize*0.5
      corners = corners.astype(bbox.dtype)

    zero = centroid - bsize*0.5
    corners += zero
    return corners

  @staticmethod
  def bboxes_corners(bboxes, up_axis, is_yx_zb=False):
    assert bboxes.ndim==2 and bboxes.shape[1]==7
    if is_yx_zb:
      bboxes = Bbox3D.convert_from_yx_zb_boxes(bboxes)
    corners = np.array( [Bbox3D.bbox_corners(box, up_axis) for box in bboxes], dtype=np.float32 )
    return corners

  @staticmethod
  def bboxes_centroid_lines(bboxes, cen_axis, up_axis):
    '''
      in:
        bboxes: [n,7]
        axis: 'X'/'Y'/'Z'
        up_axis: 'Y'/'Z'
      out:
        centroid_lines: [n, 2,3]
    '''
    if bboxes.shape[0] == 0:
        return np.empty([0,2,3])
    corners = Bbox3D.bboxes_corners(bboxes, up_axis)
    if cen_axis == 'X':
      neg_vs = Bbox3D._xneg_vs
      pos_vs = Bbox3D._xpos_vs
    elif cen_axis == 'Y':
      neg_vs = Bbox3D._yneg_vs
      pos_vs = Bbox3D._ypos_vs
    elif cen_axis == 'Z':
      neg_vs = Bbox3D._zneg_vs
      pos_vs = Bbox3D._zpos_vs
    else:
        raise NotImplementedError
    negc = corners[:,neg_vs].mean(1, keepdims=True)
    posc = corners[:,pos_vs].mean(1, keepdims=True)
    centroid_lines = np.concatenate([negc, posc], 1)
    return centroid_lines

  @staticmethod
  def gen_object_for_revit(bboxes, is_yx_zb, labels):
    '''
    Object type for revit:
      x_corner0, y_corner0, x_corner1, y_corner1, z_centroid, thickness, height,   +   label, wall_id
    '''
    if is_yx_zb:
      bboxes = Bbox3D.convert_from_yx_zb_boxes(bboxes)
    centroid_lines = Bbox3D.bboxes_centroid_lines(bboxes, 'X', 'Z')
    n = bboxes.shape[0]
    bboxes_new = np.zeros([n,9])
    centroid_lines_1 = centroid_lines[:,:,0:2].reshape([n,-1])
    bboxes_new[:,0:4] = centroid_lines_1
    bboxes_new[:,[4,5,6]] = bboxes[:,[2,4,5]]
    bboxes_new[:,7] = labels.squeeze()
    bboxes_new[:,8] = 0

    wall_ids = np.where(labels == 1)[0]
    other_ids = np.where(labels != 1)[0]
    walls = bboxes_new[wall_ids]
    other_bims = bboxes_new[other_ids]

    other_bims[:,0:7] = bboxes[other_ids]

    dif = bboxes[wall_ids, 0:3].reshape([1,-1,3]) - bboxes[other_ids, 0:3].reshape(-1,1,3)
    dis = np.linalg.norm(dif, axis=2)
    wall_ids = dis.argmin(axis=1)
    other_bims[:,8] = wall_ids

    bims = np.concatenate([walls, other_bims], 0)
    return bims

  @staticmethod
  def bboxes_corners_xz_central_surface(bboxes, up_axis='Z', is_yx_zb=False):
    '''
      in:
        bboxes: [n,7]
        axis: 'X'/'Y'/'Z'
        up_axis: 'Y'/'Z'
      out:
        zpos_corners: [n, 2,3]
        zneg_corners: [n, 2,3]
    '''
    if bboxes.shape[0] == 0:
        return np.empty([0,2,3]), np.empty([0,2,3])
    corners = Bbox3D.bboxes_corners(bboxes, up_axis, is_yx_zb)
    cen_axis = 'Y'
    if cen_axis == 'X':
      neg_vs = Bbox3D._xneg_vs
      pos_vs = Bbox3D._xpos_vs
    elif cen_axis == 'Y':
      neg_vs = Bbox3D._yneg_vs
      pos_vs = Bbox3D._ypos_vs
    elif cen_axis == 'Z':
      neg_vs = Bbox3D._zneg_vs
      pos_vs = Bbox3D._zpos_vs
    else:
        raise NotImplementedError
    negc = corners[:,neg_vs]
    posc = corners[:,pos_vs]
    cen_corners = (negc + posc)/2.0
    zneg_corners = cen_corners[:,[0,1],:]
    zpos_corners = cen_corners[:,[2,3],:]
    return zneg_corners, zpos_corners

  @staticmethod
  def point_in_box(points, bboxes, up_axis='Z'):
    cen_lines_x = Bbox3D.bboxes_centroid_lines(bboxes, 'X', up_axis)
    cen_lines_y = Bbox3D.bboxes_centroid_lines(bboxes, 'Y', up_axis)
    dis_x = vertical_dis_points_lines(points, cen_lines_x)
    y_inside = dis_x < bboxes[:,4]
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

  @staticmethod
  def bbox_face_centroids(bbox, up_axis):
    corners = Bbox3D.bbox_corners(bbox, up_axis)
    faces = []
    for i in range(6):
      vidxs = Bbox3D._face_vidxs[i]
      face_i = np.mean(corners[vidxs, :], 0, keepdims=True)
      faces.append(face_i)
    faces = np.concatenate(faces, 0)
    return faces


  @staticmethod
  def get_yaw_R(bbox, up_axis):
    yaw = bbox[_yaw]
    import geometric_util as Geou
    if up_axis=='Y':
      R = Geou.Ry(yaw)
    elif up_axis == 'Z':
      R = Geou.Rz(yaw)
    else:
      raise NotImplementedError
    return R

  @staticmethod
  def merge_two_bbox(bbox0, bbox1, up_axis):
    yaw0 = bbox0[_yaw]
    yaw1 = bbox1[_yaw]
    assert abs(yaw0-yaw1) < SameAngleThs
    yaw = (yaw0+yaw1)/2.0

    centroid_new = (bbox0[0:3] + bbox1[0:3])*0.5
    tmp = bbox0.copy()
    tmp[-1] = 0
    corners0 = Bbox3D.bbox_corners(tmp, up_axis) - centroid_new
    tmp = bbox1.copy()
    tmp[-1] = 0
    corners1 = Bbox3D.bbox_corners(tmp, up_axis) - centroid_new
    corners_new = np.maximum(corners0, corners1) * Bbox3D._corners_tmp
    corners_new += np.minimum(corners0, corners1) * (1-Bbox3D._corners_tmp)

    sx = corners_new[1,0] - corners_new[0,0]
    sy = corners_new[2,1] - corners_new[0,1]
    sz = corners_new[-1,2] - corners_new[0,2]

    cx,cy,cz = centroid_new
    bbox_new = np.array([cx,cy,cz, sx,sy,sz, yaw])
    return bbox_new


  @staticmethod
  def define_walls_direction(boxes, up_axis, yx_zb, check_thickness=False):
    show = False
    if show:
      box_ls0 = np.concatenate([bx.reshape([-1,7]) for bx in boxes], 0)
      box_ls0[:,0] += 20
      #Bbox3D.draw_bboxes(box_ls0, up_axis, False)

    bn = len(boxes)
    for i in range(bn):
      boxes[i] = Bbox3D.define_wall_direction(boxes[i], up_axis, yx_zb, check_thickness)
      boxes[i][-1] = OBJ_DEF.limit_yaw(boxes[i][-1], yx_zb)

    if show:
      box_ls1 = np.concatenate([bx.reshape([-1,7]) for bx in boxes], 0)
      #indices = np.where(np.abs(box_ls1[:,-1]) > 0.1)[0]
      #box_ls0 = box_ls0[indices, :]
      #box_ls1 = box_ls1[indices, :]

      box_ls1 = np.concatenate([box_ls0, box_ls1], 0)
      Bbox3D.draw_bboxes(box_ls1, up_axis, False)
    return boxes

  @staticmethod
  def define_wall_direction(box, up_axis, yx_zb, check_thickness):
    '''
      bbox standard: [xc, yc, zc, x_size, y_size, z_size, yaw]

      up_axis='Z', make always  x_size > y_size, y_size is thickness
      (1) x_size > y_size, no modify
      (2) x_size < y_size, switch x_size and y_size, yaw += pi/2

      up_axis='Y', make always  x_size > z_size, z_size is thickness
      (1) x_size > z_size, no modify
      (2) x_size < z_size, switch x_size and z_size, yaw += pi/2

      yaw (-pi/2, pi/2]
    '''
    assert box.shape == (7,)
    assert up_axis == 'Z'
    assert yx_zb == False, "the rule for yx_zb is different"
    if up_axis == 'Y':
      _up = 2+3 # z is thickness dim
    if up_axis == 'Z':
      _up = 1+3 # y is thickness dim
    yaw0 = box[_yaw]
    if box[_sx] < box[_up]:
      tmp = box[_sx].copy()
      box[_sx] = box[_up].copy()
      box[_up] = tmp
      is_increase = int(yaw0<0)*2 -1
      box[_yaw] = yaw0 + np.pi * 0.5 * is_increase
      pass

    #if not box[3] >= box[_up]:
    #  Bbox3D.draw_bboxes(box, 'Z', False)
    #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
    #  pass
    assert box[3] >= box[_up]
    if check_thickness:
      assert box[_up] < 0.3 # normally thickness is small
    return box


  @staticmethod
  def line_surface_intersection(line, surface, crop):
    v0,v1 = line
    v01 = v1-v0
    idx, value = surface
    if v01[idx]==0:
      # no intersection between line and surface
      return line, False
    k = (value-v0[idx]) / (v01[idx])
    if k<0 or k>1:
      # intersection is out of range of the line
      return line, False
    intersection = v0 + v01 * k

    inversed = int(v0[idx] > v1[idx])
    if crop == 'min':
      line[0+inversed] = intersection
    else:
      line[1-inversed] = intersection
    return line, True


  @staticmethod
  def points_in_bbox(points, bboxes):
    '''
    Input:
      points:[m,3]
      bbox standard: [n,7] [xc, yc, zc, x_size, y_size, z_size, yaw]

    '''
    from second.core.box_np_ops import center_to_corner_box3d, corner_to_surfaces_3d
    from second.core.geometry import points_in_convex_polygon_3d_jit
    assert points.ndim == 2 and points.shape[1]==3 and bboxes.ndim == 2 and bboxes.shape[1]==7
    origin = [0.5,0.5,0]
    h_axis = 2
    bboxes = Bbox3D.convert_to_yx_zb_boxes(bboxes.copy())

    bbox_corners = center_to_corner_box3d(
        bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6], origin=origin, axis=h_axis)
    surfaces = corner_to_surfaces_3d(bbox_corners)
    point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)

    show = False
    if show and DEBUG:
      print(point_masks)
      Bbox3D.draw_points_bboxes(points, bboxes, 'Z', is_yx_zb=True)

    return point_masks

  @staticmethod
  def detect_intersection_corners(bbox0, bboxes_others, up_axis):
    '''
    Find if the start or end of bbox0 is inside of bboxes_others.
    Do not care if the inner points of bbox0 is inside of bboxes_others.
    can be used to find if x begin or end should be freezed
    '''
    assert up_axis == 'Z'
    assert bbox0.shape == (7,)
    assert bboxes_others.ndim == 2
    assert bboxes_others.shape[1] == 7
    corners0 = Bbox3D.bbox_corners(bbox0, up_axis)
    xneg_corners0 = np.mean(corners0[Bbox3D._xneg_vs], 0, keepdims=True)
    xpos_corners0 = np.mean(corners0[Bbox3D._xpos_vs], 0, keepdims=True)

    direction = np.expand_dims(bbox0[0:3], 0) - xneg_corners0
    direction = direction / np.linalg.norm(direction)
    offset = direction * 4e-3

    # corners on the negative direction of x
    xneg_corners = [xneg_corners0 + offset*i for i in range(10)]
    xneg_corners = np.concatenate(xneg_corners, 0)

    # corners on the positive direction of x
    xpos_corners = [xpos_corners0 - offset*i for i in range(10)]
    xpos_corners = np.concatenate(xpos_corners, 0)

    bboxes_others = bboxes_others.copy()
    #bboxes_others[:,3:6] += 1e-2
    neg_mask = Bbox3D.points_in_bbox(xneg_corners, bboxes_others)
    neg_mask = np.any(neg_mask, 0)
    pos_mask = Bbox3D.points_in_bbox(xpos_corners, bboxes_others)
    pos_mask = np.any(pos_mask, 0)
    neg_intersec = np.any(neg_mask)
    pos_intersec = np.any(pos_mask)

    dif_yaw = np.abs(bbox0[-1] - bboxes_others[:,-1])>1e-1

    x_intersec = []
    if np.any(neg_mask):
      x_intersec.append(np.where(neg_mask)[0][0])
      if not dif_yaw[x_intersec[0]]:
        x_intersec[0] = -1
    else:
      x_intersec.append(-1)

    if np.any(pos_mask):
      x_intersec.append(np.where(pos_mask)[0][0])
      if not dif_yaw[x_intersec[1]]:
        x_intersec[1] = -1
    else:
      x_intersec.append(-1)

    show = DEBUG and False
    if show:
      all_boxes = np.concatenate([bboxes_others, np.expand_dims(bbox0,0)], 0)
      labels = np.array([0]*bboxes_others.shape[0] + [1])
      Bbox3D.draw_points_bboxes(xpos_corners, all_boxes, up_axis, is_yx_zb=False, labels=labels)

    return x_intersec, np.concatenate([xneg_corners0, xpos_corners0], 0)

  @staticmethod
  def detect_all_intersection_corners(bboxes, up_axis, scene_scope=None):
    '''
    intersec_corners_idx: [n][2]
      the intersection index
    intersec_corners: [n,2,3]
    '''
    bn = bboxes.shape[0]
    intersec_corners_idx = []
    intersec_corners = []
    for i in range(bn):
      bboxes_others = bboxes[ [j for j in range(bn) if j!=i] ]
      itsc_i, xcorners_i = Bbox3D.detect_intersection_corners(bboxes[i], bboxes_others, up_axis)
      itsc_i = [d+int(d>=i) for d in  itsc_i]

      if scene_scope is not None:
        # check if the intersec_corners are inside scene_scope
        is_insides = points_in_scope(xcorners_i, scene_scope)
        if itsc_i[0]>=0 and not is_insides[0]:
          itsc_i[0] = -1
        if itsc_i[1]>=0 and (not is_insides[1]):
          itsc_i[1] = -1

      intersec_corners_idx.append( itsc_i )
      intersec_corners.append(np.expand_dims(xcorners_i,0))
    if len(intersec_corners)>0:
      intersec_corners = np.concatenate(intersec_corners, 0)
    intersec_corners_idx = np.array(intersec_corners_idx)

    return intersec_corners_idx, intersec_corners

  @staticmethod
  def crop_bbox_by_points(bbox0, points0, points_aug0, up_axis, intersec_corners0=None):
    '''
    Rotate to make yaw=0 firstly
    (1) Use points_aug0 to constrain size_x and size_z
    (2) Use points0 to constrain size_y (the thickness)
    '''
    if points_aug0.shape[0] < 10:
      # no points inside the box, rm it
      return None

    bbox0 = bbox0.reshape([7])
    assert up_axis == 'Z'
    from geometric_util import Rz

    crop_thickness = False

    centroid0 = np.expand_dims( bbox0[0:3], 0 )

    rz = Rz(bbox0[-1])
    def set_yaw0(ps0):
      return np.matmul(ps0[:,:3] - centroid0, rz)

    # make yaw  = 0
    points1 = set_yaw0(points0)
    points_aug1 = set_yaw0(points_aug0)

    #(1) Use points_aug1 to crop x axis
    xyz_min_new = np.min(points_aug1, 0)
    xyz_max_new = np.max(points_aug1, 0)

    #(2) Use intersec_corners0 to freeze x axis, in order to keep unseen intersection corners
    if intersec_corners0 is not None:
      if intersec_corners0[0] >= 0:
        xyz_min_new[0] = -bbox0[3]
        #set_yaw0( intersec_corners0[0] ).reshape([3])[0]
      if intersec_corners0[1] >= 0:
        xyz_max_new[0] = bbox0[3] # set_yaw0(intersec_corners0[1]).reshape([3])[0]

    #(3) Use points1 to crop y axis (thickness)
    if crop_thickness:
      if points1.shape[0] > 0:
        xyz_min1 = np.min(points1, 0)
        xyz_max1 = np.max(points1, 0)
        xyz_min_new[1] = xyz_min1[1]
        xyz_max_new[1] = xyz_max1[1]
      else:
        # there is no point inside, make thickness=0
        # and wall is close to points_aug1
        sign = np.sign(xyz_min_new[1])
        xyz_min_new[1] = xyz_max_new[1] = bbox0[4] * 0.5 * sign

    xyz_min_new = np.maximum(xyz_min_new, -bbox0[3:6]*0.5)
    xyz_max_new = np.minimum(xyz_max_new, bbox0[3:6]*0.5)

    centroid_new_0 = (xyz_min_new + xyz_max_new) / 2.0
    size_new = np.maximum( xyz_max_new - xyz_min_new, 0)
    #size_new = np.minimum( size_new, bbox0[3:6] )


    if not crop_thickness:
      centroid_new_0[1] = 0
      size_new[1] = bbox0[4]

    centroid_new = np.matmul(centroid_new_0, rz.T) + centroid0.reshape([3])

    bbox_new = np.concatenate([centroid_new, size_new, bbox0[-1:]])

    # do not crop along z
    bbox_new[2] = bbox0[2]
    bbox_new[5] = bbox0[5]

    show = False
    if show and DEBUG:
      bboxes = np.concatenate([bbox0, bbox_new], 0)
      Bbox3D.draw_points_bboxes(points_aug0, bboxes, up_axis='Z', is_yx_zb=False)

    return bbox_new.reshape([1,7])

  @staticmethod
  def line_intersection_2d(line0, line1, must_on0=False, must_on1=False,
          min_angle=0):
    '''
      line0: [2,2]
      line1: [2,2]
      must_on0: must on the scope of line0, no extend
      must_on1: must on the scope of line1, no extend
      out: [2]

      v01 = p1 - p0
      v23 = p3 - p2
      intersection = p0 + v01*k0 = p2 + v23 * k1
      [v01, v23][k0;-k1] = p2 - p0
      intersection between p0 and p1: 1>=k0>=0
      intersection between p2 and p3: 1>=k1>=0
    '''

    assert (line0.shape == (2,2) and line1.shape == (2,2))
            #(line0.shape == (2,3) and line1.shape == (2,3))
    dim = line0.shape[1]
    p0,p1 = line0
    p2,p3 = line1

    v01 = p1-p0
    v23 = p3-p2
    v01v23 = np.concatenate([v01.reshape([2,1]), (-1)*v23.reshape([2,1])], 1)
    p2sp0 = (p2-p0).reshape([2,1])

    try:
      inv_vov1 = np.linalg.inv(v01v23)
      K = np.matmul(inv_vov1, p2sp0)

      if must_on0 and (K[0]>1 or K[0]<0):
        return np.array([np.nan]*2)
      if must_on1 and (K[1]>1 or K[1]<0):
        return np.array([np.nan]*2)

      intersec = p0 + v01 * K[0]
      intersec_ = p2 + v23 * K[1]
      assert np.linalg.norm(intersec - intersec_) < 1e-5, f'{intersec} \n{intersec_}'

      direc0 = (line0[1] - line0[0]).reshape([1,2])
      direc1 = (line1[1] - line1[0]).reshape([1,2])
      angle = angle_of_2lines(direc0, direc1, scope_id=1)[0]
      angle = np.abs(angle)

      show = False
      if show and DEBUG:
        print(f'K:{K}\nangle:{angle}')
        lines_show = np.concatenate([np.expand_dims(line0,0), np.expand_dims(line1,0)],0)
        points_show = np.array([[intersec[0], intersec[1], 0]])
        Bbox3D.draw_points_lines(points_show, lines_show)

      if angle > min_angle:
        return intersec
      else:
        return np.array([np.nan]*2)
    except np.linalg.LinAlgError:
      return np.array([np.nan]*2)

  @staticmethod
  def refine_intersection_x(bbox0, side, bbox1, up_axis):
    assert up_axis == 'Z'
    xfc0 = Bbox3D.bbox_face_centroids(bbox0, up_axis)[0:2,0:2]
    xfc1 = Bbox3D.bbox_face_centroids(bbox1, up_axis)[0:2,0:2]
    intersec = Bbox3D.line_intersection_2d(xfc0, xfc1)
    if intersec is None:
      #bbox1[2] += 0.2
      Bbox3D.draw_bboxes(np.concatenate([bbox0.reshape([-1,7]), bbox1.reshape([-1,7])]), up_axis, False)
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass
    dis = np.linalg.norm(intersec - bbox0[0:2])

    # make fully overlap at the intersection area
    dis += bbox1[4]*0.5

    crop_value = bbox0[3]*0.5 - dis
    return crop_value


  @staticmethod
  def crop_bbox_size(bbox0, axis, values):
    '''
      [7]   'X'   [2]
    '''
    from geometric_util import Rz
    bbox1 = bbox0.reshape([7])
    centroid0 = bbox1[0:3]
    rz = Rz(bbox1[-1])
    dim = {'X':0, 'Y':1, 'Z':2}[axis]
    #def set_yaw0(ps0):
    #  return np.matmul(ps0[:,:3] - centroid0, rz)

    centroid_new = np.array([0,0,0.0])
    centroid_new[dim] = values[0]*0.5 - values[1]*0.5
    centroid_new = np.matmul(centroid_new, rz.T) + centroid0

    bbox1[0:3] = centroid_new
    bbox1[dim+3] = bbox1[dim+3] - values[0] - values[1]
    return bbox1


  @staticmethod
  def boxcorners_in_boxes(anchors, gt_boxes, up_axis):
    '''
    anchors:[na,7]
    gt_boxes:[ng,7]
    '''
    assert anchors.ndim == gt_boxes.ndim == 2
    ng = gt_boxes.shape[0]
    na = anchors.shape[0]
    gt_corners = Bbox3D.bboxes_corners(gt_boxes, up_axis).reshape([-1,3]) # [ng*8,3]
    inside_mask = Bbox3D.points_in_bbox(gt_corners, anchors)  # [ng,8,na]
    inside_mask = inside_mask.reshape([ng,8,na])
    return inside_mask


  @staticmethod
  def bbox_bv_similiarity(anchors, gt_boxes, is_yx_zb, use_croped_gt=False):
    '''
    out the overlap between anchors and gt_boxes:
      [ng,ng]
    '''
    record_t = False
    if record_t: t0 = time.time()
    assert anchors.ndim == gt_boxes.ndim == 2
    assert anchors.shape[1] == gt_boxes.shape[1] == 7
    up_axis = 'Z'
    ng = gt_boxes.shape[0]
    na = anchors.shape[0]
    assert ng>0
    anchors = anchors.copy()
    gt_boxes = gt_boxes.copy()

    if is_yx_zb:
      anchors = Bbox3D.convert_from_yx_zb_boxes(anchors)
      gt_boxes = Bbox3D.convert_from_yx_zb_boxes(gt_boxes)

    if record_t:  print(f'A {time.time() - t0}')
    # (1) get all the intersec_corners: gt_corners inside anchors
    gt_corners = Bbox3D.bboxes_corners(gt_boxes, up_axis) # [ng*8,3]
    if record_t:  print(f'B0 {time.time() - t0}')
    inside_mask = Bbox3D.points_in_bbox(gt_corners.reshape([-1,3]), anchors)  # [ng,8,na]
    if record_t:  print(f'B1 {time.time() - t0}')
    inside_mask = inside_mask.T
    inside_mask = inside_mask.reshape([na,ng,8])
    any_inside_mask = np.any(inside_mask, 2)
    is_total_inside = np.all(inside_mask, 2)

    if record_t:  print(f'B {time.time() - t0}')
    # (2) get all the lines of each bbox
    a_corners = Bbox3D.bboxes_corners(anchors, up_axis) # [ng*8,3]
    gt_lines = np.take(gt_corners, Bbox3D._lines_z0_vids, axis=1)
    a_lines = np.take(a_corners, Bbox3D._lines_z0_vids, axis=1)

    #Bbox3D.draw_lines_open3d(gt_lines.reshape([-1,2,3]), show=True)
    #Bbox3D.draw_lines_open3d(a_lines.reshape([-1,2,3]), show=True)
    if record_t:  print(f'C {time.time() - t0}')

    # (3) get all the line intersections
    l2d_intersec0 = np.zeros([na, ng, 4, 4, 2])
    for i in range(na):
      for j in range(ng):
        for k in range(4):
          for l in range(4):
            l2d_intersec0[i,j,k,l] = Bbox3D.line_intersection_2d(gt_lines[j,k,:,0:2], a_lines[i,l,:,0:2], True, True)

            #if False and DEBUG and ( not np.isnan(l2d_intersec0[i,j,k,l][0])):
            #  boxes_show = np.concatenate([gt_boxes[j:j+1], anchors[i:i+1]],0)
            #  lines_show = np.concatenate([gt_lines[np.newaxis,j,k], a_lines[np.newaxis,i,l]], 0)
            #  lines_show[:,:,2] = 1
            #  intersec_show = np.expand_dims(np.concatenate([l2d_intersec0[i,j,k,l], np.ones(1)], 0), 0)
            #  Bbox3D.draw_points_bboxes(intersec_show, boxes_show, 'Z', False, lines=lines_show)
            #  pass
            #pass
    l2d_intersec_mask0 = np.logical_not(np.isnan(l2d_intersec0[...,0]))
    l2d_intersec_mask1 = np.any(l2d_intersec_mask0, (2,3))
    # (4) No intersection: check if totally contained
    valid_gt_mask = np.logical_or(any_inside_mask, l2d_intersec_mask1)
    if record_t:  print(f'D {time.time() - t0}')
    # (5) Collect union_corners: all the intersections and gt_corners inside anchors
    ids0, ids1 = np.where(valid_gt_mask)
    another_box_ids = np.concatenate([np.expand_dims(ids0,1), np.expand_dims(ids1,1)],1)
    nis = another_box_ids.shape[0]

    overlaps = np.zeros([na,ng], dtype=np.float32)
    croped_gt_boxes = np.empty([na,ng,7], dtype=np.float32)
    croped_gt_boxes.fill(None)

    for i in range(nis):
      aidx, gidx = another_box_ids[i]
      gt_box_i = gt_boxes[gidx]
      l2d_intersec_i = l2d_intersec0[aidx, gidx].reshape([-1,2])
      l2d_intersec_i = l2d_intersec_i[ np.logical_not(np.isnan(l2d_intersec_i[:,0])) ]

      inside_corners = gt_corners[gidx][inside_mask[aidx, gidx]]
      inside_corners = inside_corners.reshape([-1,3])[:,0:2]

      union_corners = np.concatenate([l2d_intersec_i, inside_corners], 0)

      # (6) Calculate the scope of union_corners
      centroid_i = gt_box_i[0:2]
      rz = geo_Rz(gt_box_i[-1])[0:2,0:2]
      # rotate all union_corners to the gt_box_i frame
      union_corners_gtf0 = np.matmul(union_corners-centroid_i, rz)
      union_corners_gtf = union_corners_gtf0 + centroid_i

      xy_min = union_corners_gtf0.min(0)
      xy_max = union_corners_gtf0.max(0)
      sx,sy = xy_scope = xy_max - xy_min

      overlap_i = np.sum(xy_scope)
      overlaps[aidx, gidx] = overlap_i

      # (7) get cropped gt_boxes
      if use_croped_gt:
        centroid_new_i = (xy_min+xy_max)/2
        cx,cy = centroid_new_i = np.matmul(centroid_new_i, rz.T) + centroid_i
        croped_gtbox_i = np.array([cx,cy, gt_box_i[2], sx,sy,gt_box_i[5], gt_box_i[6]])
        if is_yx_zb:
          croped_gt_boxes[aidx, gidx] = Bbox3D.convert_to_yx_zb_boxes(croped_gtbox_i.reshape([1,7]))[0]
        else:
          croped_gt_boxes[aidx, gidx] = croped_gtbox_i

      if DEBUG and False:
        # before rotation
        boxes_show0 = np.concatenate([gt_boxes[gidx:gidx+1], anchors[aidx:aidx+1]], 0)
        points_show0 = np.concatenate([union_corners, np.zeros([union_corners.shape[0],1])], 1)
        Bbox3D.draw_points_bboxes(points_show0, boxes_show0, 'Z', False)

        # after rotation to gt_frame
        boxes_show1 = gt_boxes[gidx:gidx+1].copy()
        boxes_show1[:,-1] -= gt_box_i[-1]
        points_show1 = np.concatenate([union_corners_gtf, np.zeros([union_corners_gtf.shape[0],1])], 1)
        Bbox3D.draw_points_bboxes(points_show1, boxes_show1, 'Z', False)

        # croped box
        boxes_show2 = np.concatenate([boxes_show0, np.expand_dims(croped_gtbox_i,0)], 0)
        labels_show2 = np.array([1,1,0])
        Bbox3D.draw_points_bboxes(points_show0, boxes_show2, 'Z', False, labels=labels_show2)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass

    if record_t:  print(f'E {time.time() - t0}')
    return overlaps, croped_gt_boxes


  @staticmethod
  def cenline_intersection_2boxes(box0, box1, check_same_height):
    '''
      [7] [7]
      detect the intersection of two boxes along xy plane, by centroid lines
      intersection: [1,3], the intersection position
      on_box_corners:[1,2], if the intersection is on the corner of box0 and box1
    '''
    assert box0.shape == box1.shape == (7,)
    corner_dis_threshold = 1.5 # times of thickness

    cenline0 = Bbox3D.bboxes_centroid_lines(box0.reshape([1,7]), cen_axis='X', up_axis='Z')[0]
    cenline1 = Bbox3D.bboxes_centroid_lines(box1.reshape([1,7]), cen_axis='X', up_axis='Z')[0]

    intersec_2d = Bbox3D.line_intersection_2d(cenline0[:,0:2], cenline1[:,0:2], True, True,
                                  min_angle = 10. * np.pi/180)
    if not np.isnan(intersec_2d[0]):
      dis_box_ends_0 = np.linalg.norm(intersec_2d - cenline0[:,0:2], axis=1).min()
      dis_box_ends_1 = np.linalg.norm(intersec_2d - cenline1[:,0:2], axis=1).min()
      dis_box_ends = np.array([dis_box_ends_0, dis_box_ends_1])
      thickness = np.array([box0[4], box1[4]])
      on_box_corners = (dis_box_ends < thickness * corner_dis_threshold).reshape([1,2]).astype(np.int32)
    else:
      on_box_corners = np.array([[-1, -1]])
    intersec_3d = np.concatenate([intersec_2d, cenline0[0,2:3]], 0).reshape([1,3])

    if check_same_height and cenline0[0,2] != cenline1[1,2] and not np.isnan(intersec_3d[0,1]):
      boxes = np.concatenate([box0.reshape([1,7]), box1.reshape([1,7])], 0)
      Bbox3D.draw_bboxes(boxes, 'Z', False)
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      assert False, "merge two walls with different height is not implemented"

    show = False
    if show:
      if np.isnan(intersec_3d[0,0]):
        print('\n\tno intersection')
      else:
        print(intersec_3d)
      print(on_box_corners)
      box_show = np.concatenate([box0.reshape([1,7]), box1.reshape([1,7])],0)
      Bbox3D.draw_points_bboxes(intersec_3d, box_show, 'Z', False)
    return intersec_3d, on_box_corners

  @staticmethod
  def cenline_intersection(box0, boxes_other, check_same_height):
    '''
      [7]  [n,7]
    '''
    assert box0.shape == (7,)
    assert boxes_other.shape[1] == 7
    n = boxes_other.shape[0]
    intersections = []
    on_box_corners = []
    for i in range(n):
      intersection_i, on_box_corners_i =  Bbox3D.cenline_intersection_2boxes(box0, boxes_other[i], check_same_height)
      intersections.append(intersection_i)
      on_box_corners.append(on_box_corners_i)
    intersections = np.concatenate(intersections, 0)
    on_box_corners = np.concatenate(on_box_corners, 0)
    return intersections, on_box_corners

  @staticmethod
  def all_intersections_by_cenline(boxes, check_same_height, not_on_corners=False, only_on_corners=False, x_size_expand=0.08,  show_res=False):
    '''
      [n,7]
    '''
    assert not_on_corners * only_on_corners == 0
    boxes = boxes.copy()
    boxes[:,3] += x_size_expand

    n = boxes.shape[0]
    intersections = [np.zeros(shape=(0,3), dtype=np.float32)] * n
    on_box_corners = [np.zeros(shape=(0), dtype=np.int32)] * n
    another_box_ids = [np.zeros(shape=(0), dtype=np.int32)] * n
    for i in range(n-1):
      intersections_i, on_box_corners_i = Bbox3D.cenline_intersection(boxes[i], boxes[i+1:], check_same_height)

      # extract the valid intersections
      mask_i = np.logical_not( np.isnan(intersections_i[:,0]) )
      idx_i = np.where(mask_i)[0]
      inters_i = intersections_i[idx_i]

      on_box_c_i = on_box_corners_i[idx_i]
      idx_i_org = idx_i + i + 1

      # append all the intersections for box i, while i as the first box
      intersections[i] = np.concatenate([intersections[i], inters_i], 0)
      on_box_corners[i] = np.concatenate([on_box_corners[i], on_box_c_i[:,0]], 0)
      another_box_ids[i] = np.concatenate([another_box_ids[i], idx_i_org], 0)

      # append all the intersections for the second boxes, while i as the first box
      for j in range(idx_i_org.shape[0]):
        idx_j = idx_i_org[j]
        intersections[idx_j] = np.concatenate([intersections[idx_j], inters_i[j:j+1]], 0)
        on_box_corners[idx_j] = np.concatenate([on_box_corners[idx_j], on_box_c_i[j,1:2]], 0)
        another_box_ids[idx_j] = np.concatenate([another_box_ids[idx_j], np.array([i])], 0)

    if not_on_corners or only_on_corners:
      for i in range(n):
        if not_on_corners:
          # keep all the intersections not on box corner
          mask_c_i = on_box_corners[i] == 0
        else:
          # keep all the intersections on box corner
          mask_c_i = on_box_corners[i] == 1
        intersections[i] = intersections[i][mask_c_i]
        on_box_corners[i] = on_box_corners[i][mask_c_i]
        another_box_ids[i] = another_box_ids[i][mask_c_i]

    # filter repeat intersections
    for i in range(n):
        m = intersections[i].shape[0]
        if m<2:
          continue
        keep_mask = np.array([True]*m)
        for j in range(m-1):
          dis_j = intersections[i][j:j+1] - intersections[i][j+1:]
          dis_j = np.linalg.norm(dis_j,axis=1)
          same_mask_j = dis_j < 4e-2
          if np.any(same_mask_j):
            keep_mask[j] = False
            k = np.where(same_mask_j)[0] + j + 1
            intersections[i][k] = (intersections[i][j] + intersections[i][k])/2
            # set 0 if the two on_box_corners are 0 and 1
            on_box_corners[i][k] *= on_box_corners[i][j]


        intersections[i] = intersections[i][keep_mask]
        on_box_corners[i] = on_box_corners[i][keep_mask]
        another_box_ids[i] = another_box_ids[i][keep_mask]

    show = show_res and not_on_corners
    if show:
      all_inters = np.concatenate(intersections, 0)
      Bbox3D.draw_points_bboxes(all_inters, boxes, 'Z', False)
      return intersections

      for i in range(0,n):
        #if intersections[i].shape[0] == 0:
        #  continue
        show_boxes = boxes.copy()
        show_boxes[:,2] -= 1
        show_boxes = np.concatenate([show_boxes, boxes[i:i+1]], 0)
        print(on_box_corners[i])
        print(intersections[i])
        if intersections[i].shape[0] == 0:
          Bbox3D.draw_bboxes(show_boxes, 'Z', False)
        else:
          Bbox3D.draw_points_bboxes(intersections[i], show_boxes, 'Z', False)
    return intersections

  @staticmethod
  def split_wall_by_centroid_intersections(box0, cen_intersecs):
    '''
      box0: [7]
      cen_intersecs: [k,3]
    '''
    assert cen_intersecs.ndim == 2
    assert box0.shape == (7,)
    k = cen_intersecs.shape[0]
    new_walls = box0.reshape([1,7])
    for i in range(k):
      for j in range(new_walls.shape[0]):
        w = new_walls[j]
        # cen_intersecs[i] is on one of wall in new_walls
        new_walls_i = Bbox3D.split_wall_by_one_centroid_intersection(w, cen_intersecs[i])
        if new_walls_i.shape[0] != 0:
          new_walls = np.concatenate([new_walls[:j], new_walls[j+1:], new_walls_i], 0)
          break

    return new_walls

  @staticmethod
  def split_wall_by_one_centroid_intersection(box0, cen_intersec, offset_half_thickness=True):
    '''
      box0: [7]
      cen_intersec: [3]

    '''
    box0 = box0.reshape([1,7])
    cen_intersec = cen_intersec.reshape([1,3])
    cenline0 = Bbox3D.bboxes_centroid_lines(box0, cen_axis='X', up_axis='Z')[0]

    # check if cen_intersec is inside of box0
    dirs = cen_intersec - cenline0
    tmp = np.sum(dirs[0]/np.linalg.norm(dirs[0]) *  dirs[1]/np.linalg.norm(dirs[1]))
    is_inside = tmp < 0
    if not is_inside:
      return np.zeros(shape=[0,7], dtype=np.float32)

    new_centroids = (cenline0 + cen_intersec)/2.0
    x_sizes = np.linalg.norm(cenline0 - cen_intersec, axis=1)
    new_boxes = np.concatenate([box0, box0], 0)
    new_boxes[:,0:3] = new_centroids
    new_boxes[:,3] = x_sizes

    if offset_half_thickness:
      thickness_offset = 0.06 * 0.5
      new_boxes[:,3] += thickness_offset
      tmp = cen_intersec - cenline0
      tmp = tmp / np.linalg.norm(tmp, axis=1, keepdims=True)
      centroid_offset = tmp * thickness_offset
      new_boxes[:,0:3] += centroid_offset

    show = False
    if show:
      show_boxes = np.concatenate([box0, new_boxes], 0)
      show_boxes[1,2] += 2
      show_boxes[2,2] += 2.2
      Bbox3D.draw_points_bboxes(cen_intersec, show_boxes, 'Z', False)
    return new_boxes


def review_bbox_format():
  # bbox standard: [xc, yc, zc, x_size, y_size, z_size, yaw]
  # x_size > y_size,
  # yaw:(-pi/2, pi/2], clock wise is right, following rules in SECOND
  bbox0 = np.array([
                    [1,2,1, 5, 0.5, 1, 0],
                    [1,2,1, 5, 0.5, 1.5, np.pi/2.0*0.5],
                    [1,2,1, 5, 0.5, 0.5, -np.pi/2.0*0.5],
                    ])
  print(f'{bbox0}')
  Bbox3D.draw_bboxes(bbox0, 'Z', is_yx_zb=False)

  # bbox yx_zb  : [xc, yc, z_bot, y_size, x_size, z_size, yaw-0.5pi]
  bbox1 = Bbox3D.convert_to_yx_zb_boxes(bbox0)
  print(f'{bbox1}')
  Bbox3D.draw_bboxes(bbox1, 'Z', is_yx_zb=True)

  bbox2 = Bbox3D.convert_from_yx_zb_boxes(bbox1)
  print(f'{bbox2}')
  Bbox3D.draw_bboxes(bbox2, 'Z', is_yx_zb=False)


def show_bboxes():
  house_name = '0004d52d1aeeb8ae6de39d6bd993e992'
  boxes_fn = f'/home/z/SUNCG/suncg_v1/parsed/{house_name}/object_bbox/wall.txt'
  bboxes = np.loadtxt(boxes_fn)
  Bbox3D.draw_bboxes(bboxes, 'Y', False)

def test_merge_walls():
  wall_fn = '/home/z/SUNCG/suncg_v1/parsed/0004d52d1aeeb8ae6de39d6bd993e992/object_bbox/wall.txt'
  wall_bboxes = np.loadtxt(wall_fn)
  #wall_bboxes = wall_bboxes[[0,2]]
  bbox0 = wall_bboxes[0]
  bbox1 = wall_bboxes[1]
  #merged = Bbox3D.merge_2same_walls(bbox0, bbox1)
  merged = Bbox3D.merge_2close_walls(bbox0, bbox1)
  print(bbox0)
  print(bbox1)
  print(merged)

  w_n = wall_bboxes.shape[0]


  show_all = True
  show_one_by_one = False

  if show_all:
    Bbox3D.draw_bboxes(wall_bboxes, 'Y', False)


  if show_one_by_one:
    for i in range(w_n):
      print(i)
      wall_bboxes[i,1] += 1
      Bbox3D.draw_bboxes(wall_bboxes, 'Y', False)
      wall_bboxes[i,1] -= 1
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

def test_draw():
  box = np.array([[0,0,0, 2,1,2, 0]])
  #Bbox3D.draw_bboxes_mesh(box, 'Z', False)
  Bbox3D.draw_bboxes(box, 'Z', False )



if __name__ ==  '__main__':
  #test_merge_walls()
  #show_bboxes()
  #review_bbox_format()
  test_draw()
