# xyz Nov 2018
import open3d
import numpy as np
import os, pathlib, glob, sys
sys.path.insert(0, '..')
from utils3d.bbox3d_ops import Bbox3D
from collections import defaultdict
import pickle
import torch
from utils3d.geometric_util import cam2world_box, cam2world_pcl
from data3d.suncg_utils.scene_samples import SceneSamples
from maskrcnn_benchmark.structures.bounding_box_3d import  BoxList3D, merge_by_corners
#from suncg_utils.celing_floor_room_preprocessing import preprocess_cfr_standard

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#ROOT_DIR = os.path.dirname(BASE_DIR)
#sys.path.append(ROOT_DIR)
from suncg_utils.wall_preprocessing import show_walls_1by1, show_walls_offsetz

DEBUG = 1

BLOCK_SIZE0 = np.array([50, 50, -1])
ENABLE_LARGE_SIZE_BY_AREA = True
if ENABLE_LARGE_SIZE_BY_AREA:
  MAX_SIZE_FOR_VOXEL_FULL_SCALE = 40.9 # When area is small, size can > 30. But still should < 40.9, to fit VOXEL_FULL_SCALE: [2048, 2048, 512]

NUM_POINTS = 300 * 1000

DSET_DIR = '/DS/SUNCG/suncg_v1'
PARSED_DIR = f'{DSET_DIR}/parsed_NewPcl'
SPLITED_DIR = '/DS/SUNCG/suncg_v1__torch' + f'_BS_{BLOCK_SIZE0[0]}_{BLOCK_SIZE0[1]}_BN_{NUM_POINTS//1000}K_NewPcl'
MAX_FLOAT_DRIFT = 1e-6
DATASET = 'SUNCG'
CLASSES_USED = ['wall', 'window', 'door', 'ceiling', 'floor', 'room']
#CLASSES_USED = ['wall', 'window', 'door']
MIN_BOXES_NUM = 10
MAX_SCENE_SIZE = [81.92, 81.92, 10.24]

NO_SPLIT = 1

ALWAYS_UPDATE = 1
ALWAYS_BIG_SIZE = 0
ONLY_MODIFY_BOX = 0
ALWAYS_UPDATE_MULTI_SPLITS = 0
assert ALWAYS_BIG_SIZE * ONLY_MODIFY_BOX == 0

def points2pcd_open3d(points):
  assert points.shape[-1] == 3
  pcd = open3d.PointCloud()
  points = points.reshape([-1,3])
  pcd.points = open3d.Vector3dVector(points[:,0:3])
  if points.shape[1] == 6:
    pcd.normals = open3d.Vector3dVector(points[:,3:6])
  return pcd

def points_ply(points, plyfn):
  pcd = points2pcd_open3d(points)
  open3d.write_point_cloud(plyfn, pcd)

def random_sample_pcl(points0, num_points1, only_reduce=False):
  n0 = points0.shape[0]
  num_points1 = int(num_points1)
  if num_points1 == n0:
    return points0
  if num_points1 < n0:
    indices = np.random.choice(n0, num_points1, replace=False)
  else:
    if only_reduce:
      return points0
    indices0 = np.random.choice(n0, num_points1-n0, replace=True)
    indices = np.concatenate([np.arange(n0), indices0])
  points1 = np.take(points0, indices, 0)
  return points1

def add_norm(pcd):
  open3d.estimate_normals(pcd, search_param = open3d.KDTreeSearchParamHybrid(
            radius = 0.1, max_nn = 50))
  return pcd

#def read_summary(base_dir):
#  summary_fn = os.path.join(base_dir, 'summary.txt')
#  summary = {}
#  if not os.path.exists(summary_fn):
#    return summary, False
#  with open(summary_fn, 'r') as f:
#    for line in f:
#      line = line.strip()
#      items = [e for e in line.split(' ') if e!='']
#      summary[items[0][:-1]] = int(items[1])
#  return summary

#def write_summary(base_dir, name, value, style='w'):
#  summary_fn = os.path.join(base_dir, 'summary.txt')
#  with open(summary_fn, style) as f:
#    f.write(f"{name}: {value}\n")
#  print(f'write summary: {summary_fn}')


def adjust_wall_corner_to_connect( walls_standard ):
    bboxes_3d = torch.from_numpy(walls_standard)
    boxlist = BoxList3D(bboxes_3d, size3d=None, mode='standard', examples_idxscope=None, constants={})
    new_boxlist = merge_by_corners(boxlist)
    #boxlist.show_with_corners()
    #new_boxlist.show_with_corners()
    assert new_boxlist.mode == 'standard'
    return  new_boxlist.bbox3d.data.numpy()

class IndoorData():
  _block_size0 = BLOCK_SIZE0
  #_block_size0 = np.array([16,16,3])
  _block_stride_rate = np.array([0.8,0.8,0.8])
  _num_points = NUM_POINTS
  _min_pn_inblock = NUM_POINTS / 10

  @staticmethod
  def split_scene(scene_dir, splited_path):
    from suncg_utils.suncg_preprocess import check_house_intact, read_summary, write_summary

    scene_name = os.path.basename(scene_dir)
    splited_path = os.path.join(splited_path, scene_name)
    summary_0 = read_summary(splited_path)


    always_update = ALWAYS_UPDATE

    #house_intact, intacts = check_house_intact(scene_dir)
    #if not house_intact:
    #  return
    summary_raw = read_summary(scene_dir)

    if 'level_num' in summary_raw and  summary_raw['level_num'] != 1:
      return

    #is_big_size = (summary_raw['scene_size'] > MAX_SCENE_SIZE).any()
    #always_update = always_update or ( ALWAYS_BIG_SIZE and is_big_size )

    if (not always_update) and 'split_num' in summary_0:
      sn = summary_0['split_num'].squeeze()
      still_split = ALWAYS_UPDATE_MULTI_SPLITS and sn  >1
      if not still_split:
        print(f'skip {splited_path}')
        return
    print(f'spliting {scene_dir}')
    gen_ply = False


    pcl_fn = os.path.join(scene_dir, 'pcl_camref.ply')
    if not os.path.exists(pcl_fn):
      print(f'pcl.ply not exist, skip {scene_dir}')
      return

    fns = glob.glob(os.path.join(splited_path, '*.pth'))
    if ONLY_MODIFY_BOX and len(fns)>0:
      print('Load points from pth directly')
      scene_name = os.path.basename(os.path.dirname(fns[0]))
      is_special_scene = IndoorData.is_a_special_scene(scene_name)
      points_splited = []
      for fni in fns:
        pcl_i, boxes_i = torch.load(fni)
        points_splited.append(pcl_i)
    else:
        points_splited, is_special_scene = IndoorData.split_pcl_plyf(pcl_fn)
    n_block = len(points_splited)

    bboxes_splited = {}
    boxes_nums = {}
    for obj in CLASSES_USED:
      bbox_fn = os.path.join(scene_dir, f'object_bbox/{obj}.txt')
      if os.path.exists(bbox_fn):
        if is_special_scene or n_block > 1:
            bboxes_splited[obj] = IndoorData.split_bbox(bbox_fn, points_splited)
        else:
            bboxes_splited[obj] = IndoorData.load_bboxes(bbox_fn)
        boxes_nums[obj]  = []
        for ii in range(n_block):
            boxes_nums[obj].append( len(bboxes_splited[obj][ii]) )

    if not os.path.exists(splited_path):
      os.makedirs(splited_path)

    for i in range(n_block):
      boxes_num_all_classes = sum([bn[i] for bn in boxes_nums.values()])
      if n_block>1 and  boxes_num_all_classes < MIN_BOXES_NUM:
          continue
      fni = splited_path + '/pcl_%d.pth'%(i)
      pcl_i = points_splited[i].astype(np.float32)

      #offset = pcl_i[:,0:3].mean(0)
      #pcl_i[:,0:3] = pcl_i[:,0:3] - offset
      pcl_i = np.ascontiguousarray(pcl_i)
      #pcl_i = torch.from_numpy(pcl_i)

      boxes_i = {}
      for obj in bboxes_splited:
        boxes_i[obj] = bboxes_splited[obj][i].astype(np.float32)
        if obj in ['ceiling', 'floor', 'room']:
          boxes_i[obj] = Bbox3D.set_yaw_zero(boxes_i[obj])
          #boxes_i[obj] = preprocess_cfr_standard(boxes_i[obj])

      boxes_i['wall'] = adjust_wall_corner_to_connect(boxes_i['wall'] )
      torch.save((pcl_i, boxes_i), fni)

      if gen_ply:
        Bbox3D.draw_points_bboxes(pcl_i, boxes_i['wall'], 'Z', False)
        pclfn_i = splited_path + f'/pcl_{i}.ply'
        points_ply(pcl_i[:,0:3], pclfn_i)
        boxfn_i = splited_path + f'/wall_{i}.ply'
        Bbox3D.save_bboxes_ply(boxfn_i, boxes_i['wall'], 'Z')
      print(f'save {fni}')
    write_summary(splited_path, 'split_num', n_block, 'w')

  @staticmethod
  def adjust_box_for_thickness_crop(bboxes0):
    # thickness_drift for cropping y
    thickness_drift = 0.02
    size_x_drift = -0.03
    box_offset = np.array([[0,0,0,size_x_drift, thickness_drift, 0,0]])
    bboxes1 = bboxes0 + box_offset
    tmp = np.minimum(0.1, bboxes0[:,3])
    bboxes1[:,3] = np.maximum(bboxes1[:,3], tmp)
    return bboxes1

  @staticmethod
  def load_bboxes(bbox_fn):
    # used when only one block, no need to split
    bboxes = np.loadtxt(bbox_fn).reshape([-1,7])
    return [bboxes]

  @staticmethod
  def split_bbox(bbox_fn, points_splited):
    '''
    bbox in file bbox_fn: up_axis='Y' with always x_size > z_size

    transform with cam2world_box:
      up_axis == 'Z'
      always: x_size > y_size
    '''
    obj = os.path.basename(bbox_fn).split('.')[0]
    min_point_num_per1sm = 10
    # thickness_aug for cropping x
    thickness_aug = 0.3
    assert IndoorData._block_size0[-1] == -1 # do  not crop box along z

    bboxes = np.loadtxt(bbox_fn).reshape([-1,7])
    #if DEBUG:
    #  #show_walls_1by1(bboxes)
    #  bboxes = bboxes[3:4]

    areas = bboxes[:,3] * bboxes[:,5]
    min_point_num = np.minimum( min_point_num_per1sm * areas, 200 )
    bboxes_aug = bboxes.copy()
    #bboxes_aug[:,4] += thickness_aug
    bboxes_aug[:,3:6] = np.clip(bboxes_aug[:,3:6],a_min= thickness_aug, a_max=None )
    bn = bboxes.shape[0]

    sn = len(points_splited)
    bboxes_splited = []
    for i in range(0, sn):
      #  Use to constrain size_x size_z
      point_masks_aug_i = Bbox3D.points_in_bbox(points_splited[i][:,0:3].copy(), bboxes_aug.copy())
      #  Use to constrain size_y (the thickness)
      bboxes_tc = IndoorData.adjust_box_for_thickness_crop(bboxes)
      point_masks_i = Bbox3D.points_in_bbox(points_splited[i][:,0:3].copy(), bboxes_tc)

      pn_in_box_aug_i = np.sum(point_masks_aug_i, 0)
      pn_in_box_i = np.sum(point_masks_i, 0)
      #print(f'no aug:{pn_in_box_i}\n auged:{pn_in_box_aug_i}')

      # (1) The bboxes with no points with thickness_aug will be removed firstly
      keep_box_aug_i = pn_in_box_aug_i > min_point_num
      bboxes_i = bboxes[keep_box_aug_i]

      if DEBUG and obj=='ceiling' and 0:
        rm_box_aug_i = pn_in_box_aug_i <= min_point_num
        print(rm_box_aug_i)
        bboxes_no_points_i = bboxes[rm_box_aug_i].copy()
        bboxes_no_points_i[:,0] += 30
        bboxes_show = np.concatenate([bboxes_i, bboxes_no_points_i],0)
        Bbox3D.draw_points_bboxes(points_splited[i], bboxes_show, up_axis='Z', is_yx_zb=False)

      points_aug_i = [points_splited[i][point_masks_aug_i[:,j]] for j in range(bn)]
      points_aug_i = [points_aug_i[j] for j in range(bn) if keep_box_aug_i[j]]

      points_i = [points_splited[i][point_masks_i[:,j]] for j in range(bn)]
      points_i = [points_i[j] for j in range(bn) if keep_box_aug_i[j]]

      # (2) Crop all the boxes by points and intersec_corners seperately
      bn_i = bboxes_i.shape[0]
      croped_bboxes_i = []
      keep_unseen_intersection = False
      if keep_unseen_intersection:
        intersec_corners_idx_i, intersec_corners_i = Bbox3D.detect_all_intersection_corners(bboxes_i, 'Z')
      else:
        intersec_corners_idx_i = [None]*bn_i
      invalid_bn = 0
      for k in range(bn_i):
        croped_box_k =  Bbox3D.crop_bbox_by_points(
                          bboxes_i[k], points_i[k], points_aug_i[k], 'Z', intersec_corners_idx_i[k])
        if  croped_box_k is  not None:
          croped_bboxes_i.append( croped_box_k )
        else:
          invalid_bn += 1
          #Bbox3D.draw_points_bboxes(points_splited[i], bboxes_i[k:k+1], up_axis='Z', is_yx_zb=False, points_keep_rate=1.0)
          pass
      if len(croped_bboxes_i) > 0:
        croped_bboxes_i = np.concatenate(croped_bboxes_i, 0)
      else:
        croped_bboxes_i = np.array([]).reshape([-1,7])

      # (3) Refine x size of each bbox by thickness croped of intersected bbox
      #croped_size = bboxes_i[:,3:6] - croped_bboxes_i[:,3:6]
      refine_x_by_intersection = False # not correct yet
      if refine_x_by_intersection:
        for k in range(bn_i):
          itsc0, itsc1 = intersec_corners_idx_i[k]
          crop_value = [0,0]
          if itsc0 >= 0:
            crop_value[0] = Bbox3D.refine_intersection_x(croped_bboxes_i[k], 'neg', croped_bboxes_i[itsc0], 'Z')
          if itsc1 >= 0:
            crop_value[1] = Bbox3D.refine_intersection_x(croped_bboxes_i[k], 'pos', croped_bboxes_i[itsc1], 'Z')
          if itsc0 >= 0 or itsc1 > 0:
            croped_bboxes_i[k] = Bbox3D.crop_bbox_size(croped_bboxes_i[k], 'X', crop_value)

          #crop_ysize_neg = croped_size[itsc0,1] if itsc0 >= 0 else 0
          #crop_ysize_pos = croped_size[itsc1,1] if itsc1 >= 0 else 0
          #croped_bboxes_i[k] = Bbox3D.crop_bbox_size(croped_bboxes_i[k], 'X', [crop_ysize_neg, crop_ysize_pos])

      # (4) remove too small wall
      min_wall_size_x = 0.2
      sizex_mask = croped_bboxes_i[:,3] > min_wall_size_x
      croped_bboxes_i = croped_bboxes_i[sizex_mask]
      bboxes_splited.append(croped_bboxes_i)

      show = False
      if show and DEBUG and len(points_i) > 0 and obj=='ceiling':
        print(croped_bboxes_i[:,3])
        points = np.concatenate(points_i,0)
        points = points_splited[i]
        points1 = points.copy()
        points1[:,0] += 30
        points = np.concatenate([points, points1], 0)
        bboxes_i[:,0] += 30
        bboxes_i = np.concatenate([bboxes_i, croped_bboxes_i], 0)
        #Bbox3D.draw_points_bboxes(points, bboxes_i, up_axis='Z', is_yx_zb=False)
        Bbox3D.draw_points_bboxes_mesh(points, bboxes_i, up_axis='Z', is_yx_zb=False)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass
    return bboxes_splited

  @staticmethod
  def is_a_special_scene(scene_name):
      return  scene_name in ['0058113bdc8bee5f387bb5ad316d7b28']

  @staticmethod
  def crop_special_scenes(scene_name, pcd):
      '''
      some special scenes are two large, but contain a lot empty in the middle.
      Directly split by the pipeline is not good. Manually crop
      '''
      points = np.asarray(pcd.points)
      colors = np.asarray(pcd.colors)
      if not IndoorData.is_a_special_scene( scene_name ):
          return points, colors, False
      print(f'This is a special scene: \n\t{scene_name}')
      debuging = False
      xyz_max0 = points.max(0)
      xyz_min0 = points.min(0)
      scope0 = xyz_max0 - xyz_min0

      min_thres = xyz_min0 - 0.1
      max_thres = xyz_max0 + 0.1

      if scene_name == '0058113bdc8bee5f387bb5ad316d7b28':
            max_thres[2] -= 10
      else:
            raise NotImplementedError

      mask0 = np.all( points < max_thres, 1)
      mask1 = np.all( points > min_thres, 1)
      mask = mask0 * mask1
      points_new = points[mask]
      colors_new = colors[mask]

      if debuging:
        open3d.draw_geometries([pcd])
        print(f'orignal min:{xyz_min0}, max:{xyz_max0}, scope:{scope0}')

        xyz_max1 = points_new.max(0)
        xyz_min1 = points_new.min(0)
        scope1 = xyz_max1 - xyz_min1
        print(f'new min:{xyz_min1}, max:{xyz_max1}, scope:{scope1}')

        pcd.points = open3d.Vector3dVector(points_new)
        open3d.draw_geometries([pcd])
      return points_new, colors_new, True

  @staticmethod
  def split_pcl_plyf(pcl_fn):
    assert os.path.exists(pcl_fn)
    pcd = open3d.read_point_cloud(pcl_fn)
    scene_name = os.path.basename( os.path.dirname(pcl_fn))
    points, colors, is_special_scene = IndoorData.crop_special_scenes(scene_name, pcd)

    points = cam2world_pcl(points)
    pcd.points = open3d.Vector3dVector(points)

    points = np.concatenate([points, colors], -1)
    is_add_norm = True
    if is_add_norm:
      add_norm(pcd)
      normals = np.asarray(pcd.normals)
      points = np.concatenate([points,  normals], -1)
    #open3d.draw_geometries([pcd])

    points_splited = IndoorData.points_spliting(points, pcl_fn)
    return points_splited, is_special_scene

  @staticmethod
  def points_spliting(points, pcl_fn):
    if NO_SPLIT:
      wall_fn = pcl_fn.replace('pcl_camref.ply','object_bbox/wall.txt')
      walls = np.loadtxt(wall_fn).reshape([-1,7])
      ceil_fn = pcl_fn.replace('pcl_camref.ply','object_bbox/ceiling.txt')
      ceilings = np.loadtxt(ceil_fn).reshape([-1,7])

      points = forcely_crop_scene(points, walls, ceilings)
      points_splited = [points]
    else:
      splited_vidx, block_size = IndoorData.split_xyz(points[:,0:3],
              IndoorData._block_size0.copy(), IndoorData._block_stride_rate,
              IndoorData._min_pn_inblock)
      if splited_vidx[0] is None:
        points_splited = [points]
      else:
        points_splited = [np.take(points, vidx, axis=0) for vidx in splited_vidx]

    pnums0 = [p.shape[0] for p in points_splited]
    #print(pnums0)
    points_splited = [random_sample_pcl(p, IndoorData._num_points, only_reduce=True)
                        for p in points_splited]

    show = False
    if show:
      pcds = [points2pcd_open3d(points) for points in points_splited]
      #open3d.draw_geometries(pcds)
      for pcd in pcds:
        open3d.draw_geometries([pcd])

    return points_splited

  @staticmethod
  def autoadjust_block_size(self, xyz_scope, num_vertex0):
    # keep xy area within the threshold, adjust to reduce block num
    _x,_y,_z = self.block_size
    _xy_area = _x*_y
    x0,y0,z0 = xyz_scope
    xy_area0 = x0*y0

    nv_rate = 1.0*num_vertex0/self.num_point

    if xy_area0 <= _xy_area or nv_rate<=1.0:
      #Use one block: (1) small area (2) Very small vertex number
      x2 = x0
      y2 = y0

    else:
      # Need to use more than one block. Try to make block num less which is
      # make x2, y2 large.
      # (3) Large area with large vertex number, use _x, _y
      # (4) Large area with not loo large vertex num. Increase the block size by
      # vertex num rate.
      dis_rate = math.sqrt(nv_rate)
      x1 = x0 / dis_rate * 0.9
      y1 = y0 / dis_rate * 0.9
      x2 = max(_x, x1)
      y2 = max(_y, y1)

    block_size= np.array([x2, y2, _z])
    block_size = np.ceil(10*block_size)/10.0
    print('xyz_scope:{}\nblock_size:{}'.format(xyz_scope, block_size))
    return block_size

  @staticmethod
  def split_xyz(xyz, block_size0, block_stride_rate, min_pn_inblock, dynamic_block_size=False):
    xyz_min = np.min(xyz, 0)
    xyz_max = np.max(xyz, 0)
    xyz_scope = xyz_max - xyz_min
    num_vertex0 = xyz.shape[0]

    if dynamic_block_size:
      block_size = autoadjust_block_size(xyz_scope, num_vertex0)
    else:
      block_size = block_size0

    # only split when xy area is large enough
    if ENABLE_LARGE_SIZE_BY_AREA and block_size[2] == -1 and xyz_scope.max() < MAX_SIZE_FOR_VOXEL_FULL_SCALE:
      area_thres = block_size[0] * block_size[1]
      area = xyz_scope[0] * xyz_scope[1]
      if area < area_thres:
        return [None], block_size


    if block_size[2] == -1:
      block_size[2] = np.ceil(xyz_scope[-1])
    block_stride = block_stride_rate * block_size
    block_dims0 =  (xyz_scope - block_size) / block_stride + 1
    block_dims0 = np.maximum(block_dims0, 1)
    block_dims = np.ceil(block_dims0).astype(np.int32)
    #print(block_dims)
    xyzindices = [np.arange(0, k) for k in block_dims]
    bot_indices = [np.array([[xyzindices[0][i], xyzindices[1][j], xyzindices[2][k]]]) for i in range(block_dims[0]) \
                   for j  in range(block_dims[1]) for k in range(block_dims[2])]
    bot_indices = np.concatenate(bot_indices, 0)
    bot = bot_indices * block_stride
    top = bot + block_size

    block_num = bot.shape[0]
    #print('raw scope:\n{} \nblock_num:{}'.format(xyz_scope, block_num))
    #print('splited bot:\n{} splited top:\n{}'.format(bot, top))

    if block_num == 1:
      return [None], block_size

    if block_num>1:
      for i in range(block_num):
        for j in range(3):
          if top[i,j] > xyz_scope[j]:
            top[i,j] = xyz_scope[j] - MAX_FLOAT_DRIFT
            bot[i,j] = np.maximum(xyz_scope[j] - block_size[j] + MAX_FLOAT_DRIFT, 0)

    bot += xyz_min
    top += xyz_min

    dls_splited = []
    num_points_splited = []
    splited_vidx = []
    for i in range(block_num):
      mask0 = xyz >= bot[i]
      mask1 = xyz < top[i]
      mask = mask0 * mask1
      mask = np.all(mask, 1)
      new_n = np.sum(mask)
      indices = np.where(mask)[0]
      num_point_i = indices.size
      if num_point_i < min_pn_inblock:
        #print('num point {} < {}, block {}/{}'.format(num_point_i,\
        #                                self.num_point * 0.1, i, block_num))
        continue
      num_points_splited.append(num_point_i)
      splited_vidx.append(indices)

    num_vertex_splited = [d.shape[0] for d in splited_vidx]
    return splited_vidx, block_size


def forcely_crop_scene(pcl0, walls, ceilings):
  scene_min = pcl0[:,:3].min(0)
  scene_max = pcl0[:,:3].max(0)
  scene_size = scene_max - scene_min
  abandon = scene_size - MAX_SCENE_SIZE

  masks = []
  ref_boxes = [walls, walls, ceilings]
  for i in range(3):

    if abandon[i] > 0:
      if ref_boxes[i].shape[0] == 0:
        if i<2:
          rate_min = 0.5
        else:
          rate_min = 0

      else:
        ref_min = ref_boxes[i][:,i].min()
        ref_max = ref_boxes[i][:,i].max()
        wmin = ref_min - scene_min[i]
        wmax =  scene_max[i] - ref_max
        rate_min = wmin / (wmin+wmax)
      new_mini = scene_min[i] + abandon[i] * rate_min
      new_maxi = scene_max[i] - abandon[i] * (1-rate_min)
      masks.append( (pcl0[:,i] > new_mini) * (pcl0[:,i] < new_maxi) )

  if len(masks) > 0:
    mask = masks[0]
    for m in masks:
      mask = mask * m
    pcl1 = pcl0[mask]

  else:
    pcl1 = pcl0

  show = 0
  if show:
    scene_min_new = pcl1[:,:3].min(0)
    scene_max_new = pcl1[:,:3].max(0)
    print(f'Org: {scene_min} - {scene_max}')
    print(f'New: {scene_min_new} - {scene_max_new}')
    pcl00 = pcl0.copy()
    pcl00[:,2] -= 15
    pcl_show = np.concatenate([pcl00, pcl1], 0)
    Bbox3D.draw_points_bboxes(pcl_show, walls, 'Z', False, points_keep_rate=1)
  return pcl1
def read_house_names(fn):
  with open(fn) as f:
    lines = f.readlines()
  lines = [line.strip() for line in lines]
  return lines

def box3d_t_2d(box3d, P2):
  from second.core.box_np_ops import center_to_corner_box3d, project_to_image
  locs = box3d[:, :3]
  dims = box3d[:, 3:6]
  angles = box3d[:,6]
  camera_box_origin = [0.5,1.0,0.5]
  box_corners_3d =  center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=1)
  box_corners_in_image = project_to_image(box_corners_3d, P2)

  minxy = np.min(box_corners_in_image, axis=1)
  maxxy = np.max(box_corners_in_image, axis=1)
  box_2d_preds = np.concatenate([minxy, maxxy], axis=1)
  return box_2d_preds

def get_box3d_cam(box3d_lidar, rect, Trv2c):
  from second.core.box_np_ops import box_lidar_to_camera
  box3d_cam = box_lidar_to_camera(box3d_lidar, rect, Trv2c)
  return box3d_cam

def get_alpha(box3d_lidar):
  return -np.arctan2(-box3d_lidar[:,1], box3d_lidar[:,0]) + box3d_lidar[:,6]

def get_sung_info(data_path, house_names0):
  data_path = os.path.join(data_path, 'houses')
  house_names1 = os.listdir(data_path)
  house_names = [h for h in house_names1 if h in house_names0]

  infos = []
  for house in house_names:
    house_path = os.path.join(data_path, house)
    object_path = os.path.join(house_path, 'objects')

    pcl_fns = glob.glob(os.path.join(house_path, 'pcl*.bin'))
    pcl_fns.sort()
    #pcl_fns = [pcl_fn.split('houses')[1] for pcl_fn in pcl_fns]
    pcl_num = len(pcl_fns)

    box_fns = glob.glob(os.path.join(object_path, '*.bin'))
    box_fns.sort()
    objects = set([os.path.basename(fn).split('_')[0] for fn in box_fns])

    for i in range(pcl_num):
      info = {}
      info['velodyne_path'] = pcl_fns[i]
      info['pointcloud_num_features'] = 6

      info['image_idx'] = '0'
      info['image_path'] = 'empty'
      info['calib/R0_rect'] = np.eye(4)
      info['calib/Tr_velo_to_cam'] = np.eye(4) # np.array([[0,-1,0,0]. [0,0,-1,0], [1,0,0,0], [0,0,0,1]], dtype=np.float32)
      info['calib/P2'] = np.eye(4)

      base_name = os.path.splitext( os.path.basename(pcl_fns[i]) )[0]
      idx = int(base_name.split('_')[-1])

      annos = defaultdict(list)
      for obj in objects:
        box_fn = os.path.join(object_path, obj+'_'+str(idx)+'.bin')
        box = np.fromfile(box_fn, np.float32)
        box = box.reshape([-1,7])
        box = Bbox3D.convert_to_yx_zb_boxes(box)
        box_num = box.shape[0]
        annos['location'].append(box[:,0:3])
        annos['dimensions'].append(box[:,3:6])
        annos['rotation_y'].append(box[:,6])
        annos['name'].append( np.array([obj]*box_num) )

        annos['difficulty'].append(np.array(['A']*box_num))
        annos['bbox'].append( box3d_t_2d(box, info['calib/P2'] ) )
        annos['box3d_camera'].append( get_box3d_cam(box, info['calib/R0_rect'], info['calib/Tr_velo_to_cam']) )
        bn = box.shape[0]
        annos["truncated"].append(np.array([0.0]*bn))
        annos["occluded"].append(np.array([0.0]*bn))
        annos["alpha"].append( get_alpha(box) )

      for key in annos:
        annos[key] = np.concatenate(annos[key], 0)

      info['annos'] = annos

      infos.append(info)
  return infos

def Unused_creat_indoor_info_file(data_path=SPLITED_DIR,
                           save_path=None,
                           create_trainval=False,
                           relative_path=True):
    '''
    Load splited bbox in standard type and bin format.
    Save in picke format and bbox of yx_zb type.
    '''
    house_names = {}
    house_names['train'] = read_house_names("%s/train_test_splited/train.txt"%(data_path))
    house_names['val'] = read_house_names("%s/train_test_splited/val.txt"%(data_path))

    #house_names['train'] = ['001188c384dd72ce2c2577d034b5cc92']
    #house_names['val'] = ['001188c384dd72ce2c2577d034b5cc92']

    print("Generate info. this may take several minutes.")
    if save_path is None:
        save_path = pathlib.Path(data_path)
    else:
        save_path = pathlib.Path(save_path)

    for split in house_names:
      sung_infos = get_sung_info(data_path, house_names[split])
      filename = save_path / pathlib.Path('sung_infos_%s.pkl'%(split))
      print(f"sung info {split} file is saved to {filename}")
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      with open(filename, 'wb') as f:
          pickle.dump(sung_infos, f)

def unused_read_indoor_info():
  info_path = f'{SPLITED_DIR}/sung_infos_train.pkl'
  with open(info_path, 'rb') as f:
    infos = pickle.load(f)
  idx = 0
  print(f'totally {len(infos)} blocks')
  for idx in range(0,len(infos)):
    info = infos[idx]
    pcl_path = info['velodyne_path']
    pointcloud_num_features = info['pointcloud_num_features']
    points = np.fromfile( pcl_path, dtype=np.float32).reshape([-1, pointcloud_num_features])

    annos = info['annos']
    loc = annos['location']
    dims = annos['dimensions']
    rots = annos['rotation_y']
    gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

    Bbox3D.draw_points_bboxes(points, gt_boxes, 'Z', is_yx_zb=True)

def get_house_names_1level():
  house_names_1level_fn = f'{DSET_DIR}/house_names_1level.txt'
  with open(house_names_1level_fn, 'r') as f:
    house_names_1level = [l.strip() for l in f.readlines()]
  return house_names_1level


def creat_splited_pcl_box():
  '''
  Load parsed objects for whole scene. Split house, generate splited point cloud,and bbox.
  Splited point cloud saved in bin.
  Splited bbox saved in bin with standard type.
  '''
  parsed_dir = PARSED_DIR
  splited_path = f'{SPLITED_DIR}/houses'
  house_names = os.listdir(parsed_dir)
  house_names.sort()

  #house_names = house_names[0:1000]
  #house_names = house_names[1000:2000]
  #house_names = house_names[4000:5000]
  #house_names = house_names[5000:6000]
  #house_names = house_names[6000:]


  #house_names = SceneSamples.pcl_err

  #house_names = get_house_names_1level()
  print(f'total {len(house_names)} houses')
  #house_names = ['16a5bfe1972802178762f5a052bbf450']

  scene_dirs = [os.path.join(parsed_dir, s) for s in house_names]
  scene_dirs.sort()

  #scene_dirs = scene_dirs[0:]

  sn = len(scene_dirs)
  for i,scene_dir in enumerate( scene_dirs ):
    print(f'\n{i} / {sn}')
    IndoorData.split_scene(scene_dir, splited_path)
    print(f'split ok: {scene_dir}')

def rm_some_files(house_names0):
  '''
  generate based on this file
  '''
  print('\n\nComform genrating based on this file?\n\n')
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  train_fn0 = '/DS/SUNCG/suncg_v1_torch_splited/train_test_splited/train_4069.txt'
  val_fn0 = '/DS/SUNCG/suncg_v1_torch_splited/train_test_splited/val_4069.txt'
  with open(train_fn0, 'r') as f:
    train_hns = f.readlines()
    train_hns = [h.split('\n')[0] for h in train_hns]
  with open(val_fn0, 'r') as f:
    val_hns = f.readlines()
    val_hns = [h.split('\n')[0] for h in val_hns]
  rm_hns = train_hns + val_hns

  house_names1 = [h for h in house_names0 if h not in rm_hns]
  return house_names1

def gen_train_list():
  house_names = os.listdir(os.path.join(SPLITED_DIR, 'houses'))

  #house_names = rm_some_files(house_names)

  num = len(house_names)
  if DEBUG and 0:
      house_names.sort()
      train_num = num
      train_num = int(num*0.7)
      train_num = min(num, 100)
  else:
      np.random.shuffle(house_names)
      train_num = int(num*0.8)
  print(f'train_num: {train_num}\ntest_num:{num-train_num}')
  train_hosue_names = np.sort(house_names[0:train_num])
  test_house_names = np.sort(house_names[train_num:])

  split_path = os.path.join(SPLITED_DIR, 'train_test_splited')
  if not os.path.exists(split_path):
    os.makedirs(split_path)
  train_fn = os.path.join(split_path, 'train_.txt')
  test_fn = os.path.join(split_path, 'val_.txt')
  with open(train_fn, 'w') as f:
    f.write('\n'.join(train_hosue_names))
  with open(test_fn, 'w') as f:
    f.write('\n'.join(test_house_names))
  print(f'gen_train_list ok: {train_fn}')


if __name__ == '__main__':
  #creat_splited_pcl_box()
  gen_train_list()
  pass

