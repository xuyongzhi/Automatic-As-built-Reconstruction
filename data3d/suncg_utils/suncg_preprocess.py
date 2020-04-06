# 13 Nov 2018

from __future__ import print_function
import glob, os, json, sys
from PIL import Image
import numpy as np
import open3d
from collections import defaultdict
from data3d.indoor_data_util import random_sample_pcl


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
USER_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))
from wall_preprocessing import preprocess_walls
from window_preprocessing import preprocess_windows
from door_preprocessing import preprocess_doors
from celing_floor_room_preprocessing import preprocess_cfr
from utils3d.bbox3d_ops import Bbox3D
from data3d.dataset_metas import DSET_METAS0
from scene_samples import SceneSamples


Debug = True
FunctionUncomplemented = True
MIN_CAM_NUM = 10
MIN_POINT_NUM = 10000*10
ENABLE_NO_RECTANGLE = ['Ceiling', 'Floor', 'Room']
SAGE = True

ONLY_LEVEL_1 = True

SUNCG_V1_DIR = '/DS/SUNCG/suncg_v1'
PARSED_DIR_GOING = '_parsed_NewPcl__'
PARSED_DIR_READY = 'parsed_NewPcl'
#PARSED_PATH = f'{SUNCG_V1_DIR}/{PARSED_DIR_GOING}'

def show_pcl(pcl):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(pcl[:,0:3])
    if pcl.shape[1] == 6:
      pcd.colors = open3d.Vector3dVector(pcl[:,3:6])
    open3d.draw_geometries([pcd])
def camPos2Extrinsics(cam_pos):
  '''
  cam:
  ref:
  https://github.com/shurans/SUNCGtoolbox/blob/master/gaps/apps/scn2cam/scn2cam.cpp Line207
  vx, vy, vz, tx, ty, tz, ux, uy, uz, xf, yf, value
  [vx, vy, vz] is the centroid
  [tx,ty,tz] is cam forward orientation
  [ux,uy,uz] is cam right orientation
  value is the score of camera
  xf: xfov   yf:yfov

  ref: https://github.com/shurans/sscnet/blob/master/matlab_code/utils/camPose2Extrinsics.m
  cam_pos: [12]
  '''
  #cam_pos = np.array([38.2206 ,   1.2672 ,  41.8938  ,  0.9785 ,  -0.1961  , -0.0644 ,   0.1957,    0.9806 ,  -0.0129  ,  0.5500 ,   0.4310,   14.2487])
  cam_pos = np.reshape(cam_pos, [12,1])
  vv = cam_pos[0:3,:]
  tv = cam_pos[3:6,:]
  uv = cam_pos[6:9,:]
  rv = np.reshape(np.cross(tv[:,0],uv[:,0]), [3,1])
  extrinsics = np.concatenate([rv, -uv, tv, vv], 1)

  # check R
  R = extrinsics[0:3,0:3]
  I = np.matmul(R, R.T)
  assert np.sum( np.abs(np.diag(I)-1) ) < 1e-2
  assert np.sum(np.abs(I-np.eye(3))) < 1e-2
  return extrinsics

def camFocus(cam_pos, resolution):
  '''
    Get cam focus in pixel unit
    Note: xfov and yfov is half view field
    Although there are only width -1 pixels in the fov, it actually covers width pixels.
  '''
  h,w = resolution
  #w -= 1
  #h -= 1
  focus = 0.5*w / np.tan(cam_pos[9])
  focus1 = 0.5*h / np.tan(cam_pos[10])
  assert abs(focus-focus1) < 1e-2
  return focus

def parse_face(f_line):
  # v/vt/vn
  face = [s.split('/') for s in f_line]
  face = np.array([[int(d) for d in e] for e in face])
  face = face.T - 1
  assert face.shape == (3,3)
  # keep v/vn
  face_vidx = face[0]
  face_normidx = face[2]
  return face_vidx, face_normidx

def read_obj_parts(obj_fn):
  vertex_nums = []
  face_nums = []
  part_names = []
  norm_buf = []
  vnum = 0
  fnum = 0

  vertices = []
  face_vidxs = []
  face_normidxs = []
  with open(obj_fn, 'r') as f:
    for line in f:
      line = line.strip().split(' ')
      if line[0] == 'v':
        vertex = [float(d) for d in line[1:]]
        vertices.append(vertex)
        vnum += 1
      elif line[0] == 'f':
        face_vidx, face_normidx = parse_face(line[1:])
        face_vidxs.append(face_vidx)
        face_normidxs.append(face_normidx)
        fnum += 1
      if line[0] == 'vn':
        vn = [float(e) for e in line[1:]]
        norm_buf.append(vn)
      elif line[0]=='o':
        part_names.append(line[1])
        if vnum>0:
          vertex_nums.append(vnum)
          face_nums.append(fnum)
          vnum = 0
          fnum = 0
    vertex_nums.append(vnum)
    face_nums.append(fnum)
    norm_buf = np.array(norm_buf)

    vertices = np.array(vertices)
    face_vidxs = np.array(face_vidxs)
    face_normidxs = np.array(face_normidxs)
    face_normals = np.take(norm_buf, face_normidxs[:,0], axis=0)

    # split parts
    part_num = len(vertex_nums)
    cumsum_vn = np.concatenate([[0],np.cumsum(vertex_nums)])
    cumsum_fn = np.concatenate([[0],np.cumsum(face_nums)])

    mesh_parts = []
    for i in range(part_num):
      mesh_i = {}
      mesh_i['vertices'] = vertices[cumsum_vn[i]:cumsum_vn[i+1],:]
      mesh_i['face_vidx'] = face_vidxs[cumsum_fn[i]:cumsum_fn[i+1],:] - cumsum_vn[i]
      mesh_i['face_norms'] = face_normals[cumsum_fn[i]:cumsum_fn[i+1],:]
      mesh_i['name'] = part_names[i]
      if np.isnan(mesh_i['vertices']).all():
        continue
      bbox = get_part_bbox(mesh_i['vertices'], mesh_i['face_vidx'], mesh_i['face_norms'],  mesh_i['name'])
      if bbox is None and FunctionUncomplemented:
        continue
      mesh_i['bbox'] = bbox
      mesh_parts.append(mesh_i)
    mesh_parts_new = merge_inside_out(mesh_parts)
    return mesh_parts_new

def merge_two_parts(p0, p1):
  p1['face_vidx']  += p0['vertices'].shape[0]
  for item in ['vertices', 'face_vidx', 'face_norms']:
    p0[item]  = np.concatenate([p0[item], p1[item]], 0)
  p0['name'] = p0['name'].replace('Inside', '')
  p0['bbox'] = Bbox3D.merge_two_bbox(p0['bbox'], p1['bbox'], 'Y')
  return p0

def merge_inside_out(mesh_parts):
  part_num = len(mesh_parts)
  part_names = [part['name'] for part in mesh_parts]
  new_parts = []

  for i in range(part_num):
    part_i = mesh_parts[i]
    name_i = part_i['name']
    if 'Inside' not in name_i and 'Outside' not in name_i:
      new_parts.append(part_i)
    else:
      if 'Inside' in name_i:
        outside_name = name_i.replace('Inside', 'Outside')
        for j in range(part_num):
          if mesh_parts[j]['name'] == outside_name:
            part_ij = merge_two_parts(mesh_parts[i], mesh_parts[j])
            new_parts.append(part_ij)
            break
  return new_parts

def show_mesh(vertices, triangle, color=[0,0,0], only_genmesh=False):
  mesh = open3d.TriangleMesh()
  mesh.vertices = open3d.Vector3dVector(vertices)
  mesh.triangles = open3d.Vector3iVector(triangle)
  mesh.paint_uniform_color(color)
  centroid = np.mean(vertices, 0)
  mesh_frame = open3d.create_mesh_coordinate_frame(size = 1.6, origin = centroid)
  if not only_genmesh:
    open3d.draw_geometries([mesh, mesh_frame])
  return mesh

def get_part_bbox(vertices, triangle, triangle_norms, name=''):
  '''
  bbox: [xc, yc, zc, x_size, y_size, z_size, yaw]
  '''
  #show_mesh(vertices, triangle)
  class_name = name.split('#')[0]

  box_min = np.min( vertices, 0)
  box_max = np.max( vertices, 0)

  centroid = (box_min + box_max)  /2.0
  y_size = box_max[1] - box_min[1]


  ## Find the 8 outside corners
  distances = np.linalg.norm( vertices - np.expand_dims( centroid, 0 ), axis=1)
  max_dis = max(distances)
  out_corner_mask = (abs(distances-max_dis) < 1e-5)
  n0 = vertices.shape[0]
  #print([i for i in range(n0) if out_corner_mask[i]])
  out_vertices = [vertices[i:i+1,:] for i in range(n0) if out_corner_mask[i]]
  if len(out_vertices) == 0:
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass
    return None
  out_vertices = np.concatenate(out_vertices, 0)


  if out_vertices.shape[0] != 8:
    #Bbox3D.draw_points_open3d(out_vertices, show=True)
    #Bbox3D.draw_points_open3d(vertices, show=True)
    #show_mesh(vertices, triangle)
    if class_name  not in ENABLE_NO_RECTANGLE:
        print(f'\nFailed to find bbox, not rectangle, {class_name} \n {out_vertices.shape[0]} vertices\n')
        assert False
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass
        return None
    else:
        print(f'\nNot rectangle, use no yaw box, {class_name} \n {out_vertices.shape[0]} vertices\n')
        min_max = {'min':box_min, 'max':box_max}
        bbox_noyaw = Bbox3D.bbox_from_minmax( min_max )
        #Bbox3D.draw_points_bboxes(vertices, bbox_noyaw, 'Y', is_yx_zb=False)
        return bbox_noyaw

  ## Find the 4 corners on one side
  x_right_mask = out_vertices[:,0]-centroid[0] > 0
  x_right_corners = np.concatenate([out_vertices[i:i+1,:] for i in range(8) if x_right_mask[i]],0)
  assert x_right_corners.shape[0] == 4
  x_right_cen = np.mean(x_right_corners, 0)

  ## get length
  x_size = np.linalg.norm(x_right_cen - centroid) * 2

  if not abs(x_right_cen[1] - centroid[1]) < 1e-7:
    print("y should be same, pitch and rool should be 0")

  ## the angle between x_right_cen and x axis
  x_direc = x_right_cen-centroid
  x_direc = (x_direc) / np.linalg.norm(x_direc)
  x_axis = np.array([1,0,0])
  # [-90, 90]
  yaw = np.arccos( np.sum(x_direc * x_axis) )
  if abs(yaw)<0.01:
    yaw = 0
  else:
    assert x_direc[0] > 0
    assert x_direc[1] == 0
    yaw *= np.sign(x_direc[2])

  ## Find the 2 corners at top within x_right_corners
  top_mask = x_right_corners[:,1] - centroid[1]>0
  x_right_top_corners = np.concatenate( [x_right_corners[i:i+1,:] for i in range(4) if top_mask[i]], 0)
  assert x_right_top_corners.shape[0] == 2
  # get z_size
  z_size = np.linalg.norm(x_right_top_corners[0] - x_right_top_corners[1])

  ### got the bbox
  xc,yc,zc = centroid
  bbox = np.array([xc, yc, zc, x_size, y_size, z_size, yaw])

  ### Check
  if yaw!=0 and False:
    Bbox3D.draw_bboxes(bbox, 'Y', False)

  if False:
      min_max = {'min':box_min, 'max':box_max}
      bbox_noyaw = Bbox3D.bbox_from_minmax( min_max )
      bboxes_show = np.concatenate([bbox.reshape([1,7]), bbox_noyaw.reshape([1,7])], 0)
      Bbox3D.draw_points_bboxes(vertices, bboxes_show, 'Y', is_yx_zb=False)
  return bbox


def read_room_obj(obj_fn, room_parts_dir):
  import pymesh
  #mesh0 = pymesh.load_mesh(obj_fn)
  mesh_parts = read_obj_parts(obj_fn)
  bboxs_roomparts = [part['bbox'] for part in mesh_parts]
  room_name = os.path.splitext(os.path.basename(obj_fn))[0]

  is_save_part = False
  if is_save_part:
    print(f'save room parts in:\n {room_parts_dir}')
    for i in range(len(mesh_parts)):
      part_i = mesh_parts[i]
      bbox_i = part_i['bbox']
      part_bbox_fn = os.path.join(room_parts_dir, 'bbox_'+room_name+'_'+part_i['name']+'.ply')
      Bbox3D.save_bbox_ply(part_bbox_fn, bbox_i, 'Y')

    for i in range(len(mesh_parts)):
      part_i = mesh_parts[i]
      mesh_i = pymesh.form_mesh(part_i['vertices'], part_i['face_vidx'])
      mesh_i.add_attribute('face_norm')
      mesh_i.set_attribute('face_norm', part_i['face_norms'])
      part_obj_fn = os.path.join(room_parts_dir, room_name + '_' + part_i['name']+'.ply')
      pymesh.save_mesh(part_obj_fn, mesh_i, ascii=True, use_float=True)

  return bboxs_roomparts


def read_ModelCategoryMap(tool_meta):
    import csv
    csv_f = csv.reader(open(tool_meta), delimiter=',')
    modelId_2_fine_grained_class = {}
    modelId_2_coarse_grained_class = {}
    for i,row in enumerate(csv_f):
      if i==0:
        assert row[1] == 'model_id'
        assert row[2] == 'fine_grained_class'
        assert row[3] == 'coarse_grained_class'
      else:
        modelId_2_fine_grained_class[row[1]] = row[2]
        modelId_2_coarse_grained_class[row[1]] = row[3]
    modelId_2_class = modelId_2_coarse_grained_class
    return modelId_2_class


def cam2world_box(box):
  '''
  transform from cam frame to world frame
  up_axis from 'Y' to 'Z'
  '''
  assert box.shape[1] == 7
  R = np.eye(7)
  R[1,1] = R[2,2] = 0
  R[1,2] = 1
  R[2,1] = -1
  R[4,4] = R[5,5] = 0
  R[4,5] = 1
  R[5,4] = 1
  R[6,6] = 1
  box = np.matmul(box, R)
  return box

def rm_bad_scenes(house_fns):
    house_fns_new = []
    for fn in house_fns:
        hn = os.path.basename(os.path.dirname(fn))
        if hn not in SceneSamples.bad_scenes:
            house_fns_new.append(fn)
    return house_fns_new

def cam2world_pcl(points):
  assert points.shape[1] == 3
  R = np.eye(3)
  R[1,1] = R[2,2] = 0
  R[1,2] = 1
  R[2,1] = -1
  points = np.matmul(points, R)
  return points

def world2cam_box(box):
  assert box.shape[1] == 7
  R = np.eye(7)
  R[1,1] = R[2,2] = 0
  R[1,2] = -1
  R[2,1] = 1
  R[4,4] = R[5,5] = 0
  R[4,5] = 1
  R[5,4] = 1
  R[6,6] = 1
  box = np.matmul(box, R)
  return box

class Suncg():
  #tool_bin = f"{USER_DIR}/Research/SUNCGtoolbox/gaps/bin/x86_64"
  #tool_meta = f"{USER_DIR}/Research/SUNCGtoolbox/metadata/ModelCategoryMapping.csv"
  tool_bin = f'{BASE_DIR}/tools'
  tool_meta = f'{BASE_DIR}/tools/ModelCategoryMapping.csv'
  modelId_2_class = read_ModelCategoryMap(tool_meta)
  def __init__(self, root_path):
    self.root_path = root_path

    #scene_ids = os.listdir(root_path+'/house')
    scene_ids = SceneSamples.sj_paper_samples
    house_fns = [os.path.join(root_path, 'house/%s/house.json'%(scene_id)) for scene_id in scene_ids]
    house_fns.sort()
    self.house_fns = house_fns
    # Currently totally 9000 houses used, which include over 6000 singel level houses

    #if SAGE:
    #  #self.house_fns = house_fns[5000: 7000]
    #  self.house_fns = house_fns[7000: 9000]
    #else:
    #  self.house_fns = house_fns[4460:5000]

    if Debug and 0:
      scene_id = '1314639115291b9ed8109d71008f9822'
      self.house_fns = [f'{SUNCG_V1_DIR}/house/{scene_id}/house.json']

      #self.house_fns = [f'{SUNCG_V1_DIR}/house/{scene_id}/house.json' for scene_id in SceneSamples.pcl_err]

    self.house_fns = rm_bad_scenes(self.house_fns)

    print(f'house num: {len(self.house_fns)}')

  def parse_houses_pool(self):
    import multiprocessing as mp
    threads = 8 if SAGE else 8
    p = mp.Pool(processes=threads)
    p.map(parse_house_onef, self.house_fns)
    p.close()
    p.join()

  def parse_houses(self, record_fail_fn=True):
    if record_fail_fn:
      fail_hs = open('./fail_hns.txt','w')

    #house_names_1level = []
    for k,fn in enumerate(self.house_fns):
      print(f'\nstart {k+1}th house: \n  {fn}\n')
      if not record_fail_fn:
          parse_house_onef(fn)
      else:
        try:
          parse_house_onef(fn, find_fail_scene=True)
        except:
          print('\n\t\tThis file fails\n')
          hn = os.path.basename( os.path.dirname(fn) )
          fail_hs.write('\''+hn+'\', ')
          fail_hs.flush()
      print(f'\nfinish {k+1} houses\n')

    if record_fail_fn:
        fail_hs.close()

def parse_house_onef( house_fn, find_fail_scene=False ):
    '''
    1. Generate depth image
    2. point cloud for each depth image
    3. Merge point clouds
    '''
    is_gen_bbox = 1
    is_gen_cam = 1 - find_fail_scene
    is_gen_pcl = 1 - find_fail_scene

    is_gen_house_obj = Debug and 0
    if is_gen_house_obj:
      is_gen_bbox = is_gen_cam = is_gen_pcl = 0

    if is_gen_house_obj:
      gen_house_obj(house_fn)

    if is_gen_bbox:
      gen_bbox(house_fn)

    parsed_dir = get_pcl_path(house_fn)
    summary = read_summary(parsed_dir)
    if ONLY_LEVEL_1 and 'level_num' in summary and summary['level_num'] != 1:
      return

    if is_gen_cam:
      gen_cam_images(house_fn)
    if is_gen_pcl:
      gen_pcl(house_fn)
    print(f'house ok: {house_fn}')

def write_summary(base_dir, name, value, style='w'):
  summary_fn = os.path.join(base_dir, 'summary.txt')
  value = np.array(value).reshape([-1])
  with open(summary_fn, style) as f:
    f.write(f"{name}: ")
    for v in value:
      f.write(f"{v}  ")
    f.write("\n\n")
  print(f'write summary: {summary_fn}')

def read_summary(base_dir):
  summary_fn = os.path.join(base_dir, 'summary.txt')
  summary = {}
  if not os.path.exists(summary_fn):
    return summary
  with open(summary_fn, 'r') as f:
    for line in f:
      line = line.strip()
      items = [e for e in line.split(' ') if e!='']
      if len(items)==0:
          continue
      style = items[0][:-1]
      if style == 'pcl_size':
        continue
      #print(style)
      #print(items[1])
      value = []
      for v in items[1:]:
        value.append(float(v))
      value = np.array(value)
      if style in ['xyarea', 'area']:
        pass
      elif style in ['scene_size']:
        pass
      else:
        value = value.astype(np.int)
        summary[style] = int(items[1])
      summary[style] = value
  return summary

def check_house_intact(base_dir):
  bbox_intact = check_bbox_intact(base_dir)
  pcl_intact = check_pcl_intact(base_dir)
  images_intact = check_images_intact(base_dir)
  house_intact = bbox_intact and pcl_intact and images_intact
  intacts = f'bbox_intact:{bbox_intact}, pcl_intact:{pcl_intact}, images_intact:{images_intact}'
  return house_intact, intacts

def check_bbox_intact(base_dir):
  summary = read_summary(base_dir)
  bbox_intact = 'level_num' in summary and 'wall_num' in summary
  return bbox_intact

def check_pcl_intact(base_dir):
  summary = read_summary(base_dir)
  pcl_intact = 'points_num' in summary and summary['points_num'] > MIN_POINT_NUM
  pcl_fn = os.path.join(base_dir, 'pcl_camref.ply')
  pcl_intact = pcl_intact and os.path.exists(pcl_fn)
  return pcl_intact

def check_images_intact(base_dir):
  cam_fn = os.path.join(base_dir, 'cam')
  if not os.path.exists(cam_fn):
    return False
  cam_pos = read_cam_pos(cam_fn)
  images_dir = os.path.join(base_dir, 'images')
  depth_images_num = len( glob.glob(f"{images_dir}/*_depth.png") )
  images_intact = cam_pos.shape[0] == depth_images_num
  return images_intact


def gen_bbox(house_fn):
    always_gen_bbox = Debug and 0

    parsed_dir = get_pcl_path(house_fn)
    summary = read_summary(parsed_dir)
    box_intact = 'level_num' in summary and 'wall_num' in summary
    if box_intact and (not always_gen_bbox):
      print(f'skip gen_bbox, summary intact: {parsed_dir}')
      return

    with open(house_fn) as f:
      house = json.loads(f.read())

    scaleToMeters = house['scaleToMeters']
    assert scaleToMeters == 1
    bboxes = defaultdict(list)
    bboxes['house'].append( Bbox3D.bbox_from_minmax( house['bbox'] ))

    for level in house['levels']:
      if 'bbox' not in level:
        continue
      bboxes['level'].append( Bbox3D.bbox_from_minmax( level['bbox'] ))
      nodes = level['nodes']
      for node in nodes:
        node_type = node['type']
        if node_type == 'Object':
          modelId = node['modelId']
          category = Suncg.modelId_2_class[modelId]
          bboxes[category].append(Bbox3D.bbox_from_minmax( node['bbox']))
        elif node_type == 'Room':
          if 'bbox' in node:
            bboxes['room'].append(Bbox3D.bbox_from_minmax( node['bbox']))
          room_bboxes = split_room_parts(house_fn, node['modelId'])
          for c in room_bboxes:
            bboxes[c] += room_bboxes[c]
        else:
          if 'bbox' in node:
            bboxes[node_type].append(Bbox3D.bbox_from_minmax( node['bbox']))

    centroid = (np.array(house['bbox']['min']) + np.array(house['bbox']['max']))/2.0
    mesh_frame = open3d.create_mesh_coordinate_frame(size = 0.6, origin = centroid)

    for obj in bboxes:
      if len(bboxes[obj])>0:
        bboxes[obj] = np.concatenate([b.reshape([1,7]) for b in bboxes[obj]], 0)
      else:
        bboxes[obj] = np.array(bboxes[obj]).reshape([-1,7])

      bboxes[obj] = cam2world_box(bboxes[obj])

    for obj in DSET_METAS0.class_2_label:
        if obj == 'background':
            continue
        if obj not in bboxes:
            bboxes[obj] = np.array(bboxes[obj]).reshape([-1,7])


    level_num = len(house['levels'])
    if level_num == 1:
      bboxes['wall'] = preprocess_walls(bboxes['wall'])
      bboxes['window'] = preprocess_windows(bboxes['window'], bboxes['wall'])
      bboxes['door'] = preprocess_doors(bboxes['door'], bboxes['wall'])
      bboxes['ceiling_raw']  = bboxes['ceiling'].copy()
      bboxes['floor_raw'] = bboxes['floor'].copy()
      bboxes['ceiling'] = preprocess_cfr(bboxes['ceiling'], bboxes['wall'], 'ceiling')
      bboxes['floor'] = preprocess_cfr(bboxes['floor'], bboxes['wall'], 'floor')

    # save bbox in ply and txt
    object_bbox_dir = os.path.join(parsed_dir, 'object_bbox')
    if not os.path.exists(object_bbox_dir):
      os.makedirs(object_bbox_dir)

    bboxes_num = {}
    for category in bboxes.keys():
      bboxes_num[category] = len(bboxes[category])
      boxes_fn = os.path.join(object_bbox_dir, category+'.txt')
      boxes = np.array(bboxes[category])
      np.savetxt(boxes_fn, boxes)

    #######################
    print(f'parsed_dir: {parsed_dir}')
    write_summary(parsed_dir, 'level_num', level_num, 'a')
    for obj in ['room', 'wall', 'window', 'door', 'floor', 'ceiling']:
        write_summary(parsed_dir, f'{obj}_num', bboxes_num[obj], 'a')

    #######################
    save_ply = False
    if save_ply:
      for category in bboxes.keys():
        for i,bbox in enumerate(bboxes[category]):
          box_dir = os.path.join(object_bbox_dir,'{}'.format(category))
          if not os.path.exists(box_dir):
            os.makedirs(box_dir)

          box_fn = os.path.join(box_dir, '%d.ply'%(i))
          bbox_cam = world2cam_box(bbox.reshape([1,7]))[0]
          Bbox3D.draw_bbox_open3d(bbox_cam, 'Y', plyfn=box_fn)


def split_room_parts(house_fn, modelId):
    tmp = house_fn.split('/')
    tmp[-3] = 'room'
    del tmp[-1]
    room_fn = '/'+ os.path.join(*tmp)

    room_parts_dir = os.path.join(room_fn, 'parts')
    #print(room_parts_dir)
    if not os.path.exists(room_parts_dir):
      os.makedirs(room_parts_dir)
    #else:
    #  fnum = len(glob.glob(room_parts_dir+'/*'))
    #  if fnum>0:
    #    os.system(f"rm {room_parts_dir}/*")

    obj_fns = glob.glob(room_fn+'/{}*.obj'.format(modelId))
    obj_fns.sort()
    meshes = []
    room_bboxes = {}
    for obj_fn in obj_fns:
      category = {"c":"ceiling", 'w':"wall", "f":"floor"}[obj_fn[-5]]
      room_bboxes[category] = read_room_obj(obj_fn, room_parts_dir)
    return room_bboxes

def gen_pcl(house_fn):
    always_gen_pcl = False
    check_points_out_of_house = False

    parsed_dir = get_pcl_path(house_fn)
    pcl_fn = os.path.join(parsed_dir, 'pcl_camref.ply')
    pcl_intact = check_pcl_intact(parsed_dir)
    if pcl_intact and (not always_gen_pcl):
        print(f'{pcl_fn} intact')
        return True

    # read house scope, abandon images with points out of house scope
    images_intact = check_images_intact(parsed_dir)
    if not images_intact:
      print(f'images not intact, abort generating pcl')
      return

    house_fn = os.path.join(parsed_dir, 'object_bbox/house.txt')
    house_box = np.loadtxt(house_fn)

    base_dir = os.path.dirname(house_fn)
    cam_fn = os.path.join(base_dir, parsed_dir+'/cam')
    #if not os.path.exists(cam_fn):
    #  print(f'{cam_fn} does not exist, abort gen_pcl')
    #  return False
    cam_pos = read_cam_pos(cam_fn)
    depth_fns = glob.glob(os.path.join(base_dir, parsed_dir+'/images/*_depth.png'))
    #if cam_pos.shape[0] < MIN_CAM_NUM or len(depth_fns) != cam_pos.shape[0]:
    #  print(f'{cam_pos.shape[0]} cams,  but {len(depth_fns)} images')
    #  return False
    depth_fns.sort()
    print(f'start gen pcl {pcl_fn}\n with {len(depth_fns)} images')

    gen_ply_each_image = False
    if gen_ply_each_image:
      pcl_path = parsed_dir + '/pcls'
      if not os.path.exists(pcl_path):
        os.makedirs(pcl_path)
    pcls_all = []

    max_point_num = 2e7
    cam_num = cam_pos.shape[0] * 1.0
    pre_downsample_num = max(2e5, int(max_point_num / cam_num))
    print(f'pre_downsample_num: {pre_downsample_num}')

    for i,depth_fn in enumerate(depth_fns):
      pcl_i = depth_2_pcl(depth_fn, cam_pos[i])

      if check_points_out_of_house:
        # check if the points are out of house scope
        # do not know why this happens sometims, just use this ineligant way to
        # solve it temperally
        if pcl_i.shape[0] > 1000:
          ids = np.sort(np.random.choice(pcl_i.shape[0], 1000, False))
        else:
          ids = np.arange(pcl_i.shape[0])
        mask = Bbox3D.points_in_bbox(pcl_i[ids,0:3], house_box.reshape([1,7]))
        if not mask.all():
          print(f'some points are out of box in image {i}')
          continue

      if pre_downsample_num < pcl_i.shape[0]:
        pcl_i = random_sample_pcl(pcl_i, pre_downsample_num)
      pcls_all.append(pcl_i)

      if gen_ply_each_image:
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(pcl_i)
        base_name = os.path.basename(depth_fns[i]).replace('depth.png','pcl.ply')
        pcl_fn = os.path.join(pcl_path, base_name)
        open3d.write_point_cloud(pcl_fn, pcd)
        open3d.draw_geometries([pcd])
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass

    pcls_all = np.concatenate(pcls_all, 0)
    if pcls_all.shape[0] > 5e6:
      print(f'{pcls_all.shape[0]} random to 5e6')
      pcls_all = random_sample_pcl(pcls_all, int(5e6))
    org_num = pcls_all.shape[0]
    print(f'org point num: {pcls_all.shape[0]/1000} K')
    if org_num < MIN_POINT_NUM:
      print(f'only {org_num} points, del cams and re-generate later\n del {parsed_dir}')
      import shutil
      shutil.rmtree(parsed_dir)
      return False

    #if org_num > 1e7:
    #    pcls_all = random_sample_pcl(pcls_all, int(max(1e7, org_num/10)) )
    #    print(f'random sampling to point num: {pcls_all.shape[0]/1000} K')

    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(pcls_all[:,0:3])
    pcd.colors = open3d.Vector3dVector(pcls_all[:,3:6])
    pcd = open3d.voxel_down_sample(pcd, voxel_size=0.02)
    new_num = np.asarray(pcd.points).shape[0]
    print(f'new point num: {new_num/1000.0} K')
    open3d.write_point_cloud(pcl_fn, pcd)
    #open3d.draw_geometries([pcd])

    write_summary(parsed_dir, 'points_num', new_num, 'a')
    pcl_size, xyarea = get_pcl_size(pcls_all[:,0:3])
    write_summary(parsed_dir, 'scene_size', pcl_size, 'a')
    write_summary(parsed_dir, 'xyarea', xyarea, 'a')
    #open3d.draw_geometries([pcd])
    return True

def get_pcl_size(xyzs):
  xyz_min = xyzs.min(0)
  xyz_max = xyzs.max(0)
  xyz_size = xyz_max - xyz_min
  xyarea = np.product(xyz_size[0:2])
  return xyz_size, xyarea

def read_cam_pos(cam_fn):
  cam_pos = np.loadtxt(cam_fn)
  cam_pos = np.reshape(cam_pos, [-1,12])
  #extrinsics = camPos2Extrinsics(cam_pos[0])
  return cam_pos

def depth_2_pcl(depth_fn, cam_pos):
    extrinsics = camPos2Extrinsics(cam_pos)
    depth0 = Image.open(depth_fn)
    # mm to m
    depth = np.array(depth0).astype(np.float) / 1000.0
    fc = camFocus(cam_pos, depth.shape)
    uni_depth = depth / fc

    h,w = depth.shape
    xyz = np.zeros([w,h,3], dtype=np.float)
    xyz[:,:,2] = depth.T
    xyz[:,:,0] = np.tile(np.reshape(np.arange(w),[w,1]),[1,h]) - 0.5*w + 0.5
    xyz[:,:,1] = np.tile(np.reshape(np.arange(h),[1,h]),[w,1]) - 0.5*h + 0.5
    xyz[:,:,0] = xyz[:,:,0] * uni_depth.T
    xyz[:,:,1] = xyz[:,:,1] * uni_depth.T

    #xyz_min = xyz.min((0,1))
    #xyz_max = xyz.max((0,1))
    #print(xyz_min)
    #print(xyz_max)

    ### convert xyz from camera corordinate to world corordinate
    xyz = np.reshape(xyz, [-1,3])
    # remove invalid points: xyz==0
    mask = xyz[:,2] > 0
    xyz = xyz[mask, :]

    num = xyz.shape[0]
    xyz = np.concatenate([xyz, np.ones([num,1])], 1)
    xyz = np.transpose(xyz)
    xyz = np.matmul( extrinsics, xyz )
    xyz = xyz.transpose()

    color_fn = depth_fn.replace('depth.png', 'color.jpg')
    color0 = Image.open(color_fn)
    color0 = np.array(color0) # uint8
    color = np.transpose(color0, [1,0,2])
    color = color.reshape(-1,3)
    color = color[mask] / 256.

    pcl = np.concatenate([xyz, color], 1).astype(np.float32)
    #show_pcl(pcl)
    return pcl

def get_pcl_path(house_fn, parsed_dir = PARSED_DIR_GOING):
    tmp = house_fn.split('/')
    tmp[-3] = parsed_dir
    del tmp[-1]
    tmp = ['/']+tmp
    parsed_dir = os.path.join(*tmp)
    if not os.path.exists(parsed_dir):
      os.makedirs(parsed_dir)
    return parsed_dir

def gen_house_obj(house_fn):
    house_dir = os.path.dirname(house_fn)
    parsed_dir = get_pcl_path(house_fn, PARSED_DIR_READY)

    obj_fn = os.path.join(parsed_dir, 'house.obj')
    if os.path.exists(obj_fn):
      print(f'{obj_fn} already exists')
      return

    os.system(f"cd {house_dir}; {Suncg.tool_bin}/scn2scn house.json {parsed_dir}/house.obj")

def gen_cam_images(house_fn):
    always_gen_new_cam = False
    always_gen_new_images = False


    is_add_extra_cam = 0

    xfov = 0.5 # 1.0 # 0.5
    width = 640 # 640
    height = 480 # 640 # 480

    min_visible_objects = 3 # 1 # 3
    angle_sampling = np.pi/3.0 #  np.pi/6.0 # np.pi / 3.0
    position_sampling = 0.25 # 0.25
    interpolation_step = 0.1 # 0.1

    create_world_in_hand_cameras = "" # "-create_world_in_hand_cameras" # ""
    create_room_cameras = "-create_room_cameras"
    create_object_cameras =  "-create_object_cameras" # ""
    interpolate_camera_trajectory = "" #  "-interpolate_camera_trajectory" # "" # disable, otherwise too many cams

    house_dir = os.path.dirname(house_fn)
    parsed_dir = get_pcl_path(house_fn)

    if is_add_extra_cam:
      cam_fn = os.path.join(parsed_dir, 'cam_org')
    else:
      cam_fn = os.path.join(parsed_dir, 'cam')
    images_dir = os.path.join(parsed_dir, 'images')
    if (not always_gen_new_cam) and os.path.exists(cam_fn):
      print(f'{cam_fn} already exists')
      gen_new_cam = False
    else:
      gen_new_cam = True

      print(f'generating {cam_fn}')
      os.system(f"cd {house_dir}; {Suncg.tool_bin}/scn2cam house.json {cam_fn} -categories {Suncg.tool_meta} -v  " +\
              f" -xfov {xfov} -width {width} -height {height} -position_sampling {position_sampling} " +\
              f" -angle_sampling {angle_sampling} -min_visible_objects {min_visible_objects} " +\
              f" {create_world_in_hand_cameras} {create_room_cameras} {create_object_cameras} {interpolate_camera_trajectory}")
    save_cam_ply(cam_fn)

    if gen_new_cam  and is_add_extra_cam:
      cam_fn = add_extra_cam_orientations(cam_fn)
      #cam_fn = add_exta_cam_locations(cam_fn)

    ###############
    cam_fn = os.path.join(parsed_dir, 'cam')
    if os.path.exists(cam_fn):
        cam_pos = read_cam_pos(cam_fn)
        cam_num = cam_pos.shape[0]
    else:
        cam_num = 0
    depth_images_num = len( glob.glob(f"{images_dir}/*_depth.png") )
    if depth_images_num > 0:
      print(f'cam num = {cam_num},  got {depth_images_num} depth images')
    if (not always_gen_new_images) and cam_num > MIN_CAM_NUM and depth_images_num == cam_num:
      return

    kinect_max_depth = 12 # 7
    max_vertex_spacing = 0.1 # 0.1
    kinect_min_reflection = 0.01  # 0.05

    if not os.path.exists(images_dir):
      os.makedirs(images_dir)
    else:
      import shutil
      shutil.rmtree(images_dir)

    print('generating images: {images_dir}')
    os.system(f"cd {house_dir}; {Suncg.tool_bin}/scn2img house.json {cam_fn} {parsed_dir}/images -categories {Suncg.tool_meta} -v " +\
              f" -xfov {xfov} -width {width} -height {height} " +\
              f" -kinect_max_depth {kinect_max_depth} -max_vertex_spacing {max_vertex_spacing} -kinect_min_reflection {kinect_min_reflection} " + \
              f" -capture_depth_images -capture_color_images  " )   #  -raycast

def add_extra_cam_orientations(cam_fn, show=False):
  cam_pos = read_cam_pos(cam_fn)
  cam_num = cam_pos.shape[0]
  extra_cams = []
  for i in range(cam_num):
    cam_pos_i = cam_pos[i,:]
    if i>0:
      dif = cam_pos_i - cam_pos[0:i]
      dis = dif[:,0:3]
      dis = np.linalg.norm(dis, axis=1)
      is_new = dis.min() > 1
    else:
      is_new = False
    if not is_new:
      continue

    cam_pos_i = np.reshape(cam_pos_i, [12,1])
    vv = cam_pos_i[0:3,:]
    tv = cam_pos_i[3:6,:]
    uv = cam_pos_i[6:9,:]
    rv = np.reshape(np.cross(tv[:,0],uv[:,0]), [3,1])
    tmp = cam_pos_i[9:,:]

    y0 = np.array([0,1,0]).reshape([3,1])
    tv1 = np.cross(tv[:,0], y0[:,0]).reshape([3,1])
    tv1 = tv1 / np.linalg.norm(tv1)
    extra_cam_i0 = np.reshape(np.concatenate([vv, tv1, y0, tmp], 0), [1,12])
    extra_cam_i1 = np.reshape(np.concatenate([vv, -tv1, -y0, tmp], 0), [1,12])
    extra_cams.append(extra_cam_i0)
    extra_cams.append(extra_cam_i1)

  extra_cams = np.concatenate(extra_cams, 0)
  cam_pos_new = np.concatenate([cam_pos, extra_cams], 0)
  cam_fn_new = os.path.dirname(cam_fn) +'/cam'
  np.savetxt(cam_fn_new, cam_pos_new, fmt='%.5f')
  print(f'add_extra_cam_orientations org cam num: {cam_num}, new cam num: {extra_cams.shape[0]} \n{cam_fn}')
  save_cam_ply(cam_fn, show)
  save_cam_ply(cam_fn_new, show)
  return cam_fn_new


def read_object_bbox(parsed_dir, category):
  fn = f'{parsed_dir}/object_bbox/{category}.txt'
  bboxes = np.loadtxt(fn).reshape(-1,7)
  return bboxes

def add_exta_cam_locations(cam_fn, show=False):
  parsed_dir = os.path.dirname(cam_fn)
  walls = read_object_bbox(parsed_dir, 'wall')
  if walls.shape[0] == 0:
      return cam_fn
  walls = world2cam_box(walls)
  #Bbox3D.draw_bboxes(walls, 'Y', False)
  wall_corners = Bbox3D.bboxes_corners(walls, up_axis='Y').reshape([-1,3])
  walls_min = wall_corners.min(0)
  walls_max = wall_corners.max(0)
  offset = 0.1
  step = 2
  tmp = np.ceil( (walls_max - walls_min+offset*2) / step)+1
  x_locs = np.arange(0, tmp[0]) * step + walls_min[0] - offset
  y_locs = np.arange(0, tmp[1]) * step + walls_min[1] - offset
  y_locs = [(walls_min[1] + walls_max[1])/2.0]
  z_locs = np.arange(0, tmp[2]) * step + walls_min[2] - offset

  cam_pos = read_cam_pos(cam_fn)
  cam_num = cam_pos.shape[0]
  cam_locs0 = cam_pos[:,0:3]
  extra_cams = []
  for x in x_locs:
    for y in y_locs:
      for z in z_locs:
        loc = np.array([[x,y,z]])

        # check if any wall is close to this loc
        dis = np.linalg.norm( loc - wall_corners, axis=1 )
        dis_min = dis.min()
        if dis_min > 2:
          # this loc is useles
          continue

        # check if any cam close to this loc already
        dis = np.linalg.norm( loc - cam_locs0, axis=1 )
        dis_min = dis.min()
        if dis_min > 1.5:
          cam_new = loc
          extra_cams.append(cam_new)

  back_cam = np.array([0,1,0, cam_pos[0,-3], cam_pos[0,-2], 10])
  if len(extra_cams)>0:
    extra_cams = np.concatenate(extra_cams, 0)
    cam_pos_new = []
    for loc in extra_cams:
      cam_forwards = [ [1,0,0], [-1,0,0], [0,0,1], [0,0,-1] ]
      for i in range(4):
        #if i==0:
        #  valid = loc[0] < walls_min[0]
        #elif i==1:
        #  valid = loc[0] > walls_max[0]
        #elif i==2:
        #  valid = loc[2] > walls_min[2]
        #elif i==3:
        #  valid = loc[2] > walls_max[2]

        valid = True
        if valid:
          forward = np.array(cam_forwards[i])
          cam_new_i = np.concatenate([loc, forward, back_cam],0 ).reshape([1,12])
          cam_pos_new.append( cam_new_i )

    cam_pos_new = np.concatenate(cam_pos_new, 0)
    cam_pos_new = np.concatenate([cam_pos_new, cam_pos], 0)
  else:
    cam_pos_new = cam_pos

  cam_fn_new = os.path.dirname(cam_fn) +'/cam'
  np.savetxt(cam_fn_new, cam_pos_new, fmt='%.5f')
  print(f'add_exta_cam_locations org cam num: {cam_num}, new cam num: {len(extra_cams)} \n{cam_fn}')
  save_cam_ply(cam_fn, show)
  save_cam_ply(cam_fn_new, show)
  return cam_fn_new

def gen_house_names_1level():
  parsed_path = f'{SUNCG_V1_DIR}/parsed'
  house_names0 = os.listdir(parsed_path)
  house_names0.sort()

  house_names = []
  for hn in house_names0:
    house_intact, intacts = check_house_intact(os.path.join(parsed_path, hn))
    if house_intact:
        if hn not in SceneSamples.bad_scenes:
            house_names.append(hn)


  remain_ids = []
  house_names_1l = []
  for hn in house_names:
    hfn = os.path.join(parsed_path, hn)
    summary = read_summary(hfn)
    level_num = summary['level_num']
    if level_num == 1:
      house_names_1l.append(hn)

  house_names_1l = np.array(house_names_1l)
  fn = os.path.join(SUNCG_V1_DIR, 'house_names_1level.txt')
  house_names_1l.tofile(fn, sep='\n', format='%s')

  print(f'totally {len(house_names0)} houses, got {len(house_names)} intact \n {house_names_1l.shape[0]} one level houses')
  print(f'save {fn}')

def gen_train_eval_split():
  house_path = f'{SUNCG_V1_DIR}/house'
  splited_path = f'{SUNCG_V1_DIR}_splited_torch'
  train_test_splited_path = os.path.join(splited_path, 'train_test_splited')

  house_names = np.array(os.listdir(house_path))
  n = len(house_names)
  n_train = int(n*0.7)
  n_val = n - n_train
  tmp = np.arange(n)
  np.random.shuffle(tmp)
  train_idx = np.sort(tmp[0:n_train])
  val_idx = np.sort(tmp[n_train:n])
  house_names_train = house_names[train_idx]
  house_names_val = house_names[val_idx]


  #house_names_train = np.array(['28297783bce682aac7fb35a1f35f68fa',  '7bff414da8570ef53c87c5ce5c15bc2a',  'ffbb0cd7ee77c6b0e5956275352704b8', 'ffe929c9ed4dc7dab9a09ade502ac444'])
  #house_names_val = np.array( ['28297783bce682aac7fb35a1f35f68fa',  '7bff414da8570ef53c87c5ce5c15bc2a',  'ffbb0cd7ee77c6b0e5956275352704b8', 'ffe929c9ed4dc7dab9a09ade502ac444'])

  house_names_val = house_names_train = np.array(['31a69e882e51c7c5dfdc0da464c3c02d', '7411df25770eaf8d656cac2be42a9af0', '7cd75b127f06a078929a6524396c738c', '8c033357d15373f4079b1cecef0e065a',\
                'aaa535ef80b7d34f57f5d3274eec0daf', 'c3802ae080bc1d5f4ada2f75448f7b49', 'e7b3e2566e174b6fbb2864de76b50334'])

  if not os.path.exists(train_test_splited_path):
    os.makedirs(train_test_splited_path)
  train_fn = os.path.join(train_test_splited_path, 'train.txt')
  val_fn = os.path.join(train_test_splited_path, 'val.txt')

  house_names_train.tofile(train_fn, sep='\n', format="%s")
  house_names_val.tofile(val_fn, sep='\n', format="%s")

  print(f'create ok: \n{train_fn}  \n{val_fn}')


def gen_a_house_obj():
  house_name = '000539881d82c92e43ff2e471a97fcf9'
  house_fn = f'{SUNCG_V1_DIR}/house/{house_name}/house.json'
  gen_house_obj(house_fn)

def check_house_status():
    parsed_dir = f'{SUNCG_V1_DIR}/parsed'
    scene_ids = os.listdir(parsed_dir)
    scene_ids.sort()
    N = len(scene_ids)
    N = 400
    for i in range(N):
      house_dir = os.path.join(parsed_dir, scene_ids[i])
      if os.path.exists(house_dir):
        dir_exist = True
        house_intact, intacts = check_house_intact(house_dir)
        if not house_intact:
          print(f'{i} house intact: {house_intact}, details:  {intacts}   {house_dir}')
      else:
        print(f'{i} house not exist')
        continue

      pass
      #import shutil
      #shutil.rmtree(house_dir)


def save_cam_ply(cam_fn=None, show=False, with_walls=True, with_pcl=True):
  if cam_fn is None:
    house_name = '0004d52d1aeeb8ae6de39d6bd993e992_A'
    parsed_dir = f'{SUNCG_V1_DIR}/parsed/{house_name}'
    cam_fn = os.path.join(parsed_dir, 'cam')
    cam_fn = '/home/z/parsed/28297783bce682aac7fb35a1f35f68fa/cam'
  cam_pos = read_cam_pos(cam_fn)
  parsed_dir = os.path.dirname(cam_fn)

  cam_num = cam_pos.shape[0]
  cam_centroid = cam_pos[:,0:3]
  # cam centroid box
  tmp = np.array([[0.1,0.1,0.1,0]])
  tmp = np.tile(tmp, [cam_num, 1])
  cam_cen_box = cam_pos[:,0:3]
  cam_cen_box = np.concatenate([cam_centroid, tmp], 1)
  cam_cen_box = Bbox3D.bboxes_lineset(cam_cen_box, 'Z', False, random_color=False)

  # cam orientations
  cam_ori = cam_centroid + cam_pos[:,3:6]
  cam_centroid = np.expand_dims(cam_centroid, 1)
  cam_ori = np.expand_dims(cam_ori, 1)

  cam_vec = np.concatenate([cam_centroid, cam_ori], 1)
  cam_lines = Bbox3D.draw_lines_open3d(cam_vec, color=[200,0,0], show=False)
  pcl_fn = os.path.dirname(cam_fn) + '/pcl_camref.ply'
  if show:
    if with_pcl and os.path.exists(pcl_fn):
      pcl = open3d.read_point_cloud(pcl_fn)
      open3d.draw_geometries(cam_cen_box + [ cam_lines, pcl])
    if with_walls:
      bbox_fn = f'{parsed_dir}/object_bbox/wall.txt'
      walls = np.loadtxt(bbox_fn).reshape([-1,7])
      walls = world2cam_box(walls)
      bboxes_lineset_ls = Bbox3D.bboxes_lineset(walls, 'Y', False)
      open3d.draw_geometries(cam_cen_box + [cam_lines] + bboxes_lineset_ls)

  # save cam pos as ply
  pcd = open3d.PointCloud()
  pcd.points = open3d.Vector3dVector(cam_pos[:,0:3])
  cam_ply_fn = cam_fn+'_pos.ply'
  open3d.write_point_cloud(cam_ply_fn, pcd)
  #open3d.draw_geometries([pcd])


def parse_house():
  '''
    cam, cam_org, cam_org_pos.ply, cam_pos.ply, pcl_camref in cam frame
    object_bbox in world frame
  '''
  suncg = Suncg(SUNCG_V1_DIR)
  #suncg.parse_houses_pool()
  suncg.parse_houses(False)

if __name__ == '__main__':
  parse_house()
  #gen_house_names_1level()

  #check_house_status()

  cam_fn = '/home/z/SUNCG/suncg_v1/parsed/0004d52d1aeeb8ae6de39d6bd993e992/cam'
  #add_exta_cam_locations(cam_fn, True)
  #save_cam_ply(cam_fn, True)
  #add_extra_cam_orientations(cam_fn, True)
  #gen_a_house_obj()
  pass



