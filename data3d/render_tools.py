import sys, os, glob, json
sys.path.insert(0, '..')
import open3d, pymesh
import numpy as np
from utils3d.bbox3d_ops import Bbox3D
from utils3d.geometric_util import cam2world_box, cam2world_pcl
import torch
from collections import defaultdict
from suncg_utils.scene_samples import SceneSamples
from data3d.dataset_metas import DSET_METAS0
from  maskrcnn_benchmark.structures.bounding_box_3d import BoxList3D

SUNCG_V1_DIR = '/DS/SUNCG/suncg_v1'
PARSED_DIR = f'/DS/SUNCG/suncg_v1/parsed'
#PARSED_DIR = f'/DS/SUNCG/suncg_v1/_parsed_NewPcl__'
SPLITED_DIR = '/DS/SUNCG/suncg_v1_torch_splited'

#CLASSES = ['wall', 'ceiling']
CLASSES = ['wall', 'window', 'door']
CLASSES = ['wall']
#CLASSES += ['ceiling']
#CLASSES += ['floor']
#CLASSES += ['room']

#CLASSES = ['ceiling','floor']
#CLASSES = ['floor']
#CLASSES = ['ceiling']
#CLASSES = None

SHOW_PCL = 1
POINTS_KEEP_RATE = 0.95

AniSizes = {'01b05d5581c18177f6e8444097d89db4': [120, 920, 640,1300] }

def show_walls_1by1(wall_bboxes):
  n = wall_bboxes.shape[0]
  for i in range(n):
    tmp = wall_bboxes.copy()
    tmp[:,2] -= 1
    show_box = np.concatenate([tmp, wall_bboxes[i:i+1]], 0)
    print(f'wall {i}/{n}\n{wall_bboxes[i]}')
    Bbox3D.draw_bboxes(show_box, 'Z', False)

def show_walls_offsetz(wall_bboxes):
  n = wall_bboxes.shape[0]
  wall_bboxes = wall_bboxes.copy()
  wall_bboxes[:,2] += np.random.rand(n)*1
  print(f'totally {n} boxes')
  Bbox3D.draw_bboxes(wall_bboxes, 'Z', False)


def cut_points_roof(points, keep_rate=0.9):
  z_min = np.min(points[:,2])
  z_max = np.max(points[:,2])
  threshold = z_min + (z_max - z_min) * keep_rate
  mask = points[:,2] < threshold
  points_cutted = points[mask]
  return points_cutted

def down_sample_points(points, max_size=int(5e5)):
  n = points.shape[0]
  if n<=max_size:
    return points
  choices = np.random.choice(n, max_size, replace=False)
  points_d = points[choices]
  return points_d


def render_parsed_house_walls(parsed_dir, show_pcl=SHOW_PCL, show_by_class=0):
  print(f'parsed_dir:{parsed_dir}')
  house_name = os.path.basename(parsed_dir)
  bboxes = []
  labels = []
  for obj in CLASSES:
    bbox_fn_ = f'{parsed_dir}/object_bbox/{obj}.txt'
    bboxes_  = np.loadtxt(bbox_fn_).reshape([-1,7])
    bboxes.append(bboxes_)
    label = DSET_METAS0.class_2_label[obj]
    labels += [label] * bboxes_.shape[0]
  bboxes = np.concatenate(bboxes, 0)
  labels = np.array(labels).astype(np.int8)
  if bboxes.shape[0] > 0:
    scene_size = Bbox3D.boxes_size(bboxes)
    print(f'scene wall size:{scene_size}')

    #Bbox3D.draw_bboxes(bboxes, up_axis='Z', is_yx_zb=False, labels=labels)
    #if not show_pcl:
    #Bbox3D.draw_bboxes_mesh(bboxes, up_axis='Z', is_yx_zb=False)
    #Bbox3D.draw_bboxes_mesh(bboxes, up_axis='Z', is_yx_zb=False, labels=labels)
    #show_walls_offsetz(bboxes)
    pass

    if show_by_class:
          for c in range(1,max(labels)+1):
              cs = DSET_METAS0.label_2_class[c]
              print(cs)
              if cs not in ['wall', 'window', 'door']:
              #if cs not in ['wall']:
                continue
              mask = labels == c
              bboxes_c = bboxes[mask]
              show_walls_offsetz(bboxes_c)

  if show_pcl:
    pcl_fn = f'{parsed_dir}/pcl_camref.ply'
    if not os.path.exists(pcl_fn):
        return

    pcd = open3d.io.read_point_cloud(pcl_fn)
    points = np.asarray(pcd.points)
    points = cam2world_pcl(points)
    colors = np.asarray(pcd.colors)
    pcl = np.concatenate([points, colors], 1)

    scene_size = pcl_size(pcl)
    print(f'scene pcl size:{scene_size}')
    print(f'point num: {pcl.shape[0]}')

    #pcl = cut_points_roof(pcl)

    Bbox3D.draw_points(pcl,  points_keep_rate=POINTS_KEEP_RATE, points_sample_rate=0.05)
    #Bbox3D.draw_points(pcl,  points_keep_rate=POINTS_KEEP_RATE, animation_fn='points.mp4', ani_size=AniSizes[house_name])
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    bboxes[:,2] += 0.1
    #Bbox3D.draw_points_bboxes(pcl, bboxes, up_axis='Z', is_yx_zb=False, points_keep_rate=POINTS_KEEP_RATE)
    #Bbox3D.draw_points_bboxes_mesh(pcl, bboxes, up_axis='Z', is_yx_zb=False, points_keep_rate=POINTS_KEEP_RATE )
    #Bbox3D.draw_points_bboxes_mesh(pcl, bboxes, up_axis='Z', is_yx_zb=False, points_keep_rate=POINTS_KEEP_RATE, animation_fn='mesh.mp4', ani_size=AniSizes[house_name] )


def pcl_size(pcl):
    xyz_max = pcl[:,0:3].max(0)
    xyz_min = pcl[:,0:3].min(0)
    xyz_size = xyz_max - xyz_min
    return xyz_size

def render_pth_file(pth_fn, show_by_class=0):
  pcl, bboxes0 = torch.load(pth_fn)
  pcl = down_sample_points(pcl)

  if 'clutter' in bboxes0:
    del bboxes0['clutter']
  #points = pcl[:,0:3]
  #colors = pcl[:,3:6]
  #normals = pcl[:,6:9]

  if CLASSES is None:
    bboxes =  bboxes0
  else:
    bboxes = {}
    for c in CLASSES:
      if c in bboxes0:
        bboxes[c] = bboxes0[c]

  scene_size = pcl_size(pcl)
  print(f'scene pcl size:{scene_size}')
  print(f'point num: {pcl.shape[0]}')

  #pcl = cut_points_roof(pcl)

  classes = [k for k in bboxes.keys()]
  num_classes = {k:bboxes[k].shape[0] for k in bboxes.keys()}
  print(f'\nclasses: {num_classes}\n\n')

  all_bboxes = np.concatenate([boxes for boxes in bboxes.values()], 0)
  nums = [boxes.shape[0] for boxes in bboxes.values()]
  labels = []
  for i, n in enumerate(nums):
    labels += [i]*n
  labels = np.array(labels)

  #Bbox3D.draw_points(pcl,  points_keep_rate=POINTS_KEEP_RATE)
  #show_walls_offsetz(all_bboxes)
  #Bbox3D.draw_bboxes_mesh(all_bboxes, up_axis='Z', is_yx_zb=False, labels=labels)
  #Bbox3D.draw_bboxes_mesh(all_bboxes, up_axis='Z', is_yx_zb=False)
  #Bbox3D.draw_points_bboxes_mesh(pcl, all_bboxes, up_axis='Z', is_yx_zb=False, labels=labels, points_keep_rate=POINTS_KEEP_RATE)
  #Bbox3D.draw_points_bboxes_mesh(pcl, all_bboxes, up_axis='Z', is_yx_zb=False, points_keep_rate=POINTS_KEEP_RATE)
  #Bbox3D.draw_points_bboxes(pcl, all_bboxes, up_axis='Z', is_yx_zb=False,points_keep_rate=POINTS_KEEP_RATE)
  Bbox3D.draw_points_bboxes(pcl, all_bboxes, up_axis='Z', is_yx_zb=False, labels=labels, points_keep_rate=POINTS_KEEP_RATE)
  #Bbox3D.draw_points_bboxes(pcl, all_bboxes, up_axis='Z', is_yx_zb=False, labels=labels, points_keep_rate=POINTS_KEEP_RATE, animation_fn='anima.mp4', ani_size=[280,700,550,1350])
  #Bbox3D.draw_bboxes(all_bboxes, up_axis='Z', is_yx_zb=False, labels=labels)

  #boxlist = BoxList3D(all_bboxes, size3d=None, mode='standard', examples_idxscope=None, constants={})
  #boxlist.show_with_corners()

  #show_walls_offsetz(all_bboxes)

  if show_by_class:
    for clas in bboxes.keys():
      #if clas not in ['wall']:
      #if clas not in ['wall', 'window', 'door','ceiling', 'floor']:
      #  continue
      #if clas not in CLASSES:
      #  continue
      boxes = bboxes[clas]
      bn = boxes.shape[0]
      print(clas, f'num={bn}')
      #Bbox3D.draw_points_bboxes(points, boxes, up_axis='Z', is_yx_zb=False)
      #Bbox3D.draw_points_bboxes_mesh(pcl, boxes, up_axis='Z', is_yx_zb=False, points_keep_rate=POINTS_KEEP_RATE)
      try:
        Bbox3D.draw_points_bboxes(pcl, boxes, up_axis='Z', is_yx_zb=False, points_keep_rate=POINTS_KEEP_RATE)
      except:
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass
      #show_walls_offsetz(boxes)
  pass

def render_suncg_raw_house_walls(house_fn):
    from suncg import split_room_parts, Suncg
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
      bboxes[obj] = np.concatenate([b.reshape([1,7]) for b in bboxes[obj]], 0)
      bboxes[obj] = cam2world_box(bboxes[obj])
    walls = bboxes['wall']

    print('\nThe raw SUNCG walls\n')
    #Bbox3D.draw_bboxes(walls, up_axis='Z', is_yx_zb=False)
    Bbox3D.draw_bboxes_mesh(walls, up_axis='Z', is_yx_zb=False)

def render_cam_positions(parsed_dir):
  from suncg import save_cam_ply
  cam_fn = f'{parsed_dir}/cam'
  save_cam_ply(cam_fn, show=True, with_pcl=True)


def render_houses(r_cam=True, r_whole=True, r_splited=True):
  '''
  angle%90 != 0:
        72148738e98fe68f38ec17945d5c9730 *
        b021ab18bb170a167d569dcfcaf58cd4 *
        8c033357d15373f4079b1cecef0e065a **
        b021ab18bb170a167d569dcfcaf58cd4 ** small angle
  complicate architecture:
      31a69e882e51c7c5dfdc0da464c3c02d **
  '''
  house_names = ['b021ab18bb170a167d569dcfcaf58cd4'] #

  #house_names = os.listdir(PARSED_DIR)

  with open(f'{SUNCG_V1_DIR}/house_names_1level.txt', 'r') as h1f:
      house_names_1level = h1f.read().split('\n')
  house_names = house_names_1level

  house_names.sort()

  #house_names = house_names[1100+450:]

  #house_names = SceneSamples.very_hard_wall_window_close
  #house_names = SceneSamples.complex_structures
  #house_names = SceneSamples.hard_samples_window_wall_close
  #house_names = SceneSamples.sj_paper_used
  house_names = SceneSamples.paper_samples_1

  #house_names = ['0138ea33414267375b879ff7ccc1436c']
  #house_names = ['2f3ae02201ad551e99870189e184af4f']
  #house_names = ['0055398beb892233e0664d843eb451ca']
  #house_names = ['0058113bdc8bee5f387bb5ad316d7b28']
  #house_names = ['0219bb573b54812dff157d30450dcbfd']
  #house_names = ['04e51704b00e8cea6375f0047a836c55']
  #house_names = ['0058113bdc8bee5f387bb5ad316d7b28']

  print(f'totally {len(house_names)} houses')


  for k,house_name in enumerate( house_names ):
    print(f'\n{k}: {house_name}')
    raw_house_fn = f'{SUNCG_V1_DIR}/house/{house_name}/house.json'
    #render_suncg_raw_house_walls(raw_house_fn)

    parsed_dir = f'{PARSED_DIR}/{house_name}'

    if r_cam:
      render_cam_positions(parsed_dir)

    if r_whole:
      render_parsed_house_walls(parsed_dir)

    splited_boxfn = f'{SPLITED_DIR}/houses/{house_name}/*.pth'
    pth_fns = glob.glob(splited_boxfn)
    if r_splited:
      for i,pth_fn in enumerate( pth_fns ):
        print(f'\nThe {i}-th / {len(pth_fns)} splited scene')
        render_pth_file(pth_fn)


def main():
    render_houses(
            r_cam=False,
            r_whole = 1,
            r_splited = 0
    )

def summarize():
  from suncg_utils.suncg_preprocess import read_summary, write_summary
  with open(f'{SUNCG_V1_DIR}/house_names_1level.txt', 'r') as h1f:
      house_names_1level = h1f.read().split('\n')

  show_big_size = True
  num_points = []
  xyareas = []
  scene_sizes = []
  i = 0
  for hn in house_names_1level:
    i += 1
    #if i>10:
    #  break

    parsed_dir = f'{PARSED_DIR}/{hn}'
    summary = read_summary(parsed_dir)

    if 'xyarea' in summary and 'scene_size' in summary:
      pn = summary['points_num']
      xyarea = summary['xyarea']
      scene_size = summary['scene_size']
    else:
      pcl_fn = f'{parsed_dir}/pcl_camref.ply'
      pcd = open3d.read_point_cloud(pcl_fn)

      points = np.asarray(pcd.points)
      points = cam2world_pcl(points)
      colors = np.asarray(pcd.colors)
      pcl = np.concatenate([points, colors], 1)

      scene_size = pcl_size(pcl)
      xyarea = np.product(scene_size[0:2])
      pn = pcl.shape[0]
      #if 'pcl_size' not in summary:
      #  write_summary(parsed_dir, 'pcl_size', scene_size, 'a')
      if 'xyarea' not in summary or 'scene_size' not in summary:
        write_summary(parsed_dir, 'xyarea', xyarea, 'a')
        write_summary(parsed_dir, 'scene_size', scene_size, 'a')


    scene_sizes.append(scene_size.reshape([1,-1]))
    xyareas.append(xyarea)
    num_points.append(pn)
    print(f'{i} {hn}  point num: {pn}')

    is_big_size = (scene_size > [80,80,10]).any()
    if show_big_size and is_big_size:
      print(f'\n\t\t{hn}\norg scene_size: {scene_size}')
      files = glob.glob( f'{SPLITED_DIR}/houses/{hn}/*.pth')
      for fl in files:
        render_pth_file(fl)



  num_points = np.array(num_points).astype(np.double)
  xyareas = np.array(xyareas).astype(np.double)
  scene_sizes = np.concatenate(scene_sizes, 0)
  mean_scene_size = scene_sizes.mean(axis=0)

  ave_np = np.mean(num_points).astype(np.int)
  sum_np = np.sum(num_points) / 1e6
  ave_xyarea = np.mean(xyareas)
  sum_xyarea = np.sum(xyareas) / 1e3

  scene_n = num_points.shape[0]
  big_xyarea = xyareas > 900
  big_xyarea_n = np.sum(big_xyarea)
  big_xyarea_ratio = 1.0 * big_xyarea_n / scene_n

  big_size_xy_num = (scene_sizes[:,:2].max(1) > 4096/50.0).sum()
  big_size_z_num = (scene_sizes[:,2:3].max(1) > 512/50.0).sum()

  print(f'\n\nTotall {scene_n} scenes')
  print(f'ave num: {ave_np:.3f}\nsum n: {sum_np:.5f} M')
  print(f'ave xyarea: {ave_xyarea:.5f}\n sum xyarea: {sum_xyarea:.5f} K')
  print(f'mean_scene_size: {mean_scene_size}')
  print(f'big area >900 ratio: {big_xyarea_ratio}')
  print(f'size x/y >  81.92 m: {big_size_xy_num}')
  print(f'size z > 10.24 m: {big_size_z_num}')
  pass

def check_data():
    files = glob.glob( f'{SPLITED_DIR}/houses/*/*.pth')
    for i,fn in enumerate( files ):
      print(f'{i}\t {fn}')
      size_i = os.path.getsize(fn)
      if size_i < 5e6:
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass
      #torch.load(fn)

def render_fn():
    path = '/DS/SUNCG/suncg_v1__torch_BS_50_50_BN_500K/houses'
    path = '/DS/2D-3D-Semantics_Stanford/2D-3D-Semantics_Pth/houses'
    f = 1
    if f==0:
      val_fn = '/DS/SUNCG/suncg_v1__torch_BS_50_50_BN_500K/train_test_splited/val.txt'
      #val_fn = '/DS/SUNCG/suncg_v1__torch_BS_50_50_BN_500K/train_test_splited/train.txt'
      val_hns = np.loadtxt(val_fn, dtype=str).tolist()
      house_names = val_hns[400:]
    elif f==1:
      house_names = os.listdir(path)
      house_names.sort()
      house_names = ['whole_areas']
      house_names = ['area_1']
      #house_names = ['1d84d7ca97f9e05534bf408779406e30', '1d938aa8a23c8507e035f5a7d4614180', '1dba3a1039c6ec1a3c141a1cb0ad0757', '1e694c1e4862169a5f153c8719887bfc', '1e717bef798945693244d7702bb65605']
      #house_names=  ['10afd977812749919ec417579d6dd070']
    elif f==2:
      house_names = SceneSamples.sj_paper_samples

    for hn in house_names:
      print(hn)
      file_names = glob.glob(f'{path}/{hn}/*.pth')
      for fn in file_names:
        render_pth_file(fn, 0)
      pass



if __name__ == '__main__':
    #render_fn()
    main()
    #summarize()
    #check_data()




