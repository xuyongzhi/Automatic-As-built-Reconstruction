import torch
from maskrcnn_benchmark.structures.bounding_box_3d import BoxList3D
from .suncg_metas import SUNCG_METAS
from .scene_samples import SceneSamples
from utils3d.bbox3d_ops import Bbox3D
import numpy as np
import logging

import os, glob

DEBUG = True


SHOW_RAW_INPUT = DEBUG and False
SHOW_AUG_INPUT = DEBUG and False
ADD_PAPER_SCENES = False

ENABLE_POINTS_MISSED = DEBUG and True

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
SuncgTorch_PATH = os.path.join(CUR_DIR, 'SuncgTorch')
ELEMENTS_IDS = {'xyz':[0,1,2], 'color':[3,4,5], 'normal':[6,7,8]}

POINTS_NUM = int(1e5)

def points_sample(pcl):
  if pcl.shape[0] > POINTS_NUM:
    ids = np.random.choice(pcl.shape[0], POINTS_NUM)
    ids_t = torch.from_numpy(ids)
    pcl = pcl[ids_t]
  return pcl


class SUNCGDataset(torch.utils.data.Dataset):
  def __init__(self, split, cfg):
    logger = logging.getLogger("maskrcnn_benchmark.input")
    self.is_train = is_train = split == 'train'
    self.scale = cfg.SPARSE3D.VOXEL_SCALE
    full_scale=cfg.SPARSE3D.VOXEL_FULL_SCALE
    val_reps = cfg.SPARSE3D.VAL_REPS
    batch_size = cfg.SOLVER.IMS_PER_BATCH if is_train else cfg.TEST.IMS_PER_BATCH
    self.objects_to_detect = cfg.INPUT.CLASSES
    dimension=3
    self.dset_metas = SUNCG_METAS(cfg.INPUT.CLASSES)
    self.elements = cfg.INPUT.ELEMENTS
    self.elements_ids = np.array([ELEMENTS_IDS[e] for e in self.elements]).reshape(-1)
    self.elements_ids.sort()

    self.full_scale = np.array(full_scale)
    assert self.full_scale.shape == (3,)

    dset_path = SuncgTorch_PATH
    with open(f'{dset_path}/train_test_splited/{split}.txt') as f:
      scene_names = [l.strip() for l in f.readlines()]
    scene_names = rm_bad_samples( scene_names )
    small_scenes = cfg.INPUT.SCENES
    if len(small_scenes)>0:
        logger.info(f'\nsmall scenes:\n{small_scenes}\n')
        scene_names = small_scenes
    if ADD_PAPER_SCENES and len(scene_names) > 10:
      add_paper_samples(scene_names)
    files = []
    for scene in scene_names:
      files += glob.glob(f'{dset_path}/houses/{scene}/*.pth')
    self.files = files
    assert len(self.files) > 0, 'no input data'

  def get_img_info(self, index):
    fn = self.files[index]
    basename = os.path.basename(fn)
    scene = os.path.basename(os.path.dirname(fn))
    info = f"{scene}/{basename}"
    return info

  def sampling(self, indices):
    self.files_org = self.files.copy()
    self.files =  [self.files[i] for i in indices]

  def back_to_org(self):
    self.files = self.files_org

  def __getitem__(self, index):
        is_train = self.is_train
        scale = self.scale
        full_scale = self.full_scale
        objects_to_detect = self.objects_to_detect

        zoom_rate = 0.1*0
        flip_x = False and is_train
        random_rotate = False and is_train
        distortion = False and is_train
        origin_offset = False and is_train
        norm_noise = 0.01 * int(is_train) * 0

        fn = self.files[index]
        print(f'Loading {fn} in SUNCGDataset')
        hn = os.path.basename(os.path.dirname(fn))
        #if self.is_train:
        #  print(f'\n(suncg_dataset.py) train {index}-th   {hn}\n')
        #else:
        #  print(f'\n(suncg_dataset.py) test  {index}-th  {hn}\n')
        pcl_i, bboxes_dic_i_0 = torch.load(fn)
        #points_sample(pcl_i)
        for obj in bboxes_dic_i_0:
          if type(  bboxes_dic_i_0[obj]  ) == torch.Tensor:
            bboxes_dic_i_0[obj] = bboxes_dic_i_0[obj].data.numpy()

        a = pcl_i[:,0:3].copy()
        b = pcl_i
        bboxes_dic_i = {}
        for obj in objects_to_detect:
            if not ( obj in bboxes_dic_i_0 or obj=='background'):
                print(f"unknow class {obj}")
                import pdb; pdb.set_trace()  # XXX BREAKPOINT
                assert False
        for obj in bboxes_dic_i_0:
          if ('all' in objects_to_detect) or (obj in objects_to_detect):
            bboxes_dic_i[obj] = Bbox3D.convert_to_yx_zb_boxes(bboxes_dic_i_0[obj])
            if obj in ['ceiling', 'floor', 'room']:
              bboxes_dic_i[obj] = Bbox3D.set_yaw_zero(bboxes_dic_i[obj])
        if SHOW_RAW_INPUT:
          show_pcl_boxdic(pcl_i, bboxes_dic_i)

        #---------------------------------------------------------------------
        # augmentation of xyz
        m=np.eye(3)+np.random.randn(3,3)*zoom_rate # aug: zoom
        if flip_x:
          m[0][0]*=np.random.randint(0,2)*2-1  # aug: x flip
        m*=scale
        if random_rotate:
          theta=np.random.rand()*2*math.pi # rotation aug
          m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
        a=np.matmul(a,m)
        if distortion:
          a=elastic(a,6*scale//50,40*scale/50)
          a=elastic(a,20*scale//50,160*scale/50)
        m=a.min(0)
        M=a.max(0)
        q=M-m
        # aug: the centroid between [0,full_scale]
        offset = -m
        if origin_offset:
          offset += np.clip(full_scale-M+m-0.001, 0, None) * np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
        a+=offset

        xyz_min = a.min(0) / scale
        xyz_max = a.max(0) / scale
        size3d = np.expand_dims(np.concatenate([xyz_min, xyz_max], 0), 0).astype(np.float32)
        size3d = torch.from_numpy(size3d)
        #---------------------------------------------------------------------
        # augmentation of feature
        # aug norm
        b[:,3:6] += np.random.randn(3)*norm_noise

        #---------------------------------------------------------------------
        # get elements
        b = b[:, self.elements_ids]
        if 'xyz' in self.elements:
          # import augmentation of xyz to feature
          b[:,0:3] = a / scale

        #---------------------------------------------------------------------
        # augment gt boxes
        for obj in bboxes_dic_i:
          #print(bboxes_dic_i[obj][:,0:3])
          bboxes_dic_i[obj][:,0:3] += np.expand_dims(offset,0)/scale
          #print(bboxes_dic_i[obj][:,0:3])
          pass

        #---------------------------------------------------------------------
        assert a.min() >= 0, f"point location should not < 0: {a.min()}"
        up_check = np.all(a < full_scale[np.newaxis,:], 1)
        if not np.all(up_check):
            max_scale = a.max(0)
            print(f'file: {self.files[index]}')
            print(f'\nmax scale: {max_scale} > full_scale: {full_scale}, some points will be missed\n')
            if  not ENABLE_POINTS_MISSED:
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              assert False

        idxs = (a.min(1)>=0)*(up_check)
        a=a[idxs]
        b=b[idxs]
        #c=c[idxs]
        a=torch.from_numpy(a).long()
        #locs = torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(index)],1)
        locs = a
        feats = torch.from_numpy(b)

        #---------------------------------------------------------------------
        bboxlist3d = bbox_dic_to_BoxList3D(bboxes_dic_i, size3d, self.dset_metas)
        labels = bboxlist3d
        if SHOW_AUG_INPUT:
          show_pcl_boxdic(pcl_i, bboxes_dic_i)
          bboxlist3d.show()
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass

        #batch_scopes(locs, scale)
        data = {'x': [locs,feats], 'y': labels, 'id': index, 'fn':fn}
        return data

  def get_groundtruth(self, index):
    data = self[index]
    gt_boxes = data['y']

    n = len(gt_boxes)
    difficult = torch.zeros(n)
    gt_boxes.add_field('difficult', difficult)
    return gt_boxes

  def __len__(self):
    return len(self.files)

  def map_class_id_to_class_name(self, class_id):
    class_name = self.dset_metas.label_2_class[class_id]
    return class_name


def add_paper_samples(scene_names):
  for s in SceneSamples.paper_samples:
    if s not in scene_names:
      scene_names += [s]

#Elastic distortion
blur0=np.ones((3,1,1)).astype('float32')/3
blur1=np.ones((1,3,1)).astype('float32')/3
blur2=np.ones((1,1,3)).astype('float32')/3


def elastic(x,gran,mag):
    bb=np.abs(x).max(0).astype(np.int32)//gran+3
    noise=[np.random.randn(bb[0],bb[1],bb[2]).astype('float32') for _ in range(3)]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    ax=[np.linspace(-(b-1)*gran,(b-1)*gran,b) for b in bb]
    interp=[scipy.interpolate.RegularGridInterpolator(ax,n,bounds_error=0,fill_value=0) for n in noise]
    def g(x_):
        return np.hstack([i(x_)[:,None] for i in interp])
    return x+g(x)*mag


def bbox_dic_to_BoxList3D(bbox_dic, size3d, dset_metas):
  bboxes = []
  labels = []
  for obj in bbox_dic:
    bboxes.append(bbox_dic[obj])
    label_i = dset_metas.class_2_label[obj]
    assert np.all(label_i>0), "label >1, 0 is for negative, -1 is ignore"
    labels.append(np.array([label_i]*bbox_dic[obj].shape[0]))
  bboxes = np.concatenate(bboxes, 0)
  labels = np.concatenate(labels, 0)

  examples_idxscope = torch.tensor([0, bboxes.shape[0]]).view(1,2)
  bboxlist3d = BoxList3D(bboxes, size3d=size3d, mode='yx_zb',
                        examples_idxscope=examples_idxscope, constants={})
  bboxlist3d.add_field('labels', labels)
  return bboxlist3d


def batch_scopes(location, voxel_scale):
  batch_size = torch.max(location[:,3])+1
  s = 0
  e = 0
  scopes = []
  for i in range(batch_size):
    e += torch.sum(location[:,3]==i)
    xyz = location[s:e,0:3].float() / voxel_scale
    s = e.clone()
    xyz_max = xyz.max(0)[0]
    xyz_min = xyz.min(0)[0]
    xyz_scope = xyz_max - xyz_min
    print(f"min:{xyz_min}  max:{xyz_max} scope:{xyz_scope}")
    scopes.append(xyz_scope)
  scopes = torch.cat(scopes, 0)
  return scopes


def rm_bad_samples(scene_names):
  scene_names_new = []
  for sn in scene_names:
    if sn not in SceneSamples.bad_scenes:
      scene_names_new.append(sn)
  return scene_names_new


def show_pcl_boxes(pcl, boxes):
  from utils3d.bbox3d_ops import Bbox3D
  Bbox3D.draw_points_bboxes(pcl[:,0:3], boxes, 'Z', is_yx_zb=True)
  pass


def show_pcl_boxdic(pcl, bboxes_dic):
  from utils3d.bbox3d_ops import Bbox3D
  boxes = []
  for obj in bboxes_dic:
    print(f'{obj}: {len(bboxes_dic[obj])}')
    boxes.append(bboxes_dic[obj])
  boxes = np.concatenate(boxes, 0)
  Bbox3D.draw_points_bboxes(pcl[:,0:6], boxes, 'Z', is_yx_zb=True)
  Bbox3D.draw_points_bboxes(pcl[:,0:6], bboxes_dic['wall'], 'Z', is_yx_zb=True)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

