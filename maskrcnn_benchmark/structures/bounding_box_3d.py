# Copyright (c) Facebook, BoxList3DInc. and its affiliates. All Rights Reserved.
import torch, math
import numpy as np

from utils3d.geometric_torch import limit_period, OBJ_DEF
from utils3d.bbox3d_ops_torch import Box3D_Torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

POINTS_KEEP_RATE = 0.9
POINTS_SAMPLE_RATE = 0.3
DEBUG = False

# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def cat_boxlist_3d(bboxes_ls, per_example, use_constants0=False):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes_ls (list[BoxList])
        per_example: if True, each element in bboxes_ls is an example, combine to a batch
        use_constants0: sometimes, the constants in each elements are different, if True, enable use the first one and do not check equality.
    """
    assert isinstance(bboxes_ls, (list, tuple))
    assert all(isinstance(bbox, BoxList3D) for bbox in bboxes_ls)

    none_size3d = any([b.size3d is None for b in bboxes_ls])
    if none_size3d:
      size3d = None
    else:
        if not per_example:
          size3d = bboxes_ls[0].size3d
          for bbox3d in bboxes_ls:
            #is_size_close =  torch.abs(bbox3d.size3d - size3d).max() < 0.01
            #if not is_size_close:
            if not torch.isclose( bbox3d.size3d, size3d ).all():
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              pass
        else:
          size3d = torch.cat([b.size3d for b in bboxes_ls])

    mode = bboxes_ls[0].mode
    assert all(bbox.mode == mode for bbox in bboxes_ls)

    fields = set(bboxes_ls[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes_ls)

    constants_all = bboxes_ls[0].constants
    if not use_constants0:
      if not all(constants_all == bbox.constants for bbox in bboxes_ls):
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        assert False
        pass

    batch_size0 = bboxes_ls[0].batch_size()
    for bbox in bboxes_ls:
      assert bbox.batch_size() == batch_size0

    # flatten order: [scale_num, sparse_location_num * yaws_num]
    bbox3d_cat = _cat([bbox3d.bbox3d for bbox3d in bboxes_ls], dim=0)
    if not per_example:
      examples_idxscope = torch.tensor([[0, bbox3d_cat.shape[0]]], dtype=torch.int32)
      batch_size = batch_size0
      assert batch_size0 == 1, "check if >1 if need to"
    else:
      assert batch_size0 == 1, "check if >1 if need to"
      batch_size = len(bboxes_ls)
      examples_idxscope = torch.cat([b.examples_idxscope for b in bboxes_ls])
      for b in range(1,batch_size):
        examples_idxscope[b,:] += examples_idxscope[b-1,1]
    cat_boxes = BoxList3D(bbox3d_cat, size3d, mode, examples_idxscope, constants=constants_all)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes_ls], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes

def cat_scales_anchor(anchors):
    '''
     combine anchors of scales
     anchors: list(BoxList)

     anchors_new: BoxList
     final flatten order:  [batch_size, scale_num, sparse_location_num, yaws_num]
    '''
    scale_num = len(anchors)
    batch_size = anchors[0].batch_size()
    anchors_scales = []
    for s in range(scale_num):
      anchors_scales.append( anchors[s].seperate_examples() )

    #num_examples = [[len(an) for an in ans] for ans in anchors_scales] # [batch_size, scale_num]

    examples = []
    for b in range(batch_size):
      examples.append( cat_boxlist_3d([ans[b] for ans in anchors_scales], per_example=False ) )
    anchors_all_scales = cat_boxlist_3d(examples, per_example=True)
    return anchors_all_scales

def extract_order_ids(sorted0, aim_order):
  '''
  sorted0: [n](>=0) the items are sorted alrady, but some items maybe repeated
  aim_order: 0,1,2,...
  The purpose is to find the items that appeared in aim_order time
  '''
  if sorted0.numel() == 0:
      ids = torch.empty(0, dtype=torch.int64)
      return ids
  if sorted0.numel() == 1:
    sorted0 = sorted0.view([-1])
  assert sorted0.dim() == 1
  assert sorted0.min() >= 0
  assert aim_order < 4
  previous = torch.cat([sorted0[0:1]*0-100, sorted0[:-1] ],0)
  if aim_order == 0:
    mask = sorted0 != previous
  else:
    pre_pre = torch.cat([sorted0[0:2]*0-100, sorted0[:-2] ],0)
    if aim_order == 1:
      mask0 = sorted0 == previous
      mask1 = sorted0 != pre_pre
      mask = mask0 * mask1
    else:
      pre_pre_pre = torch.cat([sorted0[0:3]*0-100, sorted0[:-3] ],0)
      if aim_order == 2:
        mask0 = sorted0 == previous
        mask1 = sorted0 == pre_pre
        mask2 = sorted0 != pre_pre_pre
        mask = mask0 * mask1 * mask2
      else:
        pre_4 = torch.cat([sorted0[0:4]*0-100, sorted0[:-4] ],0)
        if aim_order == 3:
          mask0 = sorted0 == previous
          mask1 = sorted0 == pre_pre
          mask2 = sorted0 == pre_pre_pre
          mask3 = sorted0 != pre_4
          mask = mask0 * mask1 * mask2 * mask3
  ids = torch.nonzero(mask).view([-1])
  return ids

class BoxList3D(object):
    """
    This class represents a set of 3d bounding boxes.
    The bounding boxes are represented as a Nx7 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox3d, size3d, mode, examples_idxscope, constants):
        '''
        All examples in same batch are concatenated together.
        examples_idxscope: [batch_size,2] record the index scope per example
        bbox3d: [N,7] N=sum(bbox num of each example)
        size3d: None or [N,6] -> [6] [is xyz_min, xyz_max]
        mode: "standard", "yx_zb"
        extra_fields: "label" "objectness"

        examples_idxscope=None: for batch_size=1
        constants={}: for non required
        '''
        assert mode == 'yx_zb' or mode == 'standard'
        assert bbox3d.shape[1] == 7, bbox3d.shape
        if examples_idxscope is None:
          examples_idxscope = torch.tensor([[0, bbox3d.shape[0]]], dtype=torch.int32)
        assert examples_idxscope[-1,-1] == bbox3d.shape[0]
        if size3d is not None:
          assert size3d.shape[1] == 6
          assert  size3d.shape[0] == examples_idxscope.shape[0]

        device = bbox3d.device if isinstance(bbox3d, torch.Tensor) else torch.device("cpu")
        bbox3d = torch.as_tensor(bbox3d, dtype=torch.float32, device=device)
        if bbox3d.ndimension() != 2:
            raise ValueError(
                "bbox3d should have 2 dimensions, got {}".format(bbox3d.ndimension())
            )
        if bbox3d.size(-1) != 7:
            raise ValueError(
                "last dimenion of bbox3d should have a "
                "size of 7, got {}".format(bbox3d.size(-1))
            )
        self.check_mode(mode)

        # constants: scale_num, num_anchors_per_location,
        # type='prediction'/'ground_truth'/'anchor'
        self.constants = constants

        #assert mode == 'yx_zb', "Both anchor, gt_boxes, prediction in the network is yx_zb"
        bbox3d[:,-1] =  OBJ_DEF.limit_yaw( bbox3d[:,-1], yx_zb = mode=='yx_zb') # [-pi/2, pi/2]
        if not self.is_prediction():
          OBJ_DEF.check_bboxes(bbox3d, yx_zb = mode=='yx_zb')
        else:
          pass
          #print('prediction')


        self.bbox3d = bbox3d
        self.size3d = size3d
        self.mode = mode
        self.examples_idxscope = examples_idxscope
        self.extra_fields = {}
    def centroids(self):
        centroids = self.bbox3d[:,0:3].clone()
        if self.mode == 'yx_zb':
            centroids[:,2] += self.bbox3d[:,5]*0.5
        return centroids
    def check_bboxes(self):
      OBJ_DEF.check_bboxes(self.bbox3d, self.mode=='yx_zb')

    def set_as_prediction(self):
      self.constants['prediction'] = True
    def is_prediction(self):
      return 'prediction' in self.constants and self.constants['prediction']
    def check_mode(self, mode):
        if mode not in ("standard", "yx_zb"):
            raise ValueError("mode should be 'standard' or 'yx_zb'")

    def batch_size(self):
        return self.examples_idxscope.shape[0]
    def seperate_examples(self):
      batch_size = self.batch_size()
      examples = []
      for bi in range(batch_size):
        examples.append(self.example(bi))
      return examples

    def add_field(self, field, field_data):
        if not isinstance(field_data, torch.Tensor):
          field_data = torch.Tensor(field_data)
        if len(field_data.shape) == 0:
          field_data = field_data.view(1)
        assert field_data.shape[0] == self.bbox3d.shape[0]
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def get_2corners_boxes(self):
      if self.mode != 'yx_zb':
        boxes = self.convert('yx_zb').bbox3d
      else:
        boxes = self.bbox3d
      boxes_2corners = Box3D_Torch.from_yxzb_to_2corners(boxes)
      return boxes_2corners

    def get_2top_corners_offseted(self):
      # offset the corners from end to corners by half thickness
      boxes_2corners = self.get_2corners_boxes()
      corners_top0 = boxes_2corners[:,[0,1,5]]
      corners_top1 = boxes_2corners[:,[2,3,5]]
      if self.bbox3d.shape[0] == 0:
        return corners_top0, boxes_2corners
      n = corners_top0.shape[0]
      corners_top = torch.cat([corners_top0.view(n,1,3),  corners_top1.view(n,1,3) ], 1 )
      # offset the corners_top
      centroids = corners_top.mean(dim=1, keepdim=True)
      #centroids = (corners_top0 + corners_top1) / 2.0
      offset = (centroids - corners_top)
      offset_norm = offset.norm(dim=2).view([n,2,1])
      if self.mode == 'yx_zb':
        thickness = self.bbox3d[:,3].view(n,1,1)
      elif self.mode == 'standard':
          thickness = self.bbox3d[:,4].view(n,1,1)
      offset = offset / offset_norm * thickness * 0.5
      corners_top = corners_top + offset
      zbottoms = boxes_2corners[:,4:5]
      return corners_top, boxes_2corners

    def convert(self, mode):
        # ref: utils3d/bbox3d_ops.py/Bbox3D
        self.check_mode(mode)
        if mode == self.mode:
            return self
        bbox3d0 = self.bbox3d
        bbox3d1 = bbox3d0[:,[0,1,2,4,3,5,6]]
        if mode == 'standard':
          bbox3d1[:,2] += bbox3d0[:,5] * 0.5
          bbox3d1[:,-1] += math.pi*0.5
        else:
          bbox3d1[:,2] -= bbox3d0[:,5] * 0.5
          bbox3d1[:,-1] -= math.pi*0.5
        #print(f'0 max: {bbox3d0[:,-1].min()}')
        #print(f'0 min: {bbox3d0[:,-1].max()}\n')
        #print(f'1 max: {bbox3d1[:,-1].min()}')
        #print(f'1 min: {bbox3d1[:,-1].max()}')
        bbox = BoxList3D(bbox3d1, self.size3d, mode, self.examples_idxscope, self.constants)
        bbox._copy_extra_fields(self)
        return bbox


    # Tensor-like methods
    def to(self, device):
        if self.size3d is None:
            size3d = None
        else:
            size3d = self.size3d.to(device)
        bbox3d = BoxList3D(self.bbox3d.to(device), size3d, self.mode, self.examples_idxscope, self.constants)

        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox3d.add_field(k, v)
        return bbox3d

    def detach(self):
      self.bbox3d = self.bbox3d.detach()
      for k in self.extra_fields:
        self.extra_fields[k] = self.extra_fields[k].detach()
    def example(self, idx):
        assert idx < self.batch_size()
        se = self.examples_idxscope[idx]
        examples_idxscope = torch.tensor([[0, se[1]-se[0]]], dtype=torch.int32)
        if self.size3d is None:
          size3d_i = None
        else:
          size3d_i = self.size3d[idx:idx+1]
        bbox3d = BoxList3D( self.bbox3d[se[0]:se[1],:], size3d_i, self.mode, examples_idxscope, self.constants)
        for k, v in self.extra_fields.items():
            bbox3d.add_field(k, v[se[0]:se[1]])
        return bbox3d

    def seperate_items_to_examples(self, items):
      example_idx = self.get_example_idx(items)
      bs = self.batch_size()
      items_examples = []
      for bi in range(bs):
        mask = example_idx == bi
        items_bi = items[mask] - self.examples_idxscope[bi,0]
        items_examples.append( items_bi )
      return items_examples

    def get_example_idx(self,items):
      examples_idxscope = self.examples_idxscope.long()
      example_idx = items*0
      batch_size = self.batch_size()
      for bi in range(batch_size):
        for j in range(items.shape[0]):
          if items[j].cpu() >= examples_idxscope[bi,0].cpu() and items[j].cpu() < examples_idxscope[bi,1].cpu():
            example_idx[j] = bi
      return example_idx


    def __getitem__(self, items):
      '''
      items: [n] torch.Tensor or list or numpy
          like: 2, [52,35,231], np.array([52,4,46]), torch.Tensor([101,23,45])

      mask not supported

      No matter if items contain all the examples or not, always keep the batch_size same.
      '''
      if not isinstance(items, torch.Tensor):
        items = torch.tensor(items, dtype=torch.int64)
      assert len(items.shape) <= 1
      items = items.view(-1)

      example_idxs = self.get_example_idx(items)
      batch_size = self.batch_size()
      examples_idxscope = torch.zeros((batch_size,2), dtype=torch.int64)
      for bi in range(batch_size):
        num_bi = torch.sum(example_idxs == bi)
        examples_idxscope[bi,1] += num_bi
        if bi != batch_size-1:
          examples_idxscope[bi+1:] += num_bi

      boxlist = BoxList3D(self.bbox3d[items], self.size3d, self.mode, examples_idxscope, self.constants)
      for k, v in self.extra_fields.items():
          boxlist.add_field(k, v[items])
      return boxlist


    def __len__(self):
        return self.bbox3d.shape[0]

    def clip_to_pcl(self, remove_empty=True):
        return
        raise NotImplementedError
        TO_REMOVE = 1
        self.bbox3d[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox3d[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox3d[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox3d[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        area_xy = self.bbox3d[:,3] * self.bbox3d[:,4]
        return area_xy

        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields):
        size3d = self.size3d.clone() if self.size3d is not None else self.size3d
        bbox3d_list = BoxList3D(self.bbox3d.clone(), size3d, self.mode, self.examples_idxscope.clone(), self.constants)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            bbox3d_list.add_field(field, self.get_field(field))
        return bbox3d_list

    def copy(self):
      return self.copy_with_fields(self.fields())

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "mode={})".format(self.mode)
        return s


    def clamp_size(self):
      self.bbox3d[:,3:6] = torch.clamp(self.bbox3d[:,3:6], min=0.001)


    def get_connect_corner_ids(self, threshold=0.1):
      '''
      n objects
      connect_ids: [2n,3]
        2n corners, for each corner, 3 connected corners are recorded. If the connected number is less than 3, asign -1.
        [0,1] are the corners of object 0
      '''
      corners0, _ = self.get_2top_corners_offseted() # [n,2,3]
      corners = corners0.view(-1,3) # [2n,3]
      m = corners.shape[0]
      dif = corners.view(-1,1,3) - corners.view(1,-1,3)
      device = corners.device
      tmp = torch.eye(m, dtype=torch.float32, device=device)
      dis = dif.norm(dim=2) + tmp
      mask = dis < threshold
      ids = torch.nonzero(mask)  # [m,2] the connected corner relationship, each corner may be connected to muliple corners

      # each corner can maximum connected to 3 others
      connect_ids = torch.zeros(m, 3, device=device, dtype=torch.int64)-1

      first = extract_order_ids(ids[:,0], 0)
      second = extract_order_ids(ids[:,0], 1)
      third = extract_order_ids(ids[:,0], 2)
      for j,id_ids in enumerate([first, second, third]):
        if id_ids.shape[0] > 0:
          tmp = ids[id_ids]
          connect_ids[tmp[:,0], j] =  tmp[:,1]

      # check
      check = 0
      if check:
        for i in range(m):
          ids_i = connect_ids[i]
          mask = ids_i>=0
          ids_i = ids_i[mask].view([-1])
          if ids_i.shape[0]>0:
            tmp = ids_i[0:1]*0 + i
            ids_i = torch.cat([tmp, ids_i])
            corners_i = corners[ids_i]
            print(corners_i)
            self.show(points=corners_i)
      return connect_ids

    def show(self, max_num=-1, points=None, with_centroids=False, boxes_show_together=None, points_keep_rate=POINTS_KEEP_RATE, points_sample_rate=POINTS_SAMPLE_RATE, colors=None):
      import numpy as np
      from utils3d.bbox3d_ops import Bbox3D
      boxes = self.bbox3d.cpu().data.numpy()
      if with_centroids:
        centroids = boxes.copy()
        if self.mode == 'yx_zb':
            centroids[:,2] += centroids[:,5]*0.5
        centroids[:,3:6] = 0.05
      if max_num >= 0 and max_num < boxes.shape[0]:
        step = 4
        ids0 = np.random.choice(boxes.shape[0]//step-1, max_num, replace=False).reshape(-1,1)*step
        tmp = np.arange(step).reshape(1,step)
        ids = (ids0 + tmp).reshape(-1)
        print(ids)
        boxes = boxes[ids]
      if with_centroids:
        boxes = np.concatenate([boxes, centroids], 0)
      if boxes_show_together:
        boxes_show_together = boxes_show_together.bbox3d.cpu().data.numpy()
        labels = np.array( [0]*boxes.shape[0] + [1]*boxes_show_together.shape[0])
        boxes = np.concatenate([boxes, boxes_show_together], 0)
      else:
        if 'labels' in self.fields():
            labels = self.get_field('labels').cpu().data.numpy().astype(np.int32)
        else:
            labels = None
      random_color = colors is None
      if points is None:
        Bbox3D.draw_bboxes(boxes, 'Z', is_yx_zb=self.mode=='yx_zb', \
        labels = labels, random_color=random_color)
      else:
        points = points.cpu().data.numpy()
        Bbox3D.draw_points_bboxes(points, boxes, 'Z', is_yx_zb=self.mode=='yx_zb',\
          labels = labels,  random_color=random_color, points_keep_rate=points_keep_rate, points_sample_rate=points_sample_rate, box_colors=colors)

    def show_centroids(self, max_num=-1, points=None):
      import numpy as np
      from utils3d.bbox3d_ops import Bbox3D
      boxes = self.bbox3d.cpu().data.numpy()
      if max_num > 0 and max_num < boxes.shape[0]:
        ids = np.random.choice(boxes.shape[0], max_num, replace=False)
        boxes = boxes[ids]
      if points is None:
        Bbox3D.draw_centroids(boxes, 'Z', is_yx_zb=self.mode=='yx_zb')
      else:
        Bbox3D.draw_points_centroids(points, boxes, 'Z', is_yx_zb=self.mode=='yx_zb')

    def show_with_corners(self, only_corner=0):
      import numpy as np
      from utils3d.bbox3d_ops import Bbox3D
      corners,_ = self.get_2top_corners_offseted()
      corners = corners.view([-1,3])
      points = corners.cpu().data.numpy()
      boxes = self.bbox3d.cpu().data.numpy()
      if only_corner:
        boxes = boxes[0:0,:]
      Bbox3D.draw_points_bboxes(points, boxes, 'Z', is_yx_zb=self.mode=='yx_zb',\
          random_color=True)
      pass

    def show_pcl_corners(self, pcl):
      import numpy as np
      from utils3d.bbox3d_ops import Bbox3D
      corners,_ = self.get_2top_corners_offseted()
      corners = corners.view([-1,3]).cpu().data.numpy()
      corners_bot = corners.copy()

      tmp = self.bbox3d[:,5].cpu().data.numpy()
      tmp = np.tile(np.expand_dims(tmp,1), [1,2]).reshape(-1)
      corners_bot[:,2] = corners_bot[:,2] - tmp
      corners = np.concatenate([corners, corners_bot], 0)

      n = corners.shape[0]
      boxes = np.zeros([n,7])
      boxes[:,0:3] = corners
      boxes[:,3:6] = 0.2

      #points = pcl.cpu().data.numpy()
      points = pcl
      Bbox3D.draw_points_bboxes(points, boxes, 'Z', is_yx_zb=self.mode=='standard', random_color=True)
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

    def show__together(self, boxlist_1, max_num=-1, max_num_1=-1, points=None, offset_x=None, twolabels=False,
                       mesh=False, points_keep_rate=POINTS_KEEP_RATE, points_sample_rate=POINTS_SAMPLE_RATE, random_color=False, colors=None):
      import numpy as np
      from utils3d.bbox3d_ops import Bbox3D
      boxes = self.bbox3d.cpu().data.numpy().copy()
      if max_num > 0 and max_num < boxes.shape[0]:
        ids = np.random.choice(boxes.shape[0], max_num, replace=False)
        boxes = boxes[ids]

      boxes_1 = boxlist_1.bbox3d.cpu().data.numpy().copy()
      if max_num_1 > 0 and max_num_1 < boxes_1.shape[0]:
        ids = np.random.choice(boxes_1.shape[0], max_num_1, replace=False)
        boxes_1 = boxes_1[ids]

      if offset_x is not None:
          boxes_1[:,0] += offset_x

      if not twolabels and 'labels' in self.fields():
          labels = self.get_field('labels').cpu().data.numpy().astype(np.int32)
          labels_1 = boxlist_1.get_field('labels').cpu().data.numpy().astype(np.int32)
          labels = np.concatenate([labels, labels_1], 0)
      else:
          labels = np.array([0]*boxes.shape[0] + [1]*boxes_1.shape[0])

      boxes = np.concatenate([boxes, boxes_1], 0)
      if colors is not None:
        colors = np.concatenate(colors, 0)
        assert colors.shape[0] == boxes.shape[0]

      if points is None:
        if mesh:
          Bbox3D.draw_bboxes_mesh(boxes, 'Z', is_yx_zb=self.mode=='yx_zb', labels=labels)
        else:
          Bbox3D.draw_bboxes(boxes, 'Z', is_yx_zb=self.mode=='yx_zb', labels=labels, random_color=False)
      else:
        if isinstance(points, torch.Tensor):
          points = points.cpu().data.numpy()
        if offset_x is not None:
              tp = points.copy()
              tp[:,0] += offset_x
              points = np.concatenate([points, tp], 0)

              #
              tp = tp.copy()
              tp[:,0] += offset_x
              points = np.concatenate([points, tp], 0)


        if mesh:
          Bbox3D.draw_points_bboxes_mesh(points, boxes, 'Z', is_yx_zb=self.mode=='yx_zb', labels=labels, points_keep_rate=points_keep_rate,   points_sample_rate=points_sample_rate, random_color=random_color, box_colors=colors)
        else:
          Bbox3D.draw_points_bboxes(points, boxes, 'Z', is_yx_zb=self.mode=='yx_zb', labels=labels, random_color=random_color,
                                    points_keep_rate=points_keep_rate,  points_sample_rate=points_sample_rate, box_colors=colors)

    def show_highlight(self, ids, points=None):
        from utils3d.bbox3d_ops import Bbox3D
        ids = np.array(ids)
        n = len(self)
        labels = np.zeros([n]).astype(np.int)
        labels[ids] = 1
        boxes = self.bbox3d.cpu().data.numpy()
        if points is None:
            Bbox3D.draw_bboxes(boxes, 'Z', is_yx_zb=self.mode=='yx_zb', labels=labels, random_color=False)
        else:
            if isinstance(points, torch.Tensor):
                points = points.cpu().data.numpy()
            Bbox3D.draw_points_bboxes(points, boxes, 'Z', is_yx_zb=self.mode=='yx_zb', labels=labels, random_color=False)

    def show_by_pos_anchor(self, sampled_pos_inds, sampled_neg_inds, targets=None):
      import numpy as np
      sampled_pos_inds = sampled_pos_inds.cpu().data.numpy()
      sampled_neg_inds = sampled_neg_inds.cpu().data.numpy()

      objectness = self.get_field('objectness').cpu().data.numpy()

      #  objectness of pos anchors
      posa_objectness = objectness[sampled_pos_inds]
      min_posa_objectness = posa_objectness.min() if posa_objectness.shape[0]>0 else 1
      print(f"\n objectness of positive anchors:\n {posa_objectness} \nmin is {min_posa_objectness}")
      posa_preds = self[sampled_pos_inds]
      posa_preds.show(boxes_show_together=targets)

      # show the min objectness of pos anchors
      if len(sampled_pos_inds)>0:
        print(f'\nmin objectness of pos anchors: {min_posa_objectness}')
        min_top_mask = objectness == min_posa_objectness
        min_top_ids = np.where(min_top_mask)[0]
        min_top_preds = self[min_top_ids]
        min_top_preds.show(boxes_show_together=targets)

      #************************
      # the max remaining objectness: non positive
      not_pos_inds = [i for i in range(len(self)) if i not in sampled_pos_inds ]
      objectness_notpos = objectness[not_pos_inds]
      max_notpos_obj = objectness_notpos.max()
      print(f"\nmax objectness of not-pos: {max_notpos_obj}")
      max_bot_mask = objectness == max_notpos_obj
      max_bot_ids = np.where(max_bot_mask)[0]
      max_bot_preds = self[max_bot_ids]
      max_bot_preds.show(boxes_show_together=targets)

      gap = min_posa_objectness - max_notpos_obj
      print(f'\nobjectness quality by pos anchors: {gap}\n')

    def show_by_objectness(self, threshold, targets=None,
          rpn_box_regression=None, anchors=None, regression_targets=None, below=False, points=None, points_keep_rate=POINTS_KEEP_RATE, points_sample_rate=POINTS_SAMPLE_RATE ):
      import numpy as np
      from maskrcnn_benchmark.layers.smooth_l1_loss import get_yaw_loss

      obj_sco = 'objectness' if 'objectness' in self.extra_fields else 'scores'
      objectness = self.get_field(obj_sco).cpu().data.numpy()

      # the top objectness
      if below:
        mask = objectness <= threshold
      else:
        mask = objectness > threshold
      ids = np.where(mask)[0]
      top_objectness = objectness[ids]
      min_top_objectness = top_objectness.min() if top_objectness.shape[0]>0 else 1
      num_top = ids.shape[0]
      if targets:
        targets = targets.copy()
        targets.bbox3d[:,2] += 0.01
        print(f'\n\nnum_gt={len(targets)}\nnum_top={num_top}\n\n')
      print(f"\n objectness over {threshold}: \t{num_top} \n {top_objectness} \nmin is {min_top_objectness}")
      top_preds = self[ids]
      top_preds.show(boxes_show_together=targets, points=points, points_keep_rate=points_keep_rate, points_sample_rate=points_sample_rate)

      # show the min objectness of tops
      if len(ids)>0:
        print(f'\nmin objectness of top: {min_top_objectness}')
        min_top_mask = top_objectness == min_top_objectness
        min_top_ids = np.where(min_top_mask)[0]
        min_top_preds = top_preds[min_top_ids]
        min_top_preds.show(boxes_show_together=targets)

      if anchors:
        rpn_box_regression_top = rpn_box_regression[ids]
        anchors_top = anchors[ids]
        regression_targets_top = regression_targets[ids]
        yaw_loss_sindif = get_yaw_loss('SinDiff',  rpn_box_regression_top, regression_targets_top, anchors_top)
        yaw_loss_dif = get_yaw_loss('Diff',  rpn_box_regression_top, regression_targets_top, anchors_top)
        print(f'yaw_loss_sindif: max={yaw_loss_sindif.max()}, mean={yaw_loss_sindif.mean()}')
        print(f'yaw_loss_dif: max={yaw_loss_dif.max()}, mean={yaw_loss_dif.mean()}')
        pass

      #************************
      # the max remaining objectness of (top objectness)
      mask_bottom = objectness <= threshold
      objectness_bottom = objectness[mask_bottom]
      if len(objectness_bottom)==0:
        max_bottom_objectness = 0
      else:
        max_bottom_objectness = objectness_bottom.max()
      print(f"\nmax objectness of bottom: {max_bottom_objectness}")
      max_bot_mask = objectness == max_bottom_objectness
      max_bot_ids = np.where(max_bot_mask)[0]
      max_bot_preds = self[max_bot_ids]
      max_bot_preds.show(boxes_show_together=targets)

      gap = min_top_objectness - max_bottom_objectness
      print(f'\nobjectness quality by objectness: {gap}\n')

    def remove_low(self,field, threshold):
      values = self.get_field(field).cpu().data.numpy()
      mask = values > threshold
      ids = np.where(mask)[0]
      tops = self[ids]
      return tops

      # the top objectness
      if below:
        mask = objectness <= threshold
      else:
        mask = objectness > threshold

    def select_by_labels(self,  labels_select, field_name):
      labels = self.get_field(field_name)
      mask = labels == labels_select[0]
      for i in range(1, len(labels_select)):
        mask_i = labels == labels_select[i]
        mask = mask + mask_i
      ids = np.where(mask)[0]
      return self[ids]

    def select_by_over_z(self,  z_min):
      mask = self.bbox3d[:,2] > z_min
      ids = np.where(mask)[0]
      return self[ids]

    def select_by_below_z(self,  z_min):
      mask = self.bbox3d[:,2] < z_min
      ids = np.where(mask)[0]
      return self[ids]


    def same_loc_anchors(self,items):
      '''
      items: [n]
      '''
      npa = self.constants['num_anchors_per_location']
      assert npa is not None
      items_same_loc = []
      for item in items:
        start = int(item//npa) * npa
        tmp = torch.arange(start, start+npa, dtype=torch.int64)
        items_same_loc.append(tmp.view(1,npa))
      items_same_loc = torch.cat( items_same_loc, dim=0)
      return items_same_loc

    def show_anchors_per_loc(self):
      import numpy as np
      num_anchors_per_location = self.constants['num_anchors_per_location']
      assert num_anchors_per_location is not None
      num_anchors = len(self)
      ids = np.random.randint(0, num_anchors, 5)
      for i in ids:
        j = int(i//num_anchors_per_location) * num_anchors_per_location
        anchors_i = self[range(j,j+4)]
        anchors_i.show()


    def metric_4areas(self, low_threshold, high_threshold):
      labels = self.get_field('labels')
      labels = labels.cpu().data.numpy()
      gt_objectness = labels > 0
      objectness = self.get_field('objectness')
      objectness = objectness.cpu().data.numpy()
      pos = objectness > high_threshold
      neg = objectness <= low_threshold

      T = gt_objectness == pos
      F = 1 - T

      TP = T * pos
      TN = T * neg
      FP = F * pos
      FN = F * neg

      TPi = np.where(TP)[0]
      TNi = np.where(TN)[0]
      FPi = np.where(FP)[0]
      FNi = np.where(FN)[0]

      metric_masks = {'T': T, 'F':F,'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN}
      metric_inds = {'TP':TPi, 'TN':TNi, 'FP':FPi, 'FN':FNi}
      metric_evals = {'TP':TP.sum(), 'TN':TN.sum(), 'FP':FP.sum(), 'FN':FN.sum()}
      return  metric_inds, metric_evals

    def show_by_labels(self, labels_show):
      '''
      labels_show: [1,2]
      '''
      labels = self.get_field('labels')
      ids = []
      for l in labels_show:
        ids.append( torch.nonzero(labels == l).view([-1]) )
      ids = torch.cat(ids, 0)
      if len(ids)==0:
        return
      res = self[ids]
      res.show()


def merge_by_corners(boxlist, threshold=0.1):
  wall_ids = torch.nonzero( boxlist.get_field('labels') == 1).squeeze()
  walls = boxlist[wall_ids]
  walls = merge_walls_by_corners(walls)
  boxlist.bbox3d[wall_ids] = walls.bbox3d
  return boxlist

def merge_walls_by_corners(boxlist, threshold=0.1):
      is_standard = boxlist.mode == 'standard'
      if is_standard:
        boxlist = boxlist.convert('yx_zb')
      #boxlist.show_with_corners()
      top_2corners0, boxes_2corners0 = boxlist.get_2top_corners_offseted()
      boxes_2corners = boxes_2corners0.clone()
      top_2corners = top_2corners0.clone().view([-1,3])
      n = top_2corners.shape[0]
      dis = top_2corners.view([-1,1,3]) - top_2corners.view([1,-1,3])
      dis = dis.norm(dim=2)
      mask = dis < threshold
      device = top_2corners.device
      mask = mask - torch.eye(n, dtype=torch.uint8, device=device)
      mask_merged = torch.zeros([n], dtype=torch.int32, device=device)
      for i in range(n):
        dif_i = top_2corners[i:i+1] - top_2corners
        dis_i = dif_i.norm(dim=1)
        mask_i = dis_i < threshold
        j = i + 2 * (i%2==0) - 1
        mask_i[j] = 0
        ids_i = torch.nonzero(mask_i).squeeze(1)

        # check if the close ids include one whole object
        ids_j = ids_i + 2*(ids_i%2==0).to(torch.int64) - 1
        any_same_obj = ids_i.view(-1,1) == ids_j.view(1,-1)
        if any_same_obj.sum() > 0:
          continue

        #print(ids_i)
        if ids_i.shape[0] > 1:
          ave_i = top_2corners[ids_i].mean(dim=0).view(1,3)
          top_2corners[ids_i] = ave_i
          mask_merged[ids_i] = 1
          #if DEBUG:
          #  print(f'ids: {ids_i}')
          #  print(f'org: {top_2corners[ids_i]}')
          #  print(f'ave: {ave_i}')
          pass

        if DEBUG and False:
          corners_close = top_2corners0.view([-1,3])[ids_i]
          boxlist.show(points = corners_close)

          cor_tmp = top_2corners.view(-1,2,3)
          tmp = (cor_tmp[:,0] - cor_tmp[:,1]).norm(dim=1)
          if tmp.min()==0:
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass

      ids_merged = torch.nonzero(mask_merged).squeeze(1)
      corners_merged = top_2corners[ids_merged]
      top_2corners = top_2corners.view([-1,2,3])


      # offset the corners to the end by half thickness
      centroids = top_2corners.mean(dim=1, keepdim=True)
      offset = top_2corners - centroids
      offset = offset / offset.norm(dim=2, keepdim=True) * boxes_2corners[:,-1].view(-1,1,1) * 0.5
      top_2corners = top_2corners + offset

      boxes_2corners[:,0:2] = top_2corners[:,0,0:2]
      boxes_2corners[:,2:4] = top_2corners[:,1,0:2]
      boxes_2corners[:,5] = top_2corners[:,:,2].mean(dim=1)
      boxes_2corners[:,4] = boxes_2corners[:,4].mean()


      boxlist.bbox3d = Box3D_Torch.from_2corners_to_yxzb(boxes_2corners)

      if is_standard:
        boxlist = boxlist.convert('standard')
      #boxlist.show(points=corners_merged)
      #boxlist.show_with_corners()
      return boxlist

if __name__ == "__main__":
    bbox = BoxList([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)


