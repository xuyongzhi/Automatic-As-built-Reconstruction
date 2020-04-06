# xyz Nov 2018
import numpy as np
import open3d
from collections import defaultdict
from utils3d.bbox3d_ops import Bbox3D



def bbox_map(gt_boxes, bbox_targets):
  map_indices = defaultdict(list)
  gtb_n = gt_boxes.shape[0]
  targ_n = bbox_targets.shape[0]
  for i in range(gtb_n):
    for j in range(targ_n):
      if np.max(np.abs(gt_boxes[i]-bbox_targets[j])) < 1e-3:
        map_indices[i].append(j)
  return map_indices

def points2pcd_open3d(points):
  assert points.shape[-1] == 3
  pcd = open3d.PointCloud()
  points = points.reshape([-1,3])
  pcd.points = open3d.Vector3dVector(points[:,0:3])
  if points.shape[1] == 6:
    pcd.normals = open3d.Vector3dVector(points[:,3:6])
  return pcd


def random_sample_points(points0, n1):
  assert points0.ndim == 2
  n0 = points0.shape[0]
  if n1 <= n0:
    ids = np.random.choice(n0, n1, replace=False)
  else:
    ids0 = np.random.choice(n0, n1-n0, replace=True)
    ids = np.concatenate([np.arange(n0), ids0], 0)
  points1 = points0[ids]
  return points1

class DataRender():
  @staticmethod
  def show_dict(dict0, pre=''):
    print('\n')
    for item in dict0:
      print(pre, end='')
      v_i = dict0[item]
      if isinstance(v_i, np.ndarray):
        print(f"{item} : {v_i.shape}")
      elif isinstance(v_i, dict):
        #print(f"{item} : {v_i.keys()}")
        show_dict(v_i, item+' / ')
      else:
        print(f"{item} : {v_i}")


  @staticmethod
  def pos_targets_torch(example0, bi, points_i, is_show=False):
    import torch
    example = {}
    for key in example0:
      if isinstance(example0[key], torch.Tensor):
        example[key] = example0[key].data.cpu().numpy()
      else:
        example[key] = example0[key]

    labels, bbox_targets, anchors, anchors_mask, reg_weights = example['labels'][bi],\
          example['reg_targets'][bi], example['anchors'][bi], example['anchors_mask'][bi],\
          example['reg_weights'][bi]
    anchors_mask = anchors_mask == 1

    if is_show:
      DataRender.show_all_valid_anchors(anchors, anchors_mask, points_i)
    boxes_decoded = DataRender.pos_targets(labels, bbox_targets,\
              anchors, is_show, points=points_i)
    return boxes_decoded

  @staticmethod
  def show_all_valid_anchors(anchors, anchors_mask, points):
    anchors = anchors[anchors_mask]
    Bbox3D.draw_points_bboxes(points, anchors, 'Z', is_yx_zb=True)

  @staticmethod
  def pos_targets(labels, bbox_targets, anchors, is_show=False,\
                  gt_boxes=None, points=None):
      anchors0 = anchors.copy()
      anchors0[:,2] -= 5
      if gt_boxes is None:
        gt_boxes = np.zeros((0,7), dtype=np.float32)
      else:
        gt_boxes =gt_boxes.copy()
        gt_boxes[:,2] -= 1
      if points is None:
        points = np.zeros((0,3), dtype=np.float32)

      pos_ids = np.where(labels==1)[0]
      neg_ids = np.where(labels==0)[0]

      bbox_targets_pos = bbox_targets[pos_ids]
      anchors_pos = anchors[pos_ids]

      pos_num = pos_ids.shape[0]
      gt_num = gt_boxes.shape[0]

      anchor_num = anchors.shape[0]
      if is_show:
        print(f"gt_box_num:{gt_num}    anchor_num: {anchor_num}")
        print(f"all positive anchors: {pos_num}")
        labels_show = np.array([0]*pos_num + [1]*gt_num)
        boxes_show_0 = np.concatenate([anchors_pos, gt_boxes],0)
        Bbox3D.draw_points_bboxes(points, boxes_show_0,
                    'Z', is_yx_zb=True, labels=labels_show)

      # decoded anchors: shoud be exactly the gt_boxes
      from second.core.box_np_ops import second_box_decode
      bboxes_decoded = second_box_decode(bbox_targets_pos, anchors_pos, smooth_dim=True)
      #Bbox3D.draw_bboxes(bboxes_decoded, 'Z', is_yx_zb=True)
      #Bbox3D.draw_bboxes(gt_boxes, 'Z', is_yx_zb=True)
      boxes_show = np.concatenate([bboxes_decoded, gt_boxes, anchors0], 0)
      labels_show = np.array([0]*pos_num + [1]*gt_num + [2]*anchors0.shape[0])
      if is_show:
        print('gt boxes from targets')
        Bbox3D.draw_points_bboxes(points, boxes_show, 'Z', is_yx_zb=True, labels=labels_show)
      return bboxes_decoded


  @staticmethod
  def voxels_debatch(example0):
    '''
    split points of each frame from a batch voxels
    '''
    from torchplus.ops.array_ops import gather_nd, scatter_nd
    import torch
    example = {}
    for key in example0:
      if isinstance(example0[key], torch.Tensor):
        example[key] = example0[key].data.cpu().numpy()
      else:
        example[key] = example0[key]
    voxels = example['voxels']
    coors = example['coordinates']
    num_points = example['num_points']
    labels = example['labels']

    batch_ids = coors[:,0:1]
    batch_size = labels.shape[0]

    points = []
    n0 = voxels.shape[0]
    start = 0

    for i in range(batch_size):
      end = start
      while end<n0 and batch_ids[end]== i:
        end += 1

      voxels_i = voxels[start:end]
      num_points_i = num_points[start:end]
      n_i = voxels_i.shape[0]
      points_i = []
      for j in range(n_i):
        points_i.append(voxels_i[j,0:num_points_i[j]])
      points_i = np.concatenate(points_i, 0)
      points_i = random_sample_points(points_i, 60000)
      points.append(points_i)

      start = end
    points = np.array(points)

    return points

  @staticmethod
  def show_preds(anchors0, a_mask0, batch_box_preds0, total_scores0, gt_boxes, points):
      from utils_3d.bbox3d_ops import Bbox3D
      box_preds = batch_box_preds0.data.cpu().numpy()
      anchors = anchors0.data.cpu().numpy()
      total_scores = total_scores0.data.cpu().numpy()
      a_mask = a_mask0.data.cpu().numpy() == 1
      anchors = anchors[a_mask]

      #print('all valid anchors')
      #Bbox3D.draw_points_bboxes(points, anchors, 'Z', True)
      #print('all pred boxes')
      #Bbox3D.draw_points_bboxes(points, box_preds, 'Z', True)

      score_hist = np.histogram(total_scores, bins=np.arange(0,1.1,0.1))[0]
      score_hist = score_hist / score_hist.sum() * 100
      print(f'score_hist: {score_hist}')

      DataRender.show_high_score_preds(0.7, 1, total_scores, box_preds, anchors, gt_boxes, points)

      #DataRender.show_high_score_preds(0, 0.6, total_scores, box_preds, anchors, gt_boxes, points)
      #DataRender.show_high_score_preds(0.6, 0.7, total_scores, box_preds, anchors, gt_boxes, points)
      #DataRender.show_high_score_preds(0.7, 0.8, total_scores, box_preds, anchors, gt_boxes, points)
      #DataRender.show_high_score_preds(0.8, 0.9, total_scores, box_preds, anchors, gt_boxes, points)
      #DataRender.show_high_score_preds(0.9, 1, total_scores, box_preds, anchors, gt_boxes, points)
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

  @staticmethod
  def show_high_score_preds(score_threshold0, score_threshold1, total_scores, box_preds, anchors, gt_boxes, points):
      pos_ids = np.where(np.logical_and(total_scores >= score_threshold0, total_scores < score_threshold1))[0]
      box_preds_pos = box_preds[pos_ids]
      anchors = anchors[pos_ids]
      pnum = box_preds_pos.shape[0]
      gnum = gt_boxes.shape[0]
      #gt_boxes[:,2] -= 0.2
      #points[:,2] -= 1
      labels_show = np.array([0]*pnum + [1]*gnum)

      boxes_show = np.concatenate([box_preds_pos, gt_boxes], 0)
      #Bbox3D.draw_bboxes(boxes_show, 'Z', is_yx_zb=True, labels=labels_show)
      print(f'pred boxes with scores in [{score_threshold0}, {score_threshold1}]')
      Bbox3D.draw_points_bboxes(points, boxes_show, 'Z', is_yx_zb=True, labels=labels_show)

      #print(f'anchors with scores in [{score_threshold0}, {score_threshold1}]')
      #Bbox3D.draw_points_bboxes(points, anchors, 'Z', is_yx_zb=True)

      pass

  @staticmethod
  def show_final_preds(pred_boxes, points, gt_boxes):
      pred_boxes = pred_boxes.data.cpu().numpy()
      boxes_show = np.concatenate([pred_boxes, gt_boxes], 0)
      pnum = pred_boxes.shape[0]
      gnum = gt_boxes.shape[0]
      labels_show = np.array([0]*pnum + [1]*gnum)
      Bbox3D.draw_points_bboxes(points, boxes_show, 'Z', is_yx_zb=True, labels=labels_show)

  @staticmethod
  def show_points_bbox(points, boxes, labels=None, names='', image_path=''):
    from utils_3d.bbox3d_ops import Bbox3D
    #print("\timage: ", image_path)
    Bbox3D.draw_points_bboxes(points, boxes, up_axis='Z', labels=labels, names=names, is_yx_zb=True)

  @staticmethod
  def show_posanchors_targets(anchors, targets_dict, points, gt_boxes):
    labels, bbox_targets, bbox_outside_weights = targets_dict['labels'], targets_dict['bbox_targets'], targets_dict['bbox_outside_weights']
    DataRender.pos_targets(labels, bbox_targets, anchors, True, gt_boxes, points)


  @staticmethod
  def show_anchors(points, anchors, gt_boxes, anchors_mask=None):
    # show valid anchors
    if anchors_mask is not None:
      indices = np.where(anchors_mask)[0]
      anchors = anchors[indices,:]

    show_num = -1
    if show_num>0 and show_num < anchors.shape[0]:
      indices = np.random.choice(anchors.shape[0], show_num)
      anchors = anchors[indices,:]

    n0 = anchors.shape[0]
    n1 = gt_boxes.shape[0]
    anchors = np.concatenate([anchors, gt_boxes], 0)
    labels = np.array([1]*n0 + [0]*n1)
    Bbox3D.draw_points_bboxes(points, anchors, 'Z', is_yx_zb=True, labels=labels)
    #Bbox3D.draw_bboxes(gt_boxes, 'Z', is_yx_zb=False)

  @staticmethod
  def show_points_in_box(points, bboxes, bbox_corners, surfaces, point_masks, is_yx_zb):
      if is_yx_zb:
        bboxes = Bbox3D.convert_from_yx_zb_boxes(bboxes)

      pn_in_box = np.sum(point_masks, 0)

      bn = bboxes.shape[0]
      points_in_boxes = []
      for i in range(bn):
        points_in_boxes.append( points[point_masks[:,i]] )

      pcd0 = points2pcd_open3d(points[:,:3])
      pcd0_aug = points2pcd_open3d(points[:,:3]-np.array([[0,0,4]]))

      corner_pcds = [points2pcd_open3d(corners) for corners in bbox_corners]
      surface_pcds = [points2pcd_open3d(surface) for surface in surfaces]
      points_inb_pcds = [points2pcd_open3d(points[:,:3]) for points in points_in_boxes]
      bboxes_ls0 = [Bbox3D.draw_bbox_open3d(box, 'Z') for box in bboxes]
      frame_mesh = open3d.create_mesh_coordinate_frame(size = 2.0, origin = bboxes[0,0:3])

      points_inb_pcds_valid = [pcd for i,pcd in enumerate(points_inb_pcds) if pn_in_box[i]>0]
      open3d.draw_geometries( bboxes_ls0 + points_inb_pcds_valid + [pcd0_aug])

      show_corners = False
      show_surfaces = False
      show_points_inb = False
      for i in range(bn):
        frame_mesh = open3d.create_mesh_coordinate_frame(size = 2.0, origin = bboxes[i,0:3])
        if show_corners:
          open3d.draw_geometries(bboxes_ls0[i:i+1]+corner_pcds[i:i+1]+[frame_mesh])
        if show_surfaces:
          open3d.draw_geometries(bboxes_ls0[i:i+1]+surface_pcds[i:i+1]+[frame_mesh])
        if show_points_inb:
          open3d.draw_geometries(bboxes_ls0[i:i+1]+points_inb_pcds[i:i+1]+[frame_mesh])
          pass

