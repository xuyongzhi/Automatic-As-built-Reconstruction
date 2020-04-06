# Feb 2019
import numpy as np
from utils3d.bbox3d_ops import Bbox3D
from utils3d.geometric_util import limit_period, vertical_dis_1point_lines, \
             vertical_dis_points_lines, ave_angles
from data3d.render_tools import show_walls_offsetz, show_walls_1by1
from second.core.non_max_suppression.nms_gpu import rotate_iou_gpu, rotate_iou_gpu_eval

MERGE_Z_ANYWAY_XYIOU_THRESHOLD = 0.75
DEBUG = True

def preprocess_walls(wall_bboxes):
  '''
    The framework to generate wall ground truth from SUNCG
    1) Make y-dim as thickness, x_size > y_size, yaw in (-pi/2, pi/2]
    2) If a final wall is originally splitted as pieces in SUNCG, merge them as one
    3) If a wall in SUNCG actually contains several finally walls, crop it
    4) Clean repeated walls
  '''
  show_pro = False
  if show_pro:
    print('original')
    show_walls_offsetz(wall_bboxes)
    #Bbox3D.draw_bboxes_mesh(wall_bboxes, 'Z', False)
    #Bbox3D.draw_bboxes(wall_bboxes, 'Z', False)
  if wall_bboxes.shape[0] == 0:
      return wall_bboxes

  wall_bboxes = Bbox3D.define_walls_direction(wall_bboxes, 'Z', yx_zb=False, check_thickness=True)


  wall_bboxes = merge_pieces_of_same_walls_alongY(wall_bboxes)
  if show_pro:
    print('merge_pieces_of_same_walls_alongY')
    show_walls_offsetz(wall_bboxes)

  wall_bboxes = merge_pieces_of_same_walls_alongX(wall_bboxes)
  if show_pro:
    print('merge_pieces_of_same_walls_alongX')
    show_walls_offsetz(wall_bboxes)


  wall_bboxes = crop_walls(wall_bboxes)
  if show_pro:
    print('crop_walls')
    show_walls_offsetz(wall_bboxes)

  wall_bboxes = merge_pieces_of_same_walls_alongY(wall_bboxes)
  if show_pro:
    print('merge_pieces_of_same_walls_alongY again after crop_walls to solve some special case: Originally, a wall is broken by no intersection like 0058113bdc8bee5f387bb5ad316d7b28')
    show_walls_offsetz(wall_bboxes)

  wall_bboxes = find_close_walls(wall_bboxes)
  if show_pro:
    print('clean_close_walls')
    show_walls_offsetz(wall_bboxes)


    #intersections = Bbox3D.all_intersections_by_cenline(wall_bboxes, not_on_corners=True, show_res=True)
    #mask =( wall_bboxes[:,0] < 45) *  (wall_bboxes[:,0] > 43)
    #wall_bboxes_ = wall_bboxes[mask]

  if DEBUG and 0:
    show_walls_offsetz(wall_bboxes)
  return wall_bboxes


def merge_2pieces_of_1wall(bbox0, bbox1, dim):
  '''
    [1,7] [1,7]
    two box can merge, when:
    z, size_y, size_z same
    yaw0 = yaw1 = angle of c0_c1
    out: [1,7]
  '''
  from utils3d.geometric_util import angle_with_x
  assert dim =='X' or 'Y'
  dim = 0 if dim=='X' else 1

  bbox0 = bbox0.reshape([1,7])
  bbox1 = bbox1.reshape([1,7])
  dif = bbox1 - bbox0
  dif[0,-1] = limit_period(dif[0,-1], 0.5, np.pi)

  z_same = np.abs(dif[0,2]) < 0.01
  if dim == 0:
    # o means the other here, if dim=='X', o='Y'
    so_same = np.abs(dif[0,3+1-dim]) < 0.05
  else:
    so_same = np.abs(dif[0,3+1-dim]) < 0.15
  sz_same = np.abs(dif[0,3+2]) < 0.01
  z_sames0 = z_same and sz_same
  if z_sames0:
      z_sames = z_sames0
  else:
      z0_max = bbox0[0,2] + bbox0[0,5] * 0.5
      z0_min = bbox0[0,2] - bbox0[0,5] * 0.5
      z1_max = bbox1[0,2] + bbox1[0,5] * 0.5
      z1_min = bbox1[0,2] - bbox1[0,5] * 0.5
      zmin_dif = np.abs(z1_min - z0_min)
      zmax_dif = np.abs(z1_max - z0_max)
      zmin_same = zmin_dif < 0.01
      zmax_same = zmax_dif < 0.03
      z_sames = zmin_same and zmax_same

      if not zmax_same:
        iou_xy = rotate_iou_gpu(bbox0[:,[0,1,3,4,6]], bbox1[:,[0,1,3,4,6]])
        #print(f'box0 : {bbox0}')
        #print(f'box1 : {bbox1}')
        print(f'zmin_dif:{zmin_dif}, zmax_dif:{zmax_dif}\n iou_xy:{iou_xy}\n')
        if iou_xy > MERGE_Z_ANYWAY_XYIOU_THRESHOLD:
            print('Merge even z is different')
            z_sames = True
        else:
            print('abort merging because of z is different')
        if DEBUG and False:
            box_show = np.concatenate([bbox0, bbox1], 0)
            Bbox3D.draw_bboxes(box_show, 'Z', False)
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass

      if z_sames:
        zmin_new = min(z0_min, z1_min)
        zmax_new = max(z0_max, z1_max)
        z_new = (zmin_new + zmax_new) / 2.0
        z_size_new = zmax_new - zmin_new

        bbox0[0,2] = z_new
        bbox0[0,5] = z_size_new
        bbox1[0,2] = z_new
        bbox1[0,5] = z_size_new

  yaw_same = np.abs(dif[0,-1]) < 0.05


  if not (z_sames and so_same and yaw_same):
    return None

  cen_direc = dif[:,0:2]
  if dim == 0:
    cen_line0 = Bbox3D.bboxes_centroid_lines(bbox0, 'X', 'Z')
    dis_01 = vertical_dis_1point_lines(bbox1[0,0:3], cen_line0)[0]
    overlap_mask0 = dis_01 < bbox0[0,4] * 0.51 + 0.01
    if not overlap_mask0:
      return None

    cen_dis = np.linalg.norm(dif[0,0:3])
    overlap_mask1 = cen_dis < (bbox0[0,3] + bbox1[0,3])*0.5+0.01
    if not overlap_mask1:
      return None


  centroid_lines1 = Bbox3D.bboxes_centroid_lines(bbox1, 'X' if dim == 1 else 'Y', 'Z')
  cen_dis = vertical_dis_1point_lines(bbox0[0,0:3], centroid_lines1)[0]

  size_dim0 = bbox0[0,3+dim]
  size_dim1 = bbox1[0,3+dim]

  is_box0_covered_by_1 = size_dim1*0.5 > cen_dis + size_dim0*0.5
  is_box1_covered_by_0 = size_dim0*0.5 > cen_dis + size_dim1*0.5


  if is_box0_covered_by_1:
    merged = bbox1
  elif is_box1_covered_by_0:
    merged = bbox0
  else:
    k = size_dim1 / (size_dim0 + size_dim1)
    new_centroid = bbox0[0,0:3] + (bbox1[0,0:3] - bbox0[0,0:3]) * k
    new_size_dim = (size_dim0 + size_dim1) / 2 + cen_dis

    merged = (bbox0 + bbox1) / 2
    merged[:,-1] = ave_angles(bbox0[:,-1], bbox1[:,-1], scope_id=1)
    if np.abs(bbox1[0,-1]-bbox0[0,-1]) > np.pi * 0.5:
      merged[0,-1] =  ( np.abs(bbox0[0,-1]) + np.abs(bbox1[0,-1]) )/2.0
    merged[0,0:3] = new_centroid
    merged[0,3+dim] = new_size_dim

  #
  show = False
  if show:
    box_show = np.concatenate([bbox0, bbox1, merged], 0)
    box_show[2,2] += 0.1
    Bbox3D.draw_bboxes(box_show, 'Z', False)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass
  return merged

def merge_pieces_of_same_walls_alongX(wall_bboxes):
  '''
    1) Find all the walls with not both corners intersected
    2) Merge all the walls can be merged
  '''
  if wall_bboxes.shape[0] == 0:
      return wall_bboxes
  intersections = Bbox3D.all_intersections_by_cenline(wall_bboxes, check_same_height=False, only_on_corners=True)
  num_inters = np.array([it.shape[0] for it in intersections])
  mask = num_inters < 2
  candidate_ids = np.where(mask)[0]


  show = DEBUG and False
  if show:
    show_boxes = wall_bboxes.copy()
    show_boxes[:,2] -= 1
    show_boxes = np.concatenate([show_boxes, wall_bboxes[candidate_ids]], 0)
    print(f'candidate_ids:{candidate_ids}')
    Bbox3D.draw_bboxes(show_boxes, 'Z', False)
    #Bbox3D.draw_bboxes(wall_bboxes[[5,6,12]], 'Z', False)
    pass

  n = candidate_ids.shape[0]

  keep_mask = np.array([True] * wall_bboxes.shape[0])
  for i in range(n-1):
    idx_i = candidate_ids[i]
    for j in range(i+1, n):
        idx_next = candidate_ids[j]

        merged_i = merge_2pieces_of_1wall(wall_bboxes[idx_i],
                                          wall_bboxes[idx_next], 'X')
        if merged_i is not None:

          keep_mask[idx_i] = False
          wall_bboxes[idx_next] = merged_i[0]

          if show:
            show_boxes = wall_bboxes.copy()
            show_boxes[:,2] -= 1
            show_boxes = np.concatenate([show_boxes, wall_bboxes[[idx_i, idx_next]], merged_i],0)
            show_boxes[-1,2] += 1
            Bbox3D.draw_bboxes(show_boxes, 'Z', False)
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass


  wall_bboxes = wall_bboxes[keep_mask]


  rm_num = np.sum(1-keep_mask)
  print(f'merge along X: rm {rm_num} walls')

  if show:
    show_walls_offsetz(wall_bboxes)
  return wall_bboxes


def merge_pieces_of_same_walls_alongY(wall_bboxes):
  '''
    1) Find all the walls which are A) parallel along X, B) one of cen_corners is very close
      C) centroid of one wall is inside of another
    2) Merge along Y
  '''
  from utils3d.geometric_util import angle_with_x, vertical_dis_points_lines
  if wall_bboxes.shape[0] == 0:
      return wall_bboxes
  show = False
  if show:
    wall_bboxes0 = wall_bboxes.copy()
    wall_bboxes0[:,2] -= 1

  n = wall_bboxes.shape[0]
  cen_lines = Bbox3D.bboxes_centroid_lines(wall_bboxes, cen_axis='X', up_axis='Z') # [n,2,3]
  cen_direc = cen_lines[:,1,:] - cen_lines[:,0,:] # [n,3]
  np.linalg.norm(cen_direc[:,0:2], axis=1)
  angles = angle_with_x(cen_direc[:,0:2], 1)
  boxes_merged_extra = []
  remain_mask = np.array([True]*n)
  merge_y_num = 0
  split_and_merge_num = 0

  for i in range(n-1):
    # 1) parallel along X
    angles_dif = angles[i] - angles[i+1:]
    # make to [0,np.pi/2]
    angles_dif = np.abs( limit_period(angles_dif, 0.5, np.pi) )
    angle_mask = angles_dif < 7 * np.pi/180

    # 2) one of centroid line corners is very close
    cen_dis = cen_lines[i:i+1].reshape([1,2,1,3]) - cen_lines[i+1:].reshape([-1,1,2,3])
    cen_dis = np.linalg.norm(cen_dis, axis=3)
    cen_dis_min = cen_dis.min(axis=1).min(axis=1)
    cen_dis_max = cen_dis.max(axis=1).max(axis=1)
    thickness_sum = wall_bboxes[:,4]*0.5 + wall_bboxes[i,4]*0.5
    thickness_sum = thickness_sum[i+1:]
    cendis_mask = cen_dis_min < thickness_sum

    # 3) the other centroid line corner is inside of the long box
    centroid_dis = np.linalg.norm(wall_bboxes[i:i+1,0:3] - wall_bboxes[i+1:,0:3], axis=1)
    #tmp = np.maximum(wall_bboxes[i,3], wall_bboxes[i+1:,3])
    tmp = (wall_bboxes[i,3] + wall_bboxes[i+1:,3])*0.45 - 0.1
    centroid_mask = centroid_dis < tmp

    #if i==4:
    #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
    #  show_boxes = wall_bboxes[[4,12]]
    #  Bbox3D.draw_bboxes(show_boxes, 'Z', False)
    #  pass

    mask_i = angle_mask * cendis_mask * centroid_mask
    if mask_i.any():
      ids = np.where(mask_i)[0] + i + 1
      # 5) the centroid corner used to split the other

      # 4) vertical dis
      vertical_dis = vertical_dis_points_lines(cen_lines[i], cen_lines[ids]).mean(0)
      ave_thickness = (wall_bboxes[i,4] + wall_bboxes[ids, 4]) * 0.5
      thickness_gap_rate = vertical_dis / ave_thickness
      verdis_mask = (thickness_gap_rate > 0.2) * (thickness_gap_rate < 1.2)

      ids = ids[verdis_mask]
      vertical_dis = vertical_dis[verdis_mask]

      is_need_merge = ids.shape[0] > 0


      if is_need_merge:
        # split the longer box, and merge the shorter one along Y
        k = ids.shape[0]
        for j in range(k):
          idx = ids[j]

          size_x_i = wall_bboxes[i,3]
          size_x_j = wall_bboxes[idx,3]
          size_x_rate = size_x_i / size_x_j
          print(f'size_x_rate: {size_x_rate}')
          if np.abs(size_x_rate-1) < 0.15:
              merge_y_num += 1
              box_merge = merge_2pieces_of_1wall(wall_bboxes[i], wall_bboxes[idx], 'Y')

              if show and False:
                show_boxes = np.concatenate([wall_bboxes[i:i+1], wall_bboxes[idx:idx+1]], 0)
                if box_merge is not None:
                  show_boxes = np.concatenate([show_boxes, box_merge], 0)
                  show_boxes[-1,2] += 0.2
                else:
                  import pdb; pdb.set_trace()  # XXX BREAKPOINT
                  pass
                show_boxes = np.concatenate([show_boxes, wall_bboxes0], 0)
                Bbox3D.draw_bboxes(show_boxes, 'Z', False)
                pass
              if box_merge is not None:
                wall_bboxes[idx] = box_merge.reshape([7])
                remain_mask[i] = False
              else:
                print('The two walls cannot be merged along Y, this should not happen normally')
                print('merge_pieces_of_same_walls_alongY again after crop_walls to solve some special case: Originally, a wall is broken by no intersection like 0058113bdc8bee5f387bb5ad316d7b28')
                #show_boxes = np.concatenate([wall_bboxes[i:i+1], wall_bboxes[idx:idx+1]], 0)
                #Bbox3D.draw_bboxes(wall_bboxes, 'Z', False, highlight_ids=[idx, i])
                #Bbox3D.draw_bboxes(show_boxes, 'Z', False)
                #import pdb; pdb.set_trace()  # XXX BREAKPOINT
                #pass
                #raise NotImplementedError

          else:
              # the longer box need to be split along X before merging
              split_and_merge_num += 1
              cen_dis_ij = cen_dis[idx-i-1]
              if size_x_rate>1:
                longer_idx = i
                short_idx = idx
                cen_dis_ij = cen_dis_ij.min(axis=0)
              else:
                longer_idx = idx
                short_idx = i
                cen_dis_ij = cen_dis_ij.min(axis=1)
              # find the intersection point on longer box
              far_corner_id = int(cen_dis_ij[0] < cen_dis_ij[1])
              cen_dir_longer = cen_lines[longer_idx, 1] - cen_lines[longer_idx, 0]
              cen_dir_longer /= np.linalg.norm(cen_dir_longer)
              sl_dir = cen_lines[short_idx, far_corner_id] - cen_lines[longer_idx, 0]
              intersection = np.sum(sl_dir * cen_dir_longer) * cen_dir_longer + cen_lines[longer_idx,0]
              # offset: half of the thickness
              intersection += cen_dir_longer * wall_bboxes[i,4] * 0.5

              splited_boxes = Bbox3D.split_wall_by_centroid_intersections(wall_bboxes[longer_idx], intersection.reshape([1,3])) # [2,7]

              if splited_boxes.shape[0] == 1:
                  print('\n\n\t\tComplicated situation, not solved well yet.\n\n')
                  box_merge = None
                  if False and DEBUG and splited_boxes.shape[0] == 1:
                      box_tmp = np.array([[0,0,0, 0.5,0.5,0.5, 0]])
                      box_tmp[0,0:3] = intersection.reshape([1,3])
                      boxes_show = np.concatenate([box_tmp, wall_bboxes[longer_idx].reshape([-1,7])], 0)
                      Bbox3D.draw_points_bboxes(intersection.reshape([1,3]), boxes_show, 'Z', False)
              else:
                  tmp = wall_bboxes[short_idx,0:3] - splited_boxes[:,0:3] # [2,3]
                  tmp = np.linalg.norm(tmp, axis=1) # [2]
                  merge_id = int(tmp[0] > tmp[1])
                  box_merge = merge_2pieces_of_1wall(wall_bboxes[short_idx], splited_boxes[merge_id], 'Y')

              if show and False:
              #if box_merge is None:
                show_boxes = np.concatenate([wall_bboxes[i:i+1], wall_bboxes[idx:idx+1]], 0)
                Bbox3D.draw_points_bboxes(intersection.reshape([1,3]), show_boxes, 'Z', False)
                show_boxes = np.concatenate([show_boxes, splited_boxes], 0)
                show_boxes[-1,2] += 0.5
                show_boxes[-2,2] += 0.7
                if box_merge is not None:
                  show_boxes = np.concatenate([show_boxes, box_merge], 0)
                  show_boxes[-1,2] += 1
                show_boxes = np.concatenate([show_boxes, wall_bboxes0], 0)
                Bbox3D.draw_bboxes(show_boxes, 'Z', False)
                import pdb; pdb.set_trace()  # XXX BREAKPOINT
                pass

              if box_merge is None:
                # temperally solution, may because of a wall being changed several
                # times
                remain_mask[short_idx] = False
              else:
                wall_bboxes[longer_idx] = splited_boxes[1-merge_id]
                wall_bboxes[short_idx] = box_merge.reshape([7])
                boxes_merged_extra.append(box_merge.reshape([1,7]))

  wall_bboxes_new = wall_bboxes[remain_mask]
  #boxes_merged_extra = np.concatenate(boxes_merged_extra, 0)
  #wall_bboxes_new = np.concatenate([wall_bboxes, boxes_merged_extra], 0)
  print(f'merge_y_num:{merge_y_num}  split_and_merge_num:{split_and_merge_num}')
  if show:
    show_walls_offsetz(wall_bboxes_new)
    #show_walls_offsetz(wall_bboxes)
    #show_walls_offsetz(wall_bboxes0[remain_mask])
    show_walls_offsetz(wall_bboxes0[np.logical_not(remain_mask)])
    #show_walls_offsetz(boxes_merged_extra)
  return wall_bboxes_new

def merge_2close_walls(bbox0, bbox1):
    '''
      in: [n,7] [n,7]
      out: [n,7]

      centroid: mean
      x_size: max
      y_size: max
      z_size: max
      yaw: mean
    '''
    bboxes = np.concatenate([bbox0.reshape([-1,1,7]), bbox1.reshape([-1,1,7])], 1)
    bboxes_new = bboxes.mean(axis=1)
    bboxes_new[:,-1] = ave_angles(bboxes[:,0,-1], bboxes[:,1,-1], scope_id=1)
    bboxes_new[:,3:6] = bboxes[:,:,3:6].max(axis=1)

    show = False
    if show:
      show_boxes = np.concatenate([bboxes[0], bboxes_new], 0)
      show_boxes[-1,2] += 0.2
      Bbox3D.draw_bboxes(show_boxes, 'Z', False)
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

    return bboxes_new

def is_close_2walls(wall0, wall1):
  '''
    check if 2 walls are close whem iou < 0.85
  '''
  assert wall0.shape == wall1.shape == (7,)
  dif = (wall0 - wall1)
  cen_dis = np.linalg.norm(dif[0:3])
  size_dif = np.max(np.abs(dif[3:6]))
  yaw_dif = np.abs( Bbox3D.limit_period(dif[-1], 0.5, np.pi))

  close = (cen_dis < 0.05) and (size_dif < 0.07) and (yaw_dif < 0.05)
  #print(f'cen_dis:{cen_dis}, size_dif:{size_dif}, yaw_dif:{yaw_dif}\nclose:{close}')
  return close

def find_close_walls(walls):
  '''
    walls: [n,7]
  '''
  show = False
  if show:
    walls0 = walls.copy()

  corners = Bbox3D.bboxes_corners(walls, 'Z') # [n,8,3]
  cen_lines_x = Bbox3D.bboxes_centroid_lines(walls, 'X', 'Z') # [n,2,3]
  cen_lines_y = Bbox3D.bboxes_centroid_lines(walls, 'Y', 'Z') # [n,2,3]
  n = walls.shape[0]

  dis_x0 = vertical_dis_points_lines(corners.reshape([-1,3])[:,0:2], cen_lines_x[:,:,0:2]).reshape([n,8,n])
  dis_x_ave = dis_x0.mean(1) # [n,n]
  dis_x_max = dis_x0.max(1) # [n,n]
  inside_x_ave = dis_x_ave / (walls[:,4].reshape([1,n]) * 0.501 + 0.01)
  inside_x_max = dis_x_max / (walls[:,4].reshape([1,n]) * 0.8 + 0.03)
  inside_x_mask = (inside_x_ave < 1) * (inside_x_max < 1)

  dis_y0 = vertical_dis_points_lines(corners.reshape([-1,3])[:,0:2], cen_lines_y[:,:,0:2]).reshape([n,8,n])
  dis_y_ave = dis_y0.max(1) # [n,n]
  dis_y_max = dis_y0.max(1) # [n,n]
  inside_y_ave = dis_y_ave / (walls[:,3].reshape([1,n]) * 0.515 + 0.03)
  inside_y_max = dis_y_max / (walls[:,3].reshape([1,n]) * 0.55 + 0.05)
  inside_y_mask = (inside_y_ave < 1) * (inside_y_max < 1)

  inside_mask = inside_x_mask * inside_y_mask
  # inside_mask[i,j] is True means walls[i] is inside of walls[j]

  #print(inside_x)
  #print(inside_y)
  #print(inside_mask)

  remain_mask = np.array([True]*n)
  for i in range(n-1):
    for j in range(i+1, n):
      if inside_mask[i, j] or inside_mask[j,i]:
        if inside_mask[i, j] and inside_mask[j,i]:
          # (A) If inside with each other, merge two close walls
          walls[j] = merge_2close_walls(walls[i], walls[j])
          remain_mask[i] = False
        elif inside_mask[i, j]:
          # (B) If one is inside another, remove the inside one
          remain_mask[i] = False
        elif inside_mask[j, i]:
          remain_mask[j] = False
  walls_new = walls[remain_mask]
  rm_n = np.sum(remain_mask)
  print(f'clean close walls {n} -> {rm_n}')

  if show:
    show_boxes = walls0
    show_boxes[:,2]  -= 0.2
    show_boxes = np.concatenate([show_boxes, walls_new], 0)
    Bbox3D.draw_bboxes(show_boxes, 'Z', False)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    #show_walls_offsetz(walls_new)
    #show_walls_1by1(walls_new)

  return walls_new

def is_close_walls(wall0, walls1):
  '''
  wall0: [1,7]
  wallss:[n,7]
  '''
  pass



def clean_close_walls(wall_bboxes):
  from second.core.non_max_suppression.nms_gpu import rotate_iou_gpu, rotate_iou_gpu_eval
  #Bbox3D.draw_bboxes(wall_bboxes, 'Z', False)

  walls_2d = wall_bboxes[:,[0,1,3,4,6]]

  show_merge = False
  wall_bboxes_0 = wall_bboxes.copy()
  wall_bboxes_0[:,2] -= 1

  n = wall_bboxes.shape[0]
  keep_mask = np.array([True]*n)
  for i in range(n):
    for j in range(i+1, n):
      #iou_ij = ious[i,j]
      #if iou_ij <= 0.05:
      #  continue
      #if ious[i,j] >= 0.8:
      #  close_ij = True
      #else:
      close_ij = is_close_2walls(wall_bboxes[i], wall_bboxes[j])
      if not close_ij:
        continue

      if close_ij:
        keep_mask[i] = False
        merged_i = merge_2close_walls(wall_bboxes[i], wall_bboxes[j])
        show_merge = False
      else:
        merged_i = np.zeros(shape=[0,7], dtype=np.float32)
        show_merge = False


      if show_merge:
        #print(f'iou: {ious[i,j]}\nfA:{wall_bboxes[i]}\nB:{wall_bboxes[j]}\nM:{merged_i}')
        merged_show = merged_i.copy()
        merged_show[:,2] += 1
        box_show = np.concatenate([wall_bboxes[i:i+1], wall_bboxes[j:j+1],
                          #merged_show],0)
                          merged_show, wall_bboxes_0],0)
        box_show[0,2] += 0.2
        Bbox3D.draw_bboxes(box_show, 'Z', False)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass

      if not close_ij:
        print( "2 walls are a bit close, but not close enough to merge" + \
                "This should not happen.")
        #raise NotImplementedError
      else:
        wall_bboxes[j] = merged_i

  wall_bboxes_new = wall_bboxes[keep_mask]
  merge_num = n - wall_bboxes_new.shape[0]

  print(f'wall num: {n} -> {wall_bboxes_new.shape[0]}')

  #walls_2d = wall_bboxes_new[:,[0,1,3,4,6]]
  #ious = rotate_iou_gpu(walls_2d, walls_2d)

  #show_walls_offsetz(wall_bboxes_new)
  #show_walls_1by1(wall_bboxes_new)
  return wall_bboxes_new

def crop_walls(wall_bboxes):
  '''
    crop walls with intersections not on the corner
  '''
  #show_walls_1by1(wall_bboxes)
  if wall_bboxes.shape[0]==0:
      return wall_bboxes

  intersections = Bbox3D.all_intersections_by_cenline(wall_bboxes, check_same_height=False, not_on_corners=True)
  n = wall_bboxes.shape[0]
  new_walls = []
  keep_mask = np.array([True] * n)
  for i in range(n):
    inters_i = intersections[i]
    if inters_i.shape[0] == 0:
      continue
    new_walls_i = Bbox3D.split_wall_by_centroid_intersections(wall_bboxes[i], inters_i)
    keep_mask[i] = False
    new_walls.append(new_walls_i)

    show = False
    #tmp = np.concatenate(new_walls, 0)
    #if tmp.shape[0] >= 7:
    #  show = True
    if show:
      #tmp = wall_bboxes.copy()
      tmp = new_walls_i.copy()
      tmp[:,2] += 1
      for ti in range(1, tmp.shape[0]):
        tmp[ti:,2] += 0.2
      show_box = np.concatenate([tmp, wall_bboxes[i:i+1]], 0)
      Bbox3D.draw_points_bboxes(inters_i, show_box, 'Z', False)
      #Bbox3D.draw_bboxes(show_box, 'Z', False)
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

  num_croped = np.sum(keep_mask==False)
  print(f'num croped:{num_croped}\nnum new:{len(new_walls)}')
  wall_bboxes = wall_bboxes[keep_mask]
  if len(new_walls) > 0:
    new_walls = np.concatenate(new_walls, 0)
    wall_bboxes = np.concatenate([wall_bboxes, new_walls], 0)
  #show_walls_1by1(wall_bboxes)
  #show_walls_offsetz(wall_bboxes)
  return wall_bboxes


def test_merge_X():
  wall_bboxes = np.array([[ 44.52736626, -40.29249993,   1.36749997,   1.17500176,
          0.09473495,   2.73499994,  -1.57079633],
       [ 44.57263031, -40.33499827,   1.36749997,   1.190002  ,
          0.09473877,   2.73499994,  -1.57079633]])
  find_close_walls(wall_bboxes)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT

if __name__ == '__main__'  :
  test_merge_X()
  pass
