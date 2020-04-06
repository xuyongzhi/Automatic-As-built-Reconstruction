from utils3d.bbox3d_ops import Bbox3D
import numpy as np
from utils3d.geometric_util import vertical_dis_points_lines


Debug = 1


def preprocess_cfr(ceilings_org, walls_org, obj):
  '''
  Z is up, Y is thickness
  '''
  #Bbox3D.draw_bboxes(walls, 'Z', False)
  ceilings = ceilings_org.copy()
  walls = walls_org.copy()
  walls = replace_slant_walls(walls)

  dis_threshold = 0.07

  ceiling_cens = ceilings[:,0:3]
  ceiling_cens[:,2] = 0
  ceil_cenlines_x = Bbox3D.bboxes_centroid_lines(ceilings, 'X', 'Z')
  ceil_cenlines_x[:,:,2] = 0
  #ceil_cenlines_y = Bbox3D.bboxes_centroid_lines(ceilings, 'Y', 'Z')
  wall_cenlines = Bbox3D.bboxes_centroid_lines(walls, 'X', 'Z')
  wall_cenlines[:,:,2] = 0


  ceilings_shrink = ceilings.copy()
  ceilings_shrink[:,3:5] -= 0.3

  cn = ceilings.shape[0]

  ## Find edge wall nums
  good_ceiling_ids = []
  for c in range(cn):
    # (0.1) If no any other overlap ceiling, try to keep it
    # Otherwise, delete it when  >3 wall inside ceiling
    tmp = np.delete( ceiling_cens.copy(), c, axis=0 )
    any_overlap = Bbox3D.points_in_bbox(tmp, ceilings[c:c+1]).any()
    if any_overlap:
      wall_corner0_in_ceil = Bbox3D.points_in_bbox(wall_cenlines[:,0,:], ceilings_shrink[c:c+1])
      wall_corner1_in_ceil =  Bbox3D.points_in_bbox(wall_cenlines[:,1,:], ceilings_shrink[c:c+1])
      wall_inside_ceil = wall_corner0_in_ceil + wall_corner1_in_ceil
      wall_inside_ceil_ids = np.where(wall_inside_ceil)[0]
      nwic = wall_inside_ceil_ids.shape[0]

      if nwic > 3:
        if Debug and 1:
          print(f'bad ceiling, contain {nwic} walls inside')
          box_show = np.concatenate([walls_org, ceilings_org[c:c+1]], 0)
          Bbox3D.draw_bboxes_mesh(box_show, 'Z', False)

          box_show = np.concatenate([walls_org[wall_inside_ceil_ids], ceilings_org[c:c+1]], 0)
          Bbox3D.draw_bboxes_mesh(box_show, 'Z', False)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        continue

    # (1) the central corners of wall are inside of ceiling
    wall_cenlines_auged = line_aug(wall_cenlines)
    cw_cen_dis = ceiling_cens[c].reshape([1,1,-1]) - wall_cenlines_auged
    cw_cen_dis = np.linalg.norm(cw_cen_dis, axis=2)
    ceil_diag_size  = np.linalg.norm( ceilings[c,3:5] )
    on_inside_ceil = (cw_cen_dis - ceil_diag_size/2 < dis_threshold).sum(1) >= 2

    if Debug and 0:
      #Bbox3D.draw_points_bboxes(wall_cenlines_auged.reshape([-1,3]), walls, 'Z', False)
      inside_ids = np.where(on_inside_ceil)[0]
      box_show = np.concatenate([walls_org[inside_ids], ceilings_org[c:c+1]], 0)
      Bbox3D.draw_bboxes_mesh(box_show, 'Z', False)

    # (2) along x: wall central line is on x boundaries of ceiling
    dis_cw = vertical_dis_points_lines(ceil_cenlines_x[c], wall_cenlines)
    ceil_y_thickness = ceilings[c,4]
    mask_x0 = np.abs(dis_cw[0] - dis_cw[1]) < dis_threshold
    mask_x1 = (np.abs(dis_cw - ceil_y_thickness/2) < dis_threshold).all(0)
    mask_x = mask_x0 * mask_x1 * on_inside_ceil
    wall_on_ceil_boundary_parall_x = np.where( mask_x )[0]
    num_edgew_x = clean_edge_wall_same_side(wall_cenlines[wall_on_ceil_boundary_parall_x])

    # (3) along x: wall central line is on x boundaries of ceiling
    ceil_x_thickness = ceilings[c,3]
    mask_y0 = dis_cw  < dis_threshold
    mask_y1 = np.abs(dis_cw - ceil_x_thickness) < dis_threshold
    mask_y = (mask_y0 + mask_y1).all(0)
    mask_y = mask_y * on_inside_ceil
    wall_on_ceil_boundary_parall_y = np.where(mask_y )[0]
    num_edgew_y = clean_edge_wall_same_side(wall_cenlines[wall_on_ceil_boundary_parall_y])

    #Bbox3D.point_in_box(wall_cenlines, ceilings[])

    edge_wall_num = num_edgew_x + num_edgew_y

    if edge_wall_num >= 3:
      good_ceiling_ids.append(c)


    if Debug and edge_wall_num < 3 and 0:
      print(f'edge_wall_num: {edge_wall_num}')
      box_show = np.concatenate([walls_org, ceilings_org[c:c+1]], 0)
      Bbox3D.draw_bboxes_mesh(box_show, 'Z', False)
      #Bbox3D.draw_points_bboxes(ceil_cenlines_x[c], box_show, 'Z', False)
      #Bbox3D.draw_points_bboxes(ceil_cenlines_x[c], ceilings[c:c+1], 'Z', False)

      edge_walls_x = walls_org[wall_on_ceil_boundary_parall_x]
      box_x = np.concatenate([edge_walls_x, ceilings_org[c:c+1]], 0)
      #Bbox3D.draw_bboxes_mesh(box_x, 'Z', False)

      edge_walls_y = walls_org[wall_on_ceil_boundary_parall_y]
      box_y = np.concatenate([edge_walls_y, ceilings_org[c:c+1]], 0)
      #Bbox3D.draw_bboxes_mesh(box_y, 'Z', False)

      walls_inside = walls_org[wall_inside_ceil_ids]
      box_ins = np.concatenate([walls_inside, ceilings_org[c:c+1]], 0)
      #Bbox3D.draw_bboxes_mesh(box_ins, 'Z', False)

      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

  good_ceiling_ids = np.array(good_ceiling_ids).reshape([-1])
  new_cn = good_ceiling_ids.shape[0]
  print(f'\n\n{obj} {cn} -> {new_cn}')
  if new_cn == 0:
    new_ceilings = ceilings_org[0:0]
  else:
    new_ceilings = ceilings_org[good_ceiling_ids]
  if Debug and new_cn < cn:
      print(good_ceiling_ids)
      box_show = np.concatenate([walls_org, new_ceilings], 0)
      Bbox3D.draw_bboxes_mesh(box_show, 'Z', False)

      bad_ceil_ids = np.array([i for i in range(cn) if i not in good_ceiling_ids   ]).astype(np.int32)
      if bad_ceil_ids.shape[0]>0:
        box_show = np.concatenate([walls_org, ceilings_org[bad_ceil_ids]], 0)
        Bbox3D.draw_bboxes_mesh(box_show, 'Z', False)
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
  return ceilings

def line_aug(wall_cenlines):
  cen = wall_cenlines.mean(axis=1).reshape(-1,1,3)
  aug_points = wall_cenlines * 0.6 + cen * 0.4
  wall_cenlines_auged = np.concatenate([wall_cenlines, aug_points], 1)
  return wall_cenlines_auged

def clean_edge_wall_same_side(cenlines):
  '''
  [n,2,3]
  '''
  n = cenlines.shape[0]
  if n<=1:
    return n
  rm_ids = []
  for i in range(n-1):
    centroid_i = cenlines[i].mean(0).reshape([1,3])
    dis = vertical_dis_points_lines(centroid_i, cenlines[i+1:])
    if dis.min() < 0.05:
      rm_ids.append(i)
  return  n - len(rm_ids)

def clean_repeat(ceilings):
  #Bbox3D.draw_ceilings(ceilings, 'Z', False)
  n = ceilings.shape[0]
  mdifs = []
  for i in range(1,n):
    difs = []
    for j in range(i):
      difs.append( np.sum(np.abs( ceilings[i] - ceilings[j] )) )
    mdifs.append( min(difs) )
  mdifs = np.array(mdifs)
  keep_mask = mdifs > 0.1
  ceilings = ceilings[keep_mask]
  return ceilings


