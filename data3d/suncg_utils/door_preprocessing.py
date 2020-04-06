# xyz Arpil 2019


import numpy as np
from utils3d.bbox3d_ops import Bbox3D
from utils3d.geometric_util import limit_period, vertical_dis_1point_lines, angle_of_2lines, vertical_dis_points_lines, ave_angles, lines_intersection_2d
from render_tools import show_walls_offsetz, show_walls_1by1

DEBUG = False

def preprocess_doors(doors0, walls, door_thickness=0.18):
  '''
  doors0:[d,7]
  walls:[w,7]
  '''
  door_n = doors0.shape[0]
  if door_n == 0:
      return doors0
  wall_n = walls.shape[0]
  walls_1 = walls.copy()
  # (1) Find the coresponding wall of each door
  # (1.1) Get the four top lines of each door
  door_corners = Bbox3D.bboxes_corners(doors0, 'Z')[:,0:4,:]  # [d,4,3]
  door_lines0 = np.take( door_corners,  Bbox3D._lines_z0_vids, axis=1 ) # [d,4,2,3]
  door_lines1 = door_lines0.reshape(-1,2,3)
  # (1.2) Get the centroid lines of each wall
  wall_cen_lines = Bbox3D.bboxes_centroid_lines( walls_1,'X','Z' )

  # (1.3) Get the intersections_2d between lines and walls
  # A door matches a wall means: there are two top lines of the door, which are
  # intersected with the wall centroid line.
  dw_ints = lines_intersection_2d(door_lines1[:,:,0:2], wall_cen_lines[:,:,0:2], True, True)
  dw_ints = dw_ints.reshape(door_n,4,wall_n,2) # [d,4,w,2]
  tmp0 = (1-np.isnan(dw_ints)).sum(3) # [d,4,w]
  door_mask0 = tmp0.sum(1) == 4
  door_matched0 = door_mask0.sum(1).reshape(-1,1)


  # (1.4) Sometimes on door can match two walls. Remove the confused one by:
  # The right wall should not contain any corners
  dc_in_wall_mask = Bbox3D.points_in_bbox(door_corners.reshape(-1,3), walls).reshape(door_n, 4, wall_n)
  dc_in_wall_mask = np.any(dc_in_wall_mask, axis=1)
  bad_match = (door_matched0>1) * dc_in_wall_mask
  door_mask = door_mask0 * (1-bad_match)

  door_ids, wall_ids = np.where(door_mask)
  success_num = door_ids.shape[0]

  # (2) Pick the failed door ids
  wall_num_each_door = door_mask.sum(1)
  fail_match_door_ids = np.where(wall_num_each_door!=1)[0]

  walls_good = walls[wall_ids]
  yaws = limit_period(walls_good[:,-1], 0, np.pi/2)

  intersections_2d = []
  for i in range(success_num):
    door_idx = door_ids[i]
    cids = np.where(tmp0[door_idx,:,wall_ids[i]])[0]
    ints_i  = dw_ints[door_idx, cids, wall_ids[i]]
    intersections_2d.append(ints_i.reshape(1,2,2))
  if len(intersections_2d)  == 0:
      return np.empty([0,7])
  intersections_2d = np.concatenate(intersections_2d, 0)

  doors_centroids_2d = np.mean(intersections_2d, 1)
  doors_length = np.linalg.norm(intersections_2d[:,0] - intersections_2d[:,1], axis=1)
  doors_length -= door_thickness * np.sin(yaws*2)

  doors_new = doors0[door_ids].copy()
  doors_new[:,0:2] = doors_centroids_2d
  doors_new[:,3] = doors_length
  doors_new[:,4] = door_thickness
  doors_new[:,6] = walls_good[:,-1]


  if DEBUG and fail_match_door_ids.shape[0]>0:
    print(f"fail_match_door_ids:{fail_match_door_ids}")
    show_all([doors0[fail_match_door_ids], walls])
    #for dids in fail_d_ids:
    #  fail_w_ids = np.where(door_mask[dids])[0]
    #  show_all([doors0[dids].reshape(-1,7), walls[fail_w_ids].reshape(-1,7)])
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass


  if DEBUG:
    show_all([doors0, doors_new, walls])
  return doors_new



def show_all(boxes_ls):
  boxes = np.concatenate(boxes_ls, 0)
  Bbox3D.draw_points_bboxes(boxes[:,0:3], boxes, 'Z', is_yx_zb=False)
