# xyz  April 2019

import numpy as np
from utils3d.bbox3d_ops import Bbox3D
from utils3d.geometric_util import limit_period, vertical_dis_1point_lines, angle_of_2lines, vertical_dis_points_lines, ave_angles
from render_tools import show_walls_offsetz, show_walls_1by1

DEBUG = True

def preprocess_windows(windows0, walls):
  '''
  both windows0 ad walls are standard: [xc, yc, zc, x_size, y_size, z_size, yaw]
  '''
  if DEBUG and False:
      print('input')
      show_all([windows0,walls])
  #Bbox3D.draw_bboxes(walls, 'Z', False)
  #Bbox3D.draw_bboxes(windows0, 'Z', False)
  #print(f'windows0: \n{windows0}')
  if len(windows0) == 0:
    return windows0
  windows1 = Bbox3D.define_walls_direction(windows0, 'Z', yx_zb=False)
  #print(f'windows1: \n{windows0}')
  win_bad_ids, wall_ids_for_bad_win  = find_wall_ids_for_windows(windows1, walls)

  windows_bad = windows1[win_bad_ids].reshape(-1,7)
  walls_bw = walls[wall_ids_for_bad_win].reshape(-1,7)
  windows_corrected = correct_bad_windows(windows_bad, walls_bw)
  windows1[win_bad_ids] = windows_corrected

  if DEBUG and False:
    print('out')
    show_all([windows1,walls])
  return windows1

def find_wall_ids_for_windows(windows, walls):
  #if DEBUG:
  #  windows = windows[15:17,:]
  #  Bbox3D.draw_points_bboxes(windows[:,0:3], walls, 'Z', False)
  win_in_walls = Bbox3D.points_in_bbox(windows[:,0:3], walls)
  wall_nums_per_win = win_in_walls.sum(1)
  if wall_nums_per_win.max() > 1:
    #win_mw_ids, wall_ids_ = np.where(win_in_walls)
    win_mw_ids = np.where(wall_nums_per_win > 1)[0]

    windows_thickness_multi = windows[win_mw_ids, 4]
    windows_thickness_rate_multi = windows[win_mw_ids, 4] / windows[win_mw_ids, 3]
    thickness_small = windows_thickness_multi.max() < 0.25
    thickness_rate_small = windows_thickness_rate_multi.max() < 0.25
    if not(thickness_small or thickness_rate_small):
      show_high(windows, walls, win_mw_ids, [])
      print(f'windows_missed:\n{windows[win_mw_ids]}')
      print("There is some windows, find multiple responding walls and thickness is not small.")
      assert False
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass


  if wall_nums_per_win.min()==0:
    # a window centroid is inside of no wall
    missed_win_ids = np.where(wall_nums_per_win==0)[0]
    windows_thickness_missed = windows[missed_win_ids, 4]
    windows_thickness_rate_missed = windows[missed_win_ids, 4] / windows[missed_win_ids, 3]
    thickness_small = windows_thickness_missed.max() < 0.25
    thickness_rate_small = windows_thickness_rate_missed.max() < 0.25
    if not(thickness_small or thickness_rate_small):
      #show_high(windows, walls, missed_win_ids, [])
      print(f'windows_missed:\n{windows[missed_win_ids]}')
      print("There is some windows, cannot find responding wall and thickness is not small.")
      assert False
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

    #cen_lines_wall = Bbox3D.bboxes_centroid_lines(walls, 'X', 'Z')
    #cen_lines_wall[:,:,2] = 0
    #for i in missed_win_ids:
    #  point_i = windows[i,0:3]
    #  point_i[2] = 0
    #  dis_i = vertical_dis_1point_lines(point_i, cen_lines_wall)
    #  #show_high(windows, walls, [i], [])
    #  pass

  win_bad_ids0, wall_ids0 = np.where(win_in_walls)
  wall_yaws0 = walls[wall_ids0, -1]
  # when responding wall yaw is not vertical or horizonal, the window is bad
  yaw_bad0 = (np.abs(wall_yaws0) / (np.pi*0.5)) % 1 > 0.01
  win_bad_ids1 = np.where(yaw_bad0)[0]

  win_bad_ids = win_bad_ids0[win_bad_ids1]
  wall_ids_for_bad_win = wall_ids0[win_bad_ids1]

  if DEBUG and False:
    for i in range(win_bad_ids.shape[0]):
      print(f'bad window {i}/{win_bad_ids.shape[0]}')
      show_high(windows, walls, win_bad_ids[i], wall_ids_for_bad_win[i])
  return win_bad_ids, wall_ids_for_bad_win


def correct_bad_windows(windows_bad, walls):
  windows_cor = windows_bad.copy()
  windows_cor[:,-1] = walls[:,-1]
  windows_cor[:,4] = 0.175
  windows_cor[:,3] = np.sqrt(windows_bad[:,3]**2 + windows_bad[:,4]**2)
  yaws = limit_period(walls[:,-1], 0, np.pi/2)
  windows_cor[:,3] -= 0.175 * np.sin(yaws*2)
  if DEBUG and False:
    show_all([windows_bad, windows_cor, walls])
  return windows_cor

def show_all(boxes_ls):
  boxes = np.concatenate(boxes_ls, 0)
  if boxes.shape[0]==0:
    return
  Bbox3D.draw_points_bboxes(boxes[:,0:3], boxes, 'Z', is_yx_zb=False)

def show_high(windows, walls, win_high_ids, wall_high_ids):
    boxes = np.concatenate([windows, walls], 0)
    win_high_ids = np.array(win_high_ids).reshape([-1])
    wall_high_ids = np.array(wall_high_ids).reshape([-1]) + windows.shape[0]
    high_ids = np.concatenate([win_high_ids, wall_high_ids], 0).astype(np.int)
    Bbox3D.draw_bboxes(boxes, 'Z', False, highlight_ids=high_ids)
    pass
