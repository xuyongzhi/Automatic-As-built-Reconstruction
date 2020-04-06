# May 2018 xyz
import numpy as np
import numba

def Rx( x ):
    # ref to my master notes 2015
    # anticlockwise, x: radian
    Rx = np.zeros((3,3))
    Rx[0,0] = 1
    Rx[1,1] = np.cos(x)
    Rx[1,2] = np.sin(x)
    Rx[2,1] = -np.sin(x)
    Rx[2,2] = np.cos(x)
    return Rx

def Ry( y ):
    # anticlockwise, y: radian
    Ry = np.zeros((3,3))
    Ry[0,0] = np.cos(y)
    Ry[0,2] = -np.sin(y)
    Ry[1,1] = 1
    Ry[2,0] = np.sin(y)
    Ry[2,2] = np.cos(y)
    return Ry

@numba.jit(nopython=True)
def Rz( z ):
    # anticlockwise, z: radian
    Rz = np.zeros((3,3))

    Rz[0,0] = np.cos(z)
    Rz[0,1] = np.sin(z)
    Rz[1,0] = -np.sin(z)
    Rz[1,1] = np.cos(z)
    Rz[2,2] = 1
    return Rz

def R1D( angle, axis ):
    if axis == 'x':
        return Rx(angle)
    elif axis == 'y':
        return Ry(angle)
    elif axis == 'z':
        return Rz(angle)
    else:
        raise NotImplementedError

def EulerRotate( angles, order ='zxy' ):
    R = np.eye(3)
    for i in range(3):
        R_i = R1D(angles[i], order[i])
        R = np.matmul( R_i, R )
    return R

def point_rotation_randomly( points, rxyz_max=np.pi*np.array([0.1,0.1,0.1]) ):
    # Input:
    #   points: (B, N, 3)
    #   rx/y/z: in radians
    # Output:
    #   points: (B, N, 3)
    batch_size = points.shape[0]
    for b in range(batch_size):
        rxyz = [ np.random.uniform(-r_max, r_max) for r_max in rxyz_max ]
        R = EulerRotate( rxyz, 'xyz' )
        points[b,:,:] = np.matmul( points[b,:,:], np.transpose(R) )
    return points

def angle_with_x(direc, scope_id=0):
  if direc.ndim == 2:
    x = np.array([[1.0,0]])
  if direc.ndim == 3:
    x = np.array([[1.0,0,0]])
  x = np.tile(x, [direc.shape[0],1])
  return angle_of_2lines(direc, x, scope_id)

def angle_of_2lines(line0, line1, scope_id=0):
  '''
    line0: [n,2/3]
    line1: [n,2/3]
    zero as ref

   scope_id=0: [0,pi]
            1: (-pi/2, pi/2]

   angle: [n]
  '''
  assert line0.ndim == line1.ndim == 2
  assert (line0.shape[0] == line1.shape[0]) or line0.shape[0]==1 or line1.shape[0]==1
  assert line0.shape[1] == line1.shape[1] # 2 or 3

  norm0 = np.linalg.norm(line0, axis=1, keepdims=True)
  norm1 = np.linalg.norm(line1, axis=1, keepdims=True)
  #assert norm0.min() > 1e-4 and norm1.min() > 1e-4 # return nan
  line0 = line0 / norm0
  line1 = line1 / norm1
  angle = np.arccos( np.sum(line0 * line1, axis=1) )

  if scope_id == 0:
    pass
  elif scope_id == 1:
    # (-pi/2, pi/2]: offset=0.5, period=pi
    angle = limit_period(angle, 0.5, np.pi)
  else:
    raise NotImplementedError
  return angle

def limit_period(val, offset, period):
  '''
    [0, pi]: offset=0, period=pi
    [-pi/2, pi/2]: offset=0.5, period=pi
    [-pi, 0]: offset=1, period=pi

    [0, pi/2]: offset=0, period=pi/2
    [-pi/4, pi/4]: offset=0.5, period=pi/2
    [-pi/2, 0]: offset=1, period=pi/2
  '''
  return val - np.floor(val / period + offset) * period

def ave_angles(angles0, angles1, scope_id=0):
  '''
    angles0: any shape
    angles1: same as angles0
    scope_id = 0: [-pi/2, pi/2]
    scope_id = 1: [0, pi]
    scope_id defines the scope of both input angles and averaged.
    period = np.pi

    make the angle between the average and both are below half period

    out: [n]
  '''
  assert angles0.shape == angles1.shape

  period = np.pi
  dif = angles1 - angles0
  mask = np.abs(dif) > period * 0.5
  angles1 += - period * mask * np.sign(dif)
  ave = (angles0 + angles1) / 2.0
  if scope_id==0:
    ave = limit_period(ave, 0.5, period)
  elif scope_id==1:
    ave = limit_period(ave, 0, period)
  else:
    raise NotImplementedError
  return ave

def vertical_dis_points_lines(points, lines):
  '''
  points:[n,3]
  lines:[m,2,3]
  dis: [n,m]
  '''
  dis = []
  pn = points.shape[0]
  for i in range(pn):
    dis.append( vertical_dis_1point_lines(points[i], lines).reshape([1,-1]) )
  dis = np.concatenate(dis, 0)
  return dis

def vertical_dis_1point_lines(point, lines):
  '''
  point:[3]
  lines:[m,2,3]
  dis: [m]
  '''
  assert point.ndim == 1
  assert lines.ndim == 3
  assert lines.shape[1:] == (2,3) or lines.shape[1:] == (2,2)
  # use lines[:,0,:] as ref
  point = point.reshape([1,-1])
  ln = lines.shape[0]
  direc_p = point - lines[:,0,:]
  direc_l = lines[:,1,:] - lines[:,0,:]
  angles = angle_of_2lines(direc_p, direc_l, scope_id=0)
  dis = np.sin(angles) * np.linalg.norm(direc_p, axis=1)
  mask = np.isnan(dis)
  dis[mask] = 0
  return dis

def cam2world_pcl(points):
  R = np.eye(points.shape[-1])
  R[1,1] = R[2,2] = 0
  R[1,2] = 1
  R[2,1] = -1
  points = np.matmul(points, R)
  return points

def cam2world_box(box):
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

def lines_intersection_2d(line0s, line1s, must_on0=False, must_on1=False,
          min_angle=0):
    '''
    line0s: [n,2,2]
    line1s: [m,2,2]
    return [n,m,2,2]
    '''
    shape0 = line0s.shape
    shape1 = line1s.shape
    if shape0[0] * shape1[0] == 0:
        return np.empty([shape0[0], shape1[0], 2, 2])
    assert len(shape0) == len(shape1) == 3
    assert shape0[1:] == shape1[1:] == (2,2)
    ints_all = []
    for line0 in line0s:
      ints_0 = []
      for line1 in line1s:
        ints = line_intersection_2d(line0, line1, must_on0, must_on1, min_angle)
        ints_0.append(ints.reshape(1,1,2))
      ints_0 = np.concatenate(ints_0, 1)
      ints_all.append(ints_0)
    ints_all = np.concatenate(ints_all, 0)
    return ints_all


def line_intersection_2d(line0, line1, must_on0=False, must_on1=False,
          min_angle=0):
    '''
      line0: [2,2]
      line1: [2,2]
      must_on0: must on the scope of line0, no extend
      must_on1: must on the scope of line1, no extend
      out: [2]

      v01 = p1 - p0
      v23 = p3 - p2
      intersection = p0 + v01*k0 = p2 + v23 * k1
      [v01, v23][k0;-k1] = p2 - p0
      intersection between p0 and p1: 1>=k0>=0
      intersection between p2 and p3: 1>=k1>=0

      return [2]
    '''

    assert (line0.shape == (2,2) and line1.shape == (2,2))
            #(line0.shape == (2,3) and line1.shape == (2,3))
    dim = line0.shape[1]
    p0,p1 = line0
    p2,p3 = line1

    v01 = p1-p0
    v23 = p3-p2
    v01v23 = np.concatenate([v01.reshape([2,1]), (-1)*v23.reshape([2,1])], 1)
    p2sp0 = (p2-p0).reshape([2,1])

    try:
      inv_vov1 = np.linalg.inv(v01v23)
      K = np.matmul(inv_vov1, p2sp0)

      if must_on0 and (K[0]>1 or K[0]<0):
        return np.array([np.nan]*2)
      if must_on1 and (K[1]>1 or K[1]<0):
        return np.array([np.nan]*2)

      intersec = p0 + v01 * K[0]
      intersec_ = p2 + v23 * K[1]
      assert np.linalg.norm(intersec - intersec_) < 1e-5, f'{intersec} \n{intersec_}'

      direc0 = (line0[1] - line0[0]).reshape([1,2])
      direc1 = (line1[1] - line1[0]).reshape([1,2])
      angle = angle_of_2lines(direc0, direc1, scope_id=1)[0]
      angle = np.abs(angle)

      show = False
      if show and DEBUG:
        print(f'K:{K}\nangle:{angle}')
        lines_show = np.concatenate([np.expand_dims(line0,0), np.expand_dims(line1,0)],0)
        points_show = np.array([[intersec[0], intersec[1], 0]])
        Bbox3D.draw_points_lines(points_show, lines_show)

      if angle > min_angle:
        return intersec
      else:
        return np.array([np.nan]*2)
    except np.linalg.LinAlgError:
      return np.array([np.nan]*2)

def points_in_lines(points, lines, threshold_dis=0.03):
  '''
  points:[n,3]
  lines:[m,2,3]
  dis: [n,m]
  out: [n,m]

  (1)vertial dis=0
  (2) angle>90 OR corner dis=0
  '''
  num_p = points.shape[0]
  num_l = lines.shape[0]

  pc_distances0 = points.reshape([num_p,1,1,3]) - lines.reshape([1,num_l,2,3])
  pc_distances = np.linalg.norm(pc_distances0, axis=-1).min(2)

  pl_distances = vertical_dis_points_lines(points, lines)

  tmp_l = np.tile( lines.reshape([1,num_l,2,3]), (num_p,1,1,1) )
  tmp_p = np.tile( points.reshape([num_p,1,1,3]), (1,num_l,1,1) )
  dirs0 = tmp_l - tmp_p
  dirs1 = dirs0.reshape([num_l*num_p, 2,3])
  angles0 = angle_of_2lines(dirs1[:,0,:], dirs1[:,1,:])
  angles = angles0.reshape([num_p, num_l])

  mask_pc = pc_distances < threshold_dis
  mask_pl = pl_distances < threshold_dis
  mask_a = angles > np.pi/2
  in_line_mask = (mask_a + mask_pc) * mask_pl
  return in_line_mask

def is_extend_lines(lines0, lines1, threshold_dis=0.03):
  '''
  [n,2,3]
  [m,2,3]
  [n,m]
  '''
  n0 = lines0.shape[0]
  n1 = lines1.shape[0]
  dis0 = vertical_dis_points_lines(lines0.reshape([-1,3]), lines1)
  dis1 = dis0.reshape([n0,2,n1])
  mask0 = dis1 < threshold_dis
  mask1 = mask0.all(1)
  return  mask1

class OBJ_DEF():
  @staticmethod
  def limit_yaw(yaws, yx_zb):
    '''
    standard: [0, pi]
    yx_zb: [-pi/2, pi/2]
    '''
    if yx_zb:
      yaws = limit_period(yaws, 0.5, np.pi)
    else:
      yaws = limit_period(yaws, 0, np.pi)
    return yaws

  @staticmethod
  def check_bboxes(bboxes, yx_zb):
    '''
    x_size > y_size
    '''
    ofs = 1e-6
    if bboxes.shape[0]==0:
      return
    if yx_zb:
      #assert np.all(bboxes[:,3] <= bboxes[:,4]) # prediction may not mathch
      assert np.max(np.abs(bboxes[:,-1])) <= np.pi*0.5+ofs
    else:
      #assert np.all(bboxes[:,3] >= bboxes[:,4])
      assert np.max(bboxes[:,-1]) <= np.pi + ofs
      assert np.min(bboxes[:,-1]) >= 0 - ofs
