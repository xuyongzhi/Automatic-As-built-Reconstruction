import os
import h5py
import numpy as np
import torch

STANFORD_PTH_PATH = '/DS/2D-3D-Semantics_Stanford/2D-3D-Semantics_Pth/'

def random_sample_pcl(points0, num_points1, only_reduce=False):
  n0 = points0.shape[0]
  if num_points1 == n0:
    return points0
  if num_points1 < n0:
    indices = np.random.choice(n0, num_points1, replace=False)
  else:
    if only_reduce:
      return points0
    indices0 = np.random.choice(n0, num_points1-n0, replace=True)
    indices = np.concatenate([np.arange(n0), indices0])
  points1 = np.take(points0, indices, 0)
  return points1


def read_point_could(AREA, RAW_PATH, PTH_PATH  ):
    rooms = {}
    mat0 = h5py.File(f'{RAW_PATH}/pointcloud.mat', 'r')
    names = mat0[f"Area_{AREA}/Disjoint_Space/name"]
    room_names = []
    for i in range(names.shape[1]):
        for j in range(names.shape[0]):
            room_names.append(''.join([chr(v[0]) for v in mat0[(names[j][i])]]))
    # print(room_names)
    object_names = {}
    for room_index in range(len(room_names)):
        rgb_points = []
        mybox = {}
        Bbox = {}
        box_class = []
        rgb_points = None
        object_refs = mat0[f"Area_{AREA}/Disjoint_Space/object"][room_index]
        for object_ref in object_refs:
            room_object = mat0[object_ref]
            names = room_object['name']
            object_names[room_names[room_index]] = []
            for j in range(names.shape[0]):
                n = ''.join([chr(v[0]) for v in mat0[(names[j][0])]])
                object_names[room_names[room_index]].append(n)
                box_class.append(n.split('_')[0])
                Bbox[box_class[j]] = []
                mybox[box_class[j]] = []
            for j in range(names.shape[0]):
                points_ref = mat0[object_ref]['points'][j][:]
                rgbs_ref = mat0[object_ref]['RGB_color'][j][:]
                bbox_ref = mat0[object_ref]['Bbox'][j][0]
                for point,rgb in zip(points_ref,rgbs_ref):
                    if rgb_points is None:
                        rgb_points = np.asarray(
                            [x for x in np.asarray(mat0[point])] + [c for c in np.asarray(mat0[rgb])],
                            dtype=np.float32).T
                    else:
                        rgb_points = np.vstack((rgb_points, np.asarray([x for x in np.asarray(mat0[point])]
                                                                       + [c for c in np.asarray(mat0[rgb])],
                                                                       dtype=np.float32).T))
                box = []
                for x in mat0[bbox_ref]:
                    try:
                        box.append(x[0])
                    except:
                        print(j,'_',object_names[room_names[room_index]][j],'loss_box from',room_names[room_index])
                        break
                if box:
                    bbox = np.asarray(box)
                    # print(bbox)
                    Xmin = bbox[0]
                    Ymin = bbox[1]
                    Zmin = bbox[2]
                    Xmax = bbox[3]
                    Ymax = bbox[4]
                    Zmax = bbox[5]
                    xc = (Xmin + Xmax) / 2
                    x_size = Xmax - Xmin
                    yc = (Ymin + Ymax) / 2
                    y_size = Ymax - Ymin
                    zc = (Zmin + Zmax) / 2
                    z_size = Zmax - Zmin
                    yaw_s = 0
                    mybox[box_class[j]].append([xc, yc, zc, x_size, y_size, z_size, yaw_s])
            for key in mybox.keys():
                if mybox[key]:
                    #if key == 'wall':
                    #    Bbox[key] = torch.tensor(mybox[key], dtype=torch.float32)
                    #else:
                    Bbox[key] = np.asarray(mybox[key], dtype=np.float32)
                else:
                    del Bbox[key]
            # print(object_names[room_names[room_index]])
        rgb_points[:, 3:6] = rgb_points[:, 3:6] / 255.0
        rgb_points = random_sample_pcl(rgb_points,int( 1e6))
        n = rgb_points.shape[0] / 1000
        #print(f'\n\tpoint num : {n}\n')
        rooms[room_names[room_index]] = (rgb_points, Bbox)
    return room_names, rooms



def mat_2_pth():
    for AREA in [1,2,3,4,'5a','5b',6]:
      RAW_PATH = f'/DS/2D-3D-Semantics_Stanford/2D-3D-Semantics_Raw/area_{AREA}_no_xyz/area_{AREA}/3d'
      PTH_PATH = f'{STANFORD_PTH_PATH}/houses/area_{AREA}'
      if not os.path.exists(PTH_PATH):
          os.makedirs(PTH_PATH)
      room_names, pcds = read_point_could(AREA, RAW_PATH, PTH_PATH)
      # print(pcds[room_names[2]])
      for room in room_names:
        res_fn = f'{PTH_PATH}/{room}.pth'
        torch.save(pcds[room], res_fn)
        print(f'room {room} save ok: \n\t{res_fn}')

def gen_train_eval_list():
  res_path = f'{STANFORD_PTH_PATH}/train_test_splited'
  if not os.path.exists(res_path):
    os.makedirs(res_path)
  res_fn = os.path.join(res_path, 'train_.txt')

if __name__ == "__main__":
  #mat_2_pth()
  gen_train_eval_list()
