# xyz 2019
import torch
import sparseconvnet as scn

DEBUG = False

def sparse_3d_to_dense_2d(feat_s3d):
  '''
  convert a sparse 3d tensor to dense 2d
  (1) it is actually 2d, z dim is 1
  (2) it is actually dense in xy-dim
  '''
  spatial_size = feat_s3d.spatial_size
  #assert spatial_size[2] == 1, "dim z has to be compressed to 1"

  locations_3d0 = feat_s3d.get_spatial_locations() # [x,y,z,batch_idx]
  max_map_size = locations_3d0.max(0)[0] + 1
  x_size, y_size, z_size, batch_size = max_map_size
  #assert z_size == 1, "dim z has to be compressed to 1"
  total_n = feat_s3d.features.shape[0]
  sparse_rate = total_n*1.0/max_map_size.prod().to(torch.float)
  #print(f'sparse_rate:{sparse_rate}')

  nPlane0 = feat_s3d.features.shape[1]
  to_dense_layer =  scn.sparseToDense.SparseToDense(dimension=4, nPlanes=nPlane0)
  features_3d_flat = to_dense_layer(feat_s3d) # [batch_size, channels_num, x_size, y_size, z_size]

  features_3d_flat = features_3d_flat[:,:,0:x_size, 0:y_size, 0:z_size]

  if DEBUG:
    if not total_n == max_map_size.prod():
      print( f"\nthe 2d feature map is not dense, total_n:{total_n}, max_map_size:{max_map_size}")
      for bi in range(batch_size):
        for i in range(x_size):
          for j in range(y_size):
            mask = locations_3d0[:,0:2] == torch.tensor([[i,j]])
            mask = mask.all(dim=1)
            k = torch.nonzero(mask)
            matched_ij = mask.max()==1
            fij2d = features_2d[bi,0:5,i,j]
            if  matched_ij :
              fij3d = feat_s3d.features[k[0,0]][0:5]
              assert torch.all( fij2d == fij3d )
            else:
              assert torch.all(fij2d==0)
              pass
      pass
  return features_3d_flat

