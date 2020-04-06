# xyz Jan 2019

import torch
import torch.nn as nn
import sparseconvnet as scn
from .sparseConvNetTensor import SparseConvNetTensor
import numpy as np

DEBUG = False
SHOW_MODEL = 0
CHECK_NAN = True

class FPN_Net(torch.nn.Module):
    _show = SHOW_MODEL
    def __init__(self, full_scale, dimension, raw_elements, reps, nPlanesF, nPlaneM, residual_blocks,
                  fpn_scales_from_top, roi_scales_from_top, downsample, rpn_map_sizes,
                  rpn_3d_2d_selector, leakiness=0, voxel_scale=None, bn_momentum=0.9, track_running_stats=True):
        '''
        downsample:[kernel, stride] :[[2,2,2], [2,2,2]]
        '''
        nn.Module.__init__(self)

        self.bn_momentum = bn_momentum
        self.track_running_stats = track_running_stats

        self.dimension = dimension
        self.down_kernels =  downsample[0]
        self.down_strides = downsample[1]
        self.fpn_scales_from_top = fpn_scales_from_top
        self.roi_scales_from_top = roi_scales_from_top
        scale_num = len(nPlanesF)
        assert len(self.down_kernels) == scale_num - 1 == len(self.down_strides), f"nPlanesF len = {scale_num}, kernels num = {len(self.down_kernels)}"
        assert all([len(ks)==3 for ks in self.down_kernels])
        assert all([len(ss)==3 for ss in self.down_strides])
        self._merge = 'add'  # 'cat' or 'add'

        ele_channels = {'xyz':3, 'color':3, 'normal':3}
        in_channels = sum([ele_channels[e] for e in raw_elements])

        self.layers_in_0 = scn.Sequential(
                scn.InputLayer(dimension,full_scale, mode=4))
        self.layers_in = scn.Sequential(
                scn.InputLayer(dimension,full_scale, mode=4),
                scn.SubmanifoldConvolution(dimension, in_channels, nPlanesF[0], 3, False))

        self.layers_out = scn.Sequential(
            scn.BatchNormReLU(nPlanesF[0], momentum=bn_momentum, track_running_stats=track_running_stats),
            scn.OutputLayer(dimension))

        self.linear = nn.Linear(nPlanesF[0], 20)
        self.voxel_scale = voxel_scale
        self.rpn_map_sizes = np.array( rpn_map_sizes )
        self.rpn_3d_2d_selector = rpn_3d_2d_selector

        self.convs_pro2d = nn.ModuleList()
        for zsize in self.rpn_map_sizes[:,-1]:
            self.convs_pro2d.append( scn.Convolution(self.dimension, nPlaneM, nPlaneM, [1,1,zsize], [1,1,1], False) )
        #**********************************************************************#

        def block(m, a, b):
            if residual_blocks: #ResNet style blocks
                m.add(scn.ConcatTable()
                      .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                      .add(scn.Sequential()
                        .add(scn.BatchNormLeakyReLU(a, momentum=bn_momentum,leakiness=leakiness, track_running_stats=track_running_stats))
                        .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
                        .add(scn.BatchNormLeakyReLU(b, momentum=bn_momentum,leakiness=leakiness, track_running_stats=track_running_stats))
                        .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False)))
                 ).add(scn.AddTable())
            else: #VGG style blocks
                m.add(scn.Sequential()
                     .add(scn.BatchNormLeakyReLU(a, momentum=bn_momentum,leakiness=leakiness, track_running_stats=track_running_stats))
                     .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False)))
            operation = {'kernel':[1,1,1], 'stride':[1,1,1]}
            return operation

        def down(m, nPlane_in, nPlane_downed, scale):
          #print(f'down, scale={scale}, feature={nPlane_in}->{nPlane_downed}, kernel={self.down_kernels[scale]},stride={self.down_strides[scale]}')
          m.add(scn.Sequential()
                  .add(scn.BatchNormLeakyReLU(nPlane_in, momentum=bn_momentum,leakiness=leakiness, track_running_stats=track_running_stats))
                  .add(scn.Convolution(dimension, nPlane_in, nPlane_downed,
                          self.down_kernels[scale], self.down_strides[scale], False)))
          operation = {'kernel':self.down_kernels[scale], 'stride':self.down_strides[scale]}
          return operation

        def up(m, nPlane_in, nPlane_uped, scale):
          #print(f'up, scale={scale}, feature={nPlane_in}->{nPlane_uped}, kernel={self.down_kernels[scale]}, stride={self.down_strides[scale]}')
          m.add( scn.BatchNormLeakyReLU(nPlane_in, momentum=bn_momentum, leakiness=leakiness, track_running_stats=track_running_stats)).add(
                      scn.Deconvolution(dimension, nPlane_in, nPlane_uped,
                      self.down_kernels[scale], self.down_strides[scale], False))
          operation = {'kernel':self.down_kernels[scale], 'stride':self.down_strides[scale]}
          return operation


        scales_num = len(nPlanesF)
        m_downs = nn.ModuleList()
        m_shortcuts = nn.ModuleList()
        operations_down = []
        for k in range(scales_num):
            m = scn.Sequential()
            if k > 0:
              op = down(m, nPlanesF[k-1], nPlanesF[k], k-1)
              operations_down.append(op)
            for _ in range(reps):
              op = block(m, nPlanesF[k], nPlanesF[k])
              if k==0:
                operations_down.append(op)
            m_downs.append(m)

            m = scn.SubmanifoldConvolution(dimension, nPlanesF[k], nPlaneM, 1, False)
            m_shortcuts.append(m)

        ###
        m_ups = nn.ModuleList()
        m_mergeds = nn.ModuleList()
        operations_up = []
        for k in range(scales_num-1, 0, -1):
            m = scn.Sequential()
            op = up(m, nPlaneM, nPlaneM, k-1)
            m_ups.append(m)
            operations_up.append(op)

            m_mergeds.append(scn.SubmanifoldConvolution(dimension, nPlaneM, nPlaneM, 3, False))

            #m = scn.Sequential()
            #for i in range(reps):
            #    block(m, nPlanesF[k-1] * (1+int(self._merge=='cat') if i == 0 else 1), nPlanesF[-1])
            #m_ups_decoder.append(m)

        self.m_downs = m_downs
        self.m_shortcuts = m_shortcuts
        self.m_ups = m_ups
        self.m_mergeds = m_mergeds
        self.operations_down = operations_down
        self.operations_up = operations_up




    def forward(self, net0):
      if CHECK_NAN:
        if not torch.isnan( self.layers_in[1].weight ).sum() == 0:
          self.check_grad_nan()
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass

      if self._show: print(f'\nFPN net input: {net0[0].shape}')
      net1 = self.layers_in(net0)
      net_scales = self.forward_fpn(net1)

      #net_scales = [n.to_dict() for n in net_scales]
      return net_scales

    def check_grad_nan(self):
      print_max_grad(self.convs_pro2d, 'self.convs_pro2d')
      print_max_grad(self.m_downs[0][0][1] , f'self.m_downs[0][0][1]')
      print_max_grad(self.m_downs[-1][0] , f'')
      print_max_grad(self.m_ups[-1] , f'')

      for i, up in enumerate(self.m_ups):
        print_max_grad(up , f'self.ups[{i}]')

      print_max_grad(self.m_shortcuts , f'shorcuts')

      print_max_grad(self.m_mergeds , f'm_mergeds')
      pass

    def forward_fpn(self, net):
      if self._show:
        print('input sparse format:')
        sparse_shape(net)

      scales_num = len(self.m_downs)
      downs = []
      #if self._show:    print('\ndowns:')
      for m in self.m_downs:
        net = m(net)
        #if self._show:  sparse_shape(net)
        downs.append(net)

      net = self.m_shortcuts[-1](net)
      ups = [net]
      #if self._show:    print('\nups:')
      #fpn_scales_from_top_from_back = [scales_num-1-i for i in self.fpn_scales_from_top]
      #fpn_scales_from_top_from_back.sort()
      for k in range(scales_num-1):
        #if k >= max(fpn_scales_from_top_from_back):
        #  continue
        j = scales_num-1-k-1
        net = self.m_ups[k](net)
        #if self._show:  sparse_shape(net)
        shorcut = self.m_shortcuts[j]( downs[j] )
        net = scn.add_feature_planes([ net, shorcut ])
        #net = self.m_ups_decoder[k](net)
        #if self._show:  sparse_shape(net)
        ups.append(self.m_mergeds[k](net))

      rpn_maps_3d = [ups[i] for i in self.fpn_scales_from_top]
      rpn_maps_2d = [ self.convs_pro2d[i](rpn_maps_3d[i]) for i in range(len(rpn_maps_3d)) ]
      rpn_maps = rpn_maps_3d + rpn_maps_2d
      rpn_maps = [rpn_maps[i] for i in self.rpn_3d_2d_selector]

      roi_maps = [ups[i] for i in self.roi_scales_from_top]


      for i in range(len(rpn_maps_3d)):
        assert torch.all(rpn_maps_3d[i].spatial_size == torch.tensor(self.rpn_map_sizes[i]))

      if self._show:
            receptive_field(self.operations_down, self.voxel_scale)
            print('\n\nSparse FPN\n--------------------------------------------------')
            print(f'scale num: {scales_num}')
            print('downs:')
            for i in range(len(downs)):
              #if i!=0:
              #  print(f'\tKernel:{self.down_kernels[i-1]} stride:{self.down_strides[i-1]}', end='\t')
              #else:
              #  print('\tSubmanifoldConvolution \t\t', end='\t')
              op = self.operations_down[i]
              ke = op['kernel']
              st = op['stride']
              rf = op['rf']
              tmp = f' \tKernel:{ke}, Stride:{st}, Receptive:{rf}'
              sparse_shape(downs[i], pre=f'\t{i} ', post=tmp)

            print('\n\nups:')
            for i in range(len(ups)):
              #if i==0:
              #  print('\tIdentity of the last \t\t', end='\t')
              #else:
              #  print(f'\tKernel:{self.down_kernels[-i]} stride:{self.down_strides[-i]}', end='\t')
              if i<len(self.operations_up):
                op = self.operations_up[i]
                ke = op['kernel']
                st = op['stride']
                rf = self.operations_down[-i-1]['rf']
                tmp = f' \tKernel:{ke}, Stride:{st}, Receptive:{rf}'
              else:
                tmp = ''
              sparse_shape(ups[i], pre=f'\t{i} ', post=tmp)

            print('\n\nFPN_Net out:')
            print(f'{self.fpn_scales_from_top} of ups')
            receptive_fields_fpn = [self.operations_down[-i-1]['rf'] for i in self.fpn_scales_from_top]
            receptive_fields_fpn = receptive_fields_fpn + receptive_fields_fpn
            for j,t in enumerate(rpn_maps):
              s_j = self.rpn_3d_2d_selector[j]
              tmp = f'\t Receptive:{receptive_fields_fpn[s_j]}'
              sparse_shape(t, post=tmp)
              sparse_real_size(t,'\t')
              print('\n')

            print('\n\nROI map:')
            print(f'{self.roi_scales_from_top} of ups')
            receptive_fields_roi = [self.operations_down[-i-1]['rf'] for i in self.roi_scales_from_top]
            for j,t in enumerate(roi_maps):
              tmp = f'\t Receptive:{receptive_fields_roi[j]}'
              sparse_shape(t, post=tmp)
              sparse_real_size(t,'\t')
              print('\n')
            print('--------------------------------------------------\n\n')
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass

      return rpn_maps, roi_maps


def receptive_field(operations, voxel_scale = None):
  '''
  https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
  '''
  n = len(operations)
  operations[0]['rf'] = np.array([1.0,1,1])
  jump = 1
  for i in range(1,n):
    op = operations[i]
    ke = np.array(op['kernel'])
    st = np.array(op['stride'])
    rf = operations[i-1]['rf'] + (ke-1)*jump
    operations[i]['rf'] = rf
    jump *= st

  if voxel_scale:
    for op in operations:
      op['rf'] /= 1.0*voxel_scale

def sparse_shape(t, pre='\t', post=''):
  loc = t.get_spatial_locations()
  batch_size = loc[:,-1].max().float() + 1
  sparse_rate = 1.0 * t.features.shape[0] / t.spatial_size.prod().float() / batch_size
  print(f'{pre}{t.features.shape}, {t.spatial_size}{post}, sparse_rate:{sparse_rate}')

def sparse_real_size(t,pre=''):
  loc = t.get_spatial_locations()
  loc_min = loc.min(0)[0]
  loc_max = loc.max(0)[0] + 1
  sparse_rate = 1.0 * t.features.shape[0] / loc_max.prod().float()
  print(f"{pre}min: {loc_min}, max: {loc_max}, sparse rate:{sparse_rate}")

def print_max_grad(module_list, flag):
    print(f'\n------------\n{flag}:')
    for i, mod in enumerate(module_list):
      if not hasattr(mod, 'weight'):
        print(f'\t{i}\tno weight')
        continue
      if mod.weight.grad is not None:
        max_grad = mod.weight.grad.max()
        print(f'\t{i}\t{max_grad:.5f}')
      else:
        print(f'\t{i}\tno grad')



