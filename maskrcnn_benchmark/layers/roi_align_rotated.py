# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

import numpy as np
import _C

class _ROIAlignRotated(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        # input: [4, 256, 304, 200]
        # roi: [171, 5]
        # spatial_scale: 0.25
        # output_size: [7,7]
        # sampling_ratio: 2
        output = _C.roi_align_rotated_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio
        ) # [171, 256, 7, 7]
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_rotated_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
        )
        return grad_input, None, None, None, None


roi_align = _ROIAlignRotated.apply


class ROIAlignRotated(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        '''
        output_size:[pooled_height, pooled_width]
        spatial_scale: size_of_map/size_of_original_image
        sampling_ratio: how many points to use for bilinear_interpolate
        '''
        super(ROIAlignRotated, self).__init__()
        self.output_size = output_size # (7,7)
        self.spatial_scale = spatial_scale # 0.25
        self.sampling_ratio = sampling_ratio # 2

    def forward(self, input, rois):
        '''
        input: [batch_size, feature, h, w]
        rois: [n,5] [batch_ind, center_w, center_h, roi_width, roi_height, theta]
            theta unit: degree, anti-clock wise is positive
        '''
        assert rois.shape[1] == 6
        return roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        )


    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr



def test1():
    align_roi = ROIAlignRotated((1, 3), 1, 2)
    feat = torch.arange(64).view(1, 1, 8, 8).float()
    # Note: first element is batch_idx
    rois = torch.tensor([
          [0, 3,3, 3,1, 0],
          [0, 3,3, 3,1, 90],
          [0, 3,3, 3,1, -90],
          [0, 3,3, 3,1, 30],
          [0, 3,3, 3,1, 60],
          ], dtype=torch.float32).view(-1, 6)

    print(f'feat:\n{feat}\nrois:\n{rois}')

    print('------------test on cpu------------')
    feat.requires_grad = False
    if False:
      out = align_roi(feat, rois)
      print(out)
      print('cpu version do not support backward')
    #out.sum().backward()
    #print(feat.grad)

    if torch.cuda.is_available():
        print('------------test on gpu------------')
        feat = feat.detach().cuda()
        rois = rois.cuda()
        feat.requires_grad = True
        out = align_roi(feat, rois)
        print(out)
        temp = out.sum()
        temp.backward()
        print(feat.grad)
    else:
        print('You device have not a GPU')

def test2():
  '''
  image: [h_size,w_size]
  rois: [n,5] [batch_ind, center_w, center_h, roi_width, roi_height, theta]
  the order of w and h is different
  anti-clock wise is right
  '''
  import matplotlib.pyplot as plt
  from skimage import data, color
  from skimage.transform import rescale, resize, downscale_local_mean
  from skimage.io import imread

  image0 = imread('./y.jpg')
  image0 = color.rgb2gray(image0)
  image1 = rescale(image0, scale=1/16.0, mode='reflect', multichannel=True)
  print(f"0: {image0.shape}")
  print(f"1: {image1.shape}")

  align_roi_0 = ROIAlignRotated((100,300), 1, 1)
  feat = torch.tensor(image0).to(torch.float32)
  feat = feat.unsqueeze(0).unsqueeze(0)
  print(f"f:{feat.shape}")

  # Note: first element is batch_idx
  rois_0 = torch.tensor([
        [0, 786,220, 300, 100, 0],
        [0, 786,220, 300, 100, 90],
        [0, 786,220, 300, 100, -45],
        [0, 786,220, 300, 100, -90],
        ], dtype=torch.float32).view(-1, 6)

  assert torch.cuda.is_available()
  print('------------test on gpu------------')
  feat = feat.detach().cuda()
  feat.requires_grad = True
  rois_0 = rois_0.cuda()
  out_0 = align_roi_0(feat, rois_0)
  temp = out_0.sum()
  temp.backward()
  #print(feat.grad)

  roi_image_0 = out_0[0,0].cpu().data.numpy()
  roi_image_1 = out_0[1,0].cpu().data.numpy()
  roi_image_2 = out_0[2,0].cpu().data.numpy()
  roi_image_3 = out_0[3,0].cpu().data.numpy()


  bi, cw,ch, sw,sh, yaw = rois_0[0].cpu().data.numpy().astype(np.int32)
  h0 = int(ch-sh/2)
  h1 = int(ch+sh/2)
  w0 = int(cw-sw/2)
  w1 = int(cw+sw/2)

  c0 = image0[h0:h1, w0:w1]
  print(f"crop: {h0}:{h1}, {w0}:{w1} -> {c0.shape}")

  fig, ax = plt.subplots(3, 2, figsize=(16, 10))

  ax[0,0].imshow(image0)
  ax[1,1].imshow(roi_image_0)
  ax[0,1].imshow(c0)
  ax[1,0].imshow(roi_image_1)
  ax[2,0].imshow(roi_image_2)
  ax[2,1].imshow(roi_image_3)
  plt.show()
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  return


if __name__ == '__main__':
    # note: output_size: [h,w],  rois: [w,h]
    # order is different
    #test1()
    test2()

