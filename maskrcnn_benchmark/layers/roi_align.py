# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

import _C

class _ROIAlign(Function):
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
        output = _C.roi_align_forward(
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
        grad_input = _C.roi_align_backward(
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


roi_align = _ROIAlign.apply


class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlign, self).__init__()
        self.output_size = output_size # (7,7)
        self.spatial_scale = spatial_scale # 0.25
        self.sampling_ratio = sampling_ratio # 2

    def forward(self, input, rois):
        '''
        input: [batch_size, feature, w, h]
        rois: [n,5] [batch_ind, start_w, start_h, end_w, end_h]
        '''
        assert rois.shape[1] == 5
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



if __name__ == '__main__':
    import torch

    align_roi = ROIAlign((2,2), 0.5, 2)
    feat = torch.arange(64).view(1, 1, 8, 8).float()
    # Note: first element is batch_idx
    rois = torch.tensor([
          [0, 0,1,2,3],
          [0, 0.2,1,2.1,3.4],
          ], dtype=torch.float32).view(-1, 5)

    print(f'feat:\n{feat}\nrois:\n{rois}')

    print('------------test on cpu------------')
    feat.requires_grad = False
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

