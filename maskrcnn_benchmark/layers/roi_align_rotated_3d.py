# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch, math
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from SparseConvNet.sparseconvnet.tools_3d_2d import sparse_3d_to_dense_2d
import _C


class _ROIAlignRotated3D(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        # input: [4, 256, 304, 200, 7]
        # roi: [171, 8]
        # spatial_scale: 0.25
        # output_size: [7,7,7]
        # sampling_ratio: 2
        output = _C.roi_align_rotated_3d_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], output_size[2], sampling_ratio
        ) # [171, 256, 7, 7]
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w, zsize = ctx.input_shape
        grad_input = _C.roi_align_rotated_3d_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            output_size[2],
            bs,
            ch,
            h,
            w,
            zsize,
            sampling_ratio,
        )
        return grad_input, None, None, None, None


roi_align_rotated_3d = _ROIAlignRotated3D.apply


class ROIAlignRotated3D(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        '''
        output_size:[pooled_height, pooled_width]
        spatial_scale: size_of_map/size_of_original_image
        sampling_ratio: how many points to use for bilinear_interpolate
        '''
        super(ROIAlignRotated3D, self).__init__()
        self.output_size = output_size # (7,7,7)
        self.spatial_scale = spatial_scale # 0.25
        self.sampling_ratio = sampling_ratio # 2

    def forward(self, input_s3d, rois_3d):
        '''
        input0: sparse 3d tensor
        rois_3d: 3d box, xyz order is same as input0,
                yaw unit is rad, anti-clock wise is positive

        input: [batch_size, feature, h, w]
        rois: [n,5] [batch_ind, center_w, center_h, roi_width, roi_height, theta]
        theta unit: degree, anti-clock wise is positive

        Note: the order of w and h inside of input and rois is different.
        '''
        input_d3d = sparse_3d_to_dense_2d(input_s3d)
        output = roi_align_rotated_3d(
            input_d3d, rois_3d, self.output_size, self.spatial_scale, self.sampling_ratio
        )
        return output

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr

