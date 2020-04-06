
#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor ROIAlignRotated3D_forward(const at::Tensor& input,
                            const at::Tensor& rois,
                            const float spatial_scale,
                            const int pooled_height,
                            const int pooled_width,
                            const int pooled_zsize,
                            const int sampling_ratio) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlignRotated3D_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width, pooled_zsize, sampling_ratio);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor ROIAlignRotated3D_backward(const at::Tensor& grad,
                             const at::Tensor& rois,
                             const float spatial_scale,
                             const int pooled_height,
                             const int pooled_width,
                             const int pooled_zsize,
                             const int batch_size,
                             const int channels,
                             const int height,
                             const int width,
                             const int zsize,
                             const int sampling_ratio) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlignRotated3D_backward_cuda(grad, rois, spatial_scale, pooled_height, pooled_width, pooled_zsize, batch_size, channels, height, width, zsize, sampling_ratio);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

