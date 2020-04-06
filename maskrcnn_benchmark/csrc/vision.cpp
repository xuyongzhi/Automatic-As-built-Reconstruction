// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "SigmoidFocalLoss.h"
#include "ROIAlignRotated.h"
#include "ROIAlignRotated3D.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
  m.def("sigmoid_focalloss_forward", &SigmoidFocalLoss_forward, "SigmoidFocalLoss_forward");
  m.def("sigmoid_focalloss_backward", &SigmoidFocalLoss_backward, "SigmoidFocalLoss_backward");
  m.def("roi_align_rotated_forward", &ROIAlignRotated_forward, "ROIAlignRotated_forward");
  m.def("roi_align_rotated_backward", &ROIAlignRotated_backward, "ROIAlignRotated_backward");
  m.def("roi_align_rotated_3d_forward", &ROIAlignRotated3D_forward, "ROIAlignRotated3D_forward");
  m.def("roi_align_rotated_3d_backward", &ROIAlignRotated3D_backward, "ROIAlignRotated3D_backward");
}
