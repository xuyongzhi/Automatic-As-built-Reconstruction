import numpy as np
from wall_preprocessing import merge_2pieces_of_1wall
from utils3d.bbox3d_ops import Bbox3D

box0 = np.array([[ 43.59999866, -41.88736687,   1.36749997,   3.17999993,   0.09473495,   2.73499994,   0.        ]])
box1 = np.array([[ 43.79499936, -41.95263138,   1.36749997,   2.80999994,   0.09473877,   2.73499994,   0.        ]])

box_merge = merge_2pieces_of_1wall(box0, box1, 'Y')
print(box_merge)


show_boxes = np.concatenate([box0, box1], 0)
Bbox3D.draw_bboxes(show_boxes, 'Z', False)
