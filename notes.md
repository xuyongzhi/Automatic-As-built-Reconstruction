0. 0005b50577f5871e1c0bb7a687f6cbc3 nan loss
1. wall on the first 10: nan
2. only 0004d52d1aeeb8ae6de39d6bd993e992:  60 epoch 0.8
   only 0004dd3cb11e50530676f77b55262d38: 0.8 fluctuant
3. top 5, nan in first iteration
4. 5-10, 0.4 36 epoch, 0. 1 loss


1. self.conv in RPNHead in rpn_sparse3d.py 
    Originally, kernel is 3, but is set as 1 now. Check if submanifold with kernel_size=3 is required.
2. clip_to_pcl in bounding_box_3d.py not implemented yet
3. remove_small_boxes3d is not enabled in inference_3d.py
4. About boxlist.clip_to_image 
        Originally, it is performed in forward_for_single_feature_map in RPNPostProcessor in inference.py
        This is a force fix of proposals. 
        What about directly clip the anchor size by scene size?
        Currently, do no clip yet.
5. some boxes with yaw !=0 and !=90 are not cropped properly
6. Add dirction loss later
