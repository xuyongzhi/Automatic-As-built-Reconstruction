# 3D Object Detection with Deep Neural Networks for Automatic As-built Reconstruction
This project is developed based on the following three projects:
- https://github.com/facebookresearch/maskrcnn-benchmark.git
- https://github.com/facebookresearch/SparseConvNet.git
- https://github.com/traveller59/second.pytorch.git

Install required softwares according to the guide ["./docs/Install.md"](./docs/Install.md).  
Train or test by modifying and running the script ["./run.sh"](./run.sh).

# Task
* Input: point cloud of indoor building. 
* Five objects: wall, window, door, ceiling, floor
* Output: 3D bounding boxes of objectis

# Data preparation
The dataset is modified from SUNCG by the processing functions in: ./data3d/suncg_utils/suncg_preprocess.py  

# Assumption
* Shortest wall instance: Long wall pieces are croped by intersected ones to generate short instances.

# Detection Performance
* time per building: 4.75 s

| | Wall | Window | Door | Floor| Ceiling | Classes Mean |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| AP(%)   | 89.12|	76.42|	90.91|	83.57|	88.91|	85.79 |
| AIoU(%) | 84.09|	67.92|	80.20|	78.53|	84.42|	79.03|


# Activities
* Looking for cooperation to improve the quality of the dataset and rich it with real scanning data.

# Scene 1
Each intance is colorized by a random color, except blue denotes incorrect detection or missed ground truth.

|Synthetic mesh  | Point Cloud   |
| :-------------: | :-------------: |
| ![Mesh1](./docs/detect_res/1/mesh1.png)  | ![Pcl1](./docs/detect_res/1/pcl1.png) |
| **Ground truth** | **Detection** |
| ![Gt1](./docs/detect_res/1/gt1.png) | ![Det1](./docs/detect_res/1/det1.png) | 

# Scene 2
  
|Synthetic mesh  | Point Cloud   |
| :-------------: | :-------------: |
| ![Mesh2](./docs/detect_res/2/mesh2.png)  | ![Pcl2](./docs/detect_res/2/pcl2.png)  |
| **Ground truth** | **Detection** |
|![Gt2](./docs/detect_res/2/gt2.png) | ![Det2](./docs/detect_res/2/det2.png)   |

# Scene 3
  
|Synthetic mesh  | Point Cloud   |
| :-------------: | :-------------: |
|  ![Mesh3](./docs/detect_res/3/mesh3.png)   |![Pcl3](./docs/detect_res/3/pcl3.png) |
| **Ground truth** | **Detection** |
| ![Gt3](./docs/detect_res/3/gt3.png)  | ![Det3](./docs/detect_res/3/det3.png) |

# Scene 4
  
|Synthetic mesh  | Point Cloud   |
| :-------------: | :-------------: |
|  ![Mesh4](./docs/detect_res/4/mesh4.png)  | ![Pcl4](./docs/detect_res/4/pcl4.png) |
| **Ground truth** | **Detection** |
|![Gt4](./docs/detect_res/4/gt4.png)  | ![Det4](./docs/detect_res/4/det4.png)  |

