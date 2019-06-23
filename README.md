Work in progress

# UnifiedPoseEstimation
Implementation of H+O: Unified Egocentric Recognition of 3D Hand-Object Poses and Interactions

This repository provides the code for training a model that can jointly predict hand pose, object pose, object class, and the performed action.

Disclaimer: I have actually skipped the RNN part in the implementation. With a little effort it can be done. 

### Dependencies
1. PyTorch
2. tqdm
3. tensorboardX
4. torchvision
5. trimesh
6. matplotlib

This has been tested on Python 2.7, Ubuntu 16.04, and Pytorch 1.0.1.post2. 

### Directory Structure
UnifiedPoseEstimation
  cfg
  data
  models
  unified_pose_estimation

If any of the above directory is not present, please create them. 

You need to download and place the FPHA dataset in data directory. The data directory structure will look like
  Hand_pose_annotation_v1
  Object_models
  Subjects_info
  Video_files
  ..etc

Now cd into punified_pose_estimation

### Training
python clean.py
python train.py

### Test
python test.py
