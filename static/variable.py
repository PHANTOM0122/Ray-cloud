import poselib
import pandas as pd
import os

# DATASET_PATH
DATASET_MOUNT = "../../mnt"

# GPU, CPU usage options
CUDA = 'cuda:0' # 'cpu' or 'cuda:0'

# SPARSITY LEVEL should have at least one value for any process to run
SPARSITY_LEVEL = [0.1]
# SPARSITY_LEVEL = [1.0, 0.5, 0.25, 0.1]
# NOISE LEVEL should have at least one value for any process to run
NOISE_LEVEL = [0.0]

##################################################
############## Line type selection ###############
##################################################
# Multiple LINE TPYE could be selected from ["OLC","PPL","PPLplus", "Rayclouds", "Raycloudsvoxel"]
LINE_TYPE = ["PC","OLC", "PPL", "PPLplus", "Rayclouds", "Raycloudsvoxel"]
# LINE_TYPE = ["Raycloudsvoxel"]

# Options for PPLplus
THR_LOOP = 1000
THR_PLANE = 30
THR_ANGLE = 20
##################################################
############# Options for Raycloud ###############
##################################################
# Option for prevent trivial solutions, by only using 3d rays from opposite center point.
USE_RAYS_FROM_OPPOSITE_CENTER = True
# Option for applying 3D Rays rejection sampling 
LINE_REJECTION = True # Apply line-rejction
REJECTION_RATIO = 0.25 # beta(%)

# Option for minimal solver for pose estimation 
POSE_SOLVER = 'p5+1r' # 'p5+1r', 'p6l' 
VOXEL_SIZE = 0.5 # meter-scale

##################################################
############ Outlier removal threshold ###########
##################################################
# nb_neighbors, which specifies how many neighbors are taken into account 
# in order to calculate the average distance for a given point.
THR_OUT_NN = 20
# std_ratio, which allows setting the threshold level 
# based on the standard deviation of the average distances across the point cloud. 
# The lower this number the more aggressive the filter will be.
THR_OUT_STD = 10

##################################################
###### Line to point reconstruction options ######
##################################################
# Chelani et al. (single-peak finding) options
REFINE_ITER = 3
# Structure Recovery
ESTIMATOR = ["SPF"]
# Ratio of swapping
# SWAP_RATIO = [1.0, 0.75, 0.5, 0.25, 0.0]
SWAP_RATIO = [0.0]

##################################################
################# Image recovery #################
##################################################
# Input atrribute choices=['depth','depth_sift','depth_rgb','depth_sift_rgb']
INPUT_ATTR = 'depth_sift_rgb'
# INPUT_ATTR = 'depth_rgb' # No feature test
# Scale choices=[256,394,512]
SCALE_SIZE = 512
# Crop choices=[256,512]
CROP_SIZE = 512

# Sample size must be bigger than 0
SAMPLE_SIZE = 32
SAMPLE_SIZE = max(SAMPLE_SIZE,1)

##################################################
########### Camera relocalization error ##########
##################################################

# Turn on off the refinement on pose estimation
REFINE_OPTION = True
RANDOM_SEED = 91
# Poselib options
BUNDLE_OPTIONS = poselib.BundleOptions()
RANSAC_OPTIONS = poselib.RansacOptions()
RANSAC_OPTIONS["min_iterations"] = 10
RANSAC_OPTIONS["max_epipolar_error"] = 2.0

##################################################
################## Raise errors ##################
##################################################

# Dataset check
CAMBRIDGE_DATASET = ["kingscollege","oldhospital", "shopfacade", "stmaryschurch"]
COLMAP_DATASET = ["south_building", "gerrard_hall", "graham_hall", "person_hall"]
ENERGY_DATASET = ["apt1_kitchen", "apt1_living", "apt2_bed", "apt2_kitchen", 
                    "apt2_living", "apt2_luke", "office1_gates362", "office1_gates381",
                    "office1_lounge", "office1_manolis", "office2_5a", "office2_5b"]

# SPARSITY LEVEL should have at least one value for any process to run
def raise_errors(dataset):
    work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Scale size should be smaller than crop size
    if SCALE_SIZE < CROP_SIZE:
        raise ValueError('SCALE_SIZE must be >= CROP_SIZE')
    
    # Query image & txt file list accordance check
    print("Check query image list accord with query textfile")
    basepath = os.path.join(work_dir, DATASET_MOUNT, getDatasetName(dataset),dataset)
    path_querytxt = os.path.join(basepath,'query_imgs.txt')

def getDatasetName(dataset):
    if dataset.lower() in CAMBRIDGE_DATASET:
        return "cambridge"
    
    if dataset.lower() in ENERGY_DATASET:
        return "energy"

    if dataset.lower() in COLMAP_DATASET:
        return "colmap"
    
    return "none"

def getScale(dataset):
    if dataset in CAMBRIDGE_DATASET or dataset in ENERGY_DATASET:
        return pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)),"SCALE.csv"))[dataset][0]
    else:
        return 1
    
