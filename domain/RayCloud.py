import random
import os
from .master import Master
import numpy as np
import poselib
import time
import math
import open3d as o3d
from collections import defaultdict
import itertools
import test_module.linecloud as lineCloudTest
import test_module.recontest as recontest
import matplotlib.pyplot as plt

import utils.pose.pose_estimation as pe
import utils.pose.vector as Vector
from utils.pose import dataset
from utils.pose import line
from utils.l2precon import calculate
from static import variable
from sklearn.cluster import KMeans

from utils.pose.vector import *
from utils.l2precon import save
from utils.invsfm import reconstruction
from utils.colmap.read_write_model import *

# Set random seed
np.random.seed(variable.RANDOM_SEED)
random.seed(variable.RANDOM_SEED)

class RayCloud(Master):
    def __init__(self, dataset_path, output_path):

        # COLMAP 3D Point ID -> Line direction vector mapping
        self.pts_to_line = dict()

        # COLMAP 3D point ID -> Center point mapping
        self.pts_to_center = dict()
        self.pts_to_center_indices = dict()

        # Line index -> COLMAP 3D Point ID mapping
        self.line_to_pts = dict()

        # 3D lines unit direction vectors
        self.line_3d = None

        # Load data from COLMAP
        self.pts_2d_query = None  # Images.txt
        self.pts_3d_query = None  # Points3D.txt
        self.camera_dict_gt = None  # cameras.txt
        
        self.queryIds = None
        self.queryNames = None
        self.image_dict_gt = None

        self.resultPose = list() # Estimated poses results
        self.resultRecon = list() # Reconstructed results
        self.map_type = "RayCloud" 
        self.points_3D_recon = list() # 3D points to reconstruct 
        self.lines_3D_recon = list() # 3D lines to reconstruct

        super().__init__(dataset_path, output_path)
        self.pose_solver = variable.POSE_SOLVER # Type of minimal solver for pose estimation (p6L, p5+1R)
        self.num_clusters = 2 # Number of center points(K=2) with K-means clustering

        # Ids of 3D points from COLMAP
        self.pts_3d_ids = list(self.pts_3d_query.keys())

        # Shuffle points
        np.random.shuffle(self.pts_3d_ids)
        
    def makeLineCloud(self):

        print("Ray Clouds: Line between points and center")
        _pts_3d = np.array([v.xyz for v in self.pts_3d_query.values()]) # Get (x,y,z) of 3D points
        _pts_ids = np.array([k for k in self.pts_3d_query.keys()]) # Get COLMAP indices of 3D points

        # Performing K-means clustering 
        print('K-means cluster point clouds')
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=variable.RANDOM_SEED, n_init=10)
        kmeans.fit(_pts_3d)
        center_tmp = kmeans.cluster_centers_
        self.center_pts = np.array(center_tmp)
        print('Center points:', self.center_pts)

        # Draw 3D lines(rays)
        self.points_3D, self.line_3d, self.ind_to_id, self.id_to_ind, self.center_points, self.lines_center_indices = line.drawlines_rayclouds(_pts_3d, _pts_ids, self.center_pts)

        for i, key in enumerate(self.pts_3d_query.keys()):
            self.pts_to_line[key] = self.line_3d[i] # 3D point COLMAP IDs -> 3D Line unit direction vectors
            self.line_to_pts[i] = key # 3D Line direction vectors index -> 3D point COLMAP IDs
            self.pts_to_center[key] = self.center_points[i] # Save corresponding center point position paired with ith 3D point
            self.pts_to_center_indices[key] = self.lines_center_indices[i] # Save corresponding center point index paired with ith 3D point

        # Visualize constructed 3D ray clouds
        visualize = False
        if visualize == True:

            pts = _pts_3d
            lines = self.line_3d
            print(lines.shape)

            pair = []
            num_pts = len(pts)
            portion = int(num_pts * 0.01)
            line_indices = np.random.choice(np.arange(num_pts), portion)
            for i in line_indices:
                pair.append((i, i))

            pts_0 = pts - lines * 10
            pts_1 = pts + lines * 10
            pcd_0 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_0))
            pcd_1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_1))
            lcd = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd_0, pcd_1, pair)

            pcd_origin = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            pcd_origin.paint_uniform_color([0, 0, 0])
            lcd.paint_uniform_color([0.7, 0.7, 0.7])
            o3d.visualization.draw_geometries([pcd_origin])
            
    def maskSparsity(self, sparisty_level):

        # Set reduced length of number of 3D points
        new_len = int(len(self.pts_3d_ids) * sparisty_level)
        self.sparse_line_3d_ids = set(self.pts_3d_ids[:new_len])

    def matchCorrespondences(self, query_id):

        # Get stored 2D-3D correspondences in query image with COLMAP's database
        connected_pts3d_idx = np.where(self.pts_2d_query[query_id].point3D_ids != -1)[0]
        connected_pts3d_ids = self.pts_2d_query[query_id].point3D_ids[connected_pts3d_idx]

        p2 = np.array([self.pts_3d_query[k].xyz for k in connected_pts3d_ids], dtype=np.float64) # 3D key points (x,y,z)
        x1 = np.array(self.pts_2d_query[query_id].xys[connected_pts3d_idx], dtype=np.float64) # 2D key points (x,y)

        # Connected points COLMAP IDs -> numpy indices
        pts_to_ind = {}

        # Mapping Connected 3D points COLMAP IDs -> point indices!
        for _i, k in enumerate(connected_pts3d_ids):
            pts_to_ind[k] = _i

        # Get valid indices of lines observed in query image (2D points - 3D Lines correspondences)
        self.valid_pts_3d_ids = self.sparse_line_3d_ids.intersection(set(connected_pts3d_ids))

        newIndex = [] # Point(line) index only in valid 3D pts (lines)
        _x2 = [] # Direction vector of line in valid lines
        lines_center_indices = [] # Indices of center point connected with rays

        # pts_to_ind : Mapping valid 3D points COLMAP IDs to 3D points list indices!
        for _pid in self.valid_pts_3d_ids:
            newIndex.append(pts_to_ind[_pid])
            _x2.append(self.pts_to_line[_pid])
            lines_center_indices.append(self.pts_to_center_indices[_pid])
        lines_center_indices = np.array(lines_center_indices)

        # newIndex isn't empty
        if newIndex:
            newIndex = np.array(newIndex)
            # x1: 2D Key points
            # p2: 3D Offset(Point)
            # x2: 3D Line
            self._x1 = x1[newIndex]
            self._p2 = p2[newIndex]
            self._x2 = np.array(_x2)

            # Make matches between query images and virutal camera rays
            self.mathces = []
            for idx in range(len(self.center_pts)):
                self.mathces.append(Vector.make_matches(self._x1[lines_center_indices == idx], -self._x2[lines_center_indices == idx], virtual_cam_id=idx))

            self.cam1_ext = []
            self.cam2_ext = []

            for i in range(len(self.center_pts)):
                camera1_ext = poselib.CameraPose()
                camera1_ext.R = np.identity(3)
                camera1_ext.t = -self.center_pts[i]
                self.cam1_ext.append(camera1_ext)
            
            # Query camera     
            camera2_ext_id0 = poselib.CameraPose()
            camera2_ext_id0.R = np.identity(3)
            camera2_ext_id0.t = [0.0, 0.0, 0.0]
            self.cam2_ext.append(camera2_ext_id0)

        else:
            self._x1 = np.array([])
            self._p2 = np.array([])
            self._x2 = np.array([])

        print("Found correspondences: ", self._x1.shape[0])

    def addNoise(self, noise_level):
        super().addNoise(noise_level)

    def estimatePose(self, query_id):

        # If the number of 2D-3D correspondences are less than (5* #center points), skip
        if self._x1.shape[0] < (5 * len(self.center_pts)):
            return 0

        # If there's more than 6 points
        if self._x1.shape[0] >= 6:
            gt_img = pe.get_GT_image(query_id, self.pts_2d_query, self.image_dict_gt)
            cam_id = gt_img.camera_id
            cam_p6l = [pe.convert_cam(self.camera_dict_gt[cam_id])]
            cam_virtual = {'model':'SIMPLE_PINHOLE', 'width':0,'height':0, 'params':[0,0,0]}
              
            # P6L pose estimation
            if self.pose_solver == 'p6l':
                start_time = time.time()
                res = poselib.estimate_p6l_relative_pose(self._x1, self._p2, self._x2, cam_p6l, cam_p6l, variable.RANSAC_OPTIONS, variable.BUNDLE_OPTIONS, variable.REFINE_OPTION)

            # P5+1R pose estimation
            elif self.pose_solver == 'p5+1r':
                start_time = time.time()
                res = poselib.estimate_generalized_relative_pose(self.mathces, self.cam1_ext, [cam_virtual, cam_virtual], self.cam2_ext, [cam_p6l[0]], variable.RANSAC_OPTIONS, variable.BUNDLE_OPTIONS, variable.REFINE_OPTION)

            # Localization time
            end_time = time.time()
            pose_time = end_time - start_time
            
            super().savePoseAccuracy(res, gt_img, cam_p6l[0], pose_time)

    def savePose(self, sparisty_level, noise_level):
        super().savePose(sparisty_level, noise_level, self.pose_solver)

    def saveAllPoseCSV(self):
        super().saveAllPoseCSV()

    def recoverPts(self, estimator, sparsity_level, noise_level):
        
        print("Ray Clouds recover 3D points", "\n")
        if estimator == 'SPF':
            
            print("Single peak-finding from Chelani et al.", "\n")
            self.sparse_pts_3d_ids = [] # 3D points IDs of COLMAP to reconstruct
            self.id_to_ind_recon = {} # 3D points IDs of COLMAP -> 3D points list
            self.ind_to_id_recon = {} # 3D points list -> 3D points IDs of COLMAP
            self.points_3D_recon = [] # list of 3D points to reconstruct
            self.lines_3D_recon = [] # list of 3D lines to reconstruct
            
            self.pts_to_centers_recon = []  # list of center points
            self.pts_to_center_indices_recon = []  # list of center points indices

            for i, _pts_3d_id in enumerate(self.sparse_line_3d_ids):
                self.sparse_pts_3d_ids.append(_pts_3d_id)
                self.points_3D_recon.append(self.pts_3d_query[_pts_3d_id].xyz)
                self.lines_3D_recon.append(self.pts_to_line[_pts_3d_id])

                # Append centers
                self.pts_to_centers_recon.append(self.pts_to_center[_pts_3d_id])
                self.pts_to_center_indices_recon.append(self.pts_to_center_indices[_pts_3d_id])

                # Real 3D points
                self.id_to_ind_recon[_pts_3d_id] = i
                self.ind_to_id_recon[i] = _pts_3d_id

            # Convert to numpy array
            self.points_3D_recon = np.array(self.points_3D_recon)
            self.lines_3D_recon = np.array(self.lines_3D_recon)
            self.pts_to_centers_recon = np.array(self.pts_to_centers_recon)
            self.pts_to_center_indices_recon = np.array(self.pts_to_center_indices_recon)
            
            # Prevent trivial solutions
            if variable.USE_RAYS_FROM_OPPOSITE_CENTER:
                print('Using only 3D rays from opposite center to prevent trivial solutions! \n')

            # Apply line rejection
            rejected_line_indices = None
            if variable.LINE_REJECTION:
                rejected_line_indices = calculate.line_rejection_percentage(self.lines_3D_recon, threshold_ratio=variable.REJECTION_RATIO, center_pts=self.center_pts)

            # Coarse estimation 
            if variable.LINE_REJECTION: # With line-rejection
                ests, ests_pts = calculate.coarse_est_spf_raycloud(self.points_3D_recon, self.lines_3D_recon, self.pts_to_center_indices_recon, rejected_line_indices)
                info = [sparsity_level, noise_level, 0.0, estimator, self.num_clusters, f"{str(variable.REJECTION_RATIO)}rejectcoarse"]
            else: # Without line-rejection
                ests, ests_pts = calculate.coarse_est_spf_raycloud(self.points_3D_recon, self.lines_3D_recon, self.pts_to_center_indices_recon)
                info = [sparsity_level, noise_level, 0.0, estimator, self.num_clusters, "norejectcoarse"]
            super().saveReconpoints_raycloud(ests_pts, info)

    def reconTest(self, estimator):
        # reconTest
        recontest.recontest_pt_idx([self.points_3D_recon],[self.ind_to_id_recon],self.pts_3d_query)

    def test(self, recover, esttype):
        # recon test
        print("Consistency test for", self.map_type)
        if recover:
            self.reconTest(esttype)