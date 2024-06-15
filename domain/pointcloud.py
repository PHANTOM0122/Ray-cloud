from .master import Master
import numpy as np
import poselib
import os
import math
import time

import utils.pose.pose_estimation as pe
import utils.pose.vector as vector
from utils.pose import dataset
from utils.colmap import read_write_model
from static import variable

np.random.seed(variable.RANDOM_SEED)

class PointCloud(Master):
    def __init__(self, dataset_path, output_path):
        self.pts_to_line = dict()
        self.line_to_pts = dict()
        self.line_3d = None

        self.pts_2d_query = None # Images.txt
        self.pts_3d_query = None # Points3D.txt
        self.camera_dict_gt = None # cameras.txt

        self.queryIds = None
        self.queryNames = None
        self.image_dict_gt = None

        self.resultPose = list()
        self.map_type = "PC"

        super().__init__(dataset_path, output_path)

        self.pts_3d_ids = list(self.pts_3d_query.keys())
        np.random.shuffle(self.pts_3d_ids)

    def makeLineCloud(self):
        print("Point Cloud")
        pass

    def maskSparsity(self, sparisty_level):
        new_shape = int(len(self.pts_3d_ids) * sparisty_level)
        self.sparse_pts_3d_ids = set(self.pts_3d_ids[:new_shape])

    def matchCorrespondences(self, query_id):
        connected_pts3d_idx = np.where(self.pts_2d_query[query_id].point3D_ids != -1)[0]
        connected_pts3d_ids = self.pts_2d_query[query_id].point3D_ids[connected_pts3d_idx]

        p2 = np.array([self.pts_3d_query[k].xyz for k in connected_pts3d_ids],dtype=np.float64)
        x1 = np.array(self.pts_2d_query[query_id].xys[connected_pts3d_idx],dtype=np.float64)

        pts_to_ind = {}
        for _i, k in enumerate(connected_pts3d_ids):
            pts_to_ind[k] = _i
            if self.pts_3d_query[k].xyz[0] != p2[_i][0]:
                raise Exception("Point to Index Match Error ", k)

        self.valid_pts_3d_ids = self.sparse_pts_3d_ids.intersection(set(connected_pts3d_ids))

        newIndex = []
        for _pid in self.valid_pts_3d_ids:
            newIndex.append(pts_to_ind[_pid])

        if newIndex:
            newIndex = np.array(newIndex)
            self._x1 = x1[newIndex]
            self._p2 = p2[newIndex]

        else:
            self._x1 = np.array([])
            self._p2 = np.array([])

        print("Found correspondences: ", self._x1.shape[0])


    def addNoise(self, noise_level):
        super().addNoise(noise_level)


    def estimatePose(self, query_id):
        if self._x1.shape[0] >= 3:
            gt_img = pe.get_GT_image(query_id, self.pts_2d_query, self.image_dict_gt)
            cam_id = gt_img.camera_id
            cam_p3p = pe.convert_cam(self.camera_dict_gt[cam_id])

            start = time.time()
            res = poselib.estimate_absolute_pose(self._x1, self._p2, cam_p3p, variable.RANSAC_OPTIONS, variable.BUNDLE_OPTIONS, variable.REFINE_OPTION)
            end = time.time()
            super().savePoseAccuracy(res, gt_img, cam_p3p, end-start)
        else:
            print("TOO sparse point cloud")

    
    def savePose(self, sparisty_level, noise_level):
        super().savePose(sparisty_level, noise_level)


    def saveAllPoseCSV(self):
        super().saveAllPoseCSV()


    def recoverPts(self,estimator, sparsity_level,noise_level):
        recon_output_path = os.path.join(self.output_path, "L2Precon")
        os.makedirs(recon_output_path, exist_ok=True)
        filename = "_".join([self.map_type,self.dataset,'noest','sp'+str(sparsity_level),'n'+str(noise_level),'sw0.0']) + ".txt"
        
        fout = os.path.join(recon_output_path,filename)
        new_pts={}
        for id in self.sparse_pts_3d_ids:
            new_pts[id]= self.pts_3d_query[id]
        read_write_model.write_points3D_text(new_pts,fout)
    
    def reconTest(self,estimator):
        pass


    def test(sel,recover,esttype):
        pass
