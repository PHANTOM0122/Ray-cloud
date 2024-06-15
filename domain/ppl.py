from .master import Master
import numpy as np
import poselib
from collections import defaultdict
import time

import test_module.linecloud as lineCloudTest
import test_module.recontest as recontest

import utils.pose.pose_estimation as pe
import utils.pose.vector as vector
from utils.pose import dataset
from utils.pose import line
from utils.l2precon import calculate
from static import variable

np.random.seed(variable.RANDOM_SEED)

class PPL(Master):
    def __init__(self, dataset_path, output_path):
        self.pts_to_line = dict()
        self.line_to_pts = defaultdict(list)
        self.points_3D = None
        self.line_3d = None

        self.pts_2d_query = None
        self.pts_3d_query = None
        self.camera_dict_gt = None

        self.ind_to_id = None
        self.id_to_ind = None
        self.queryIds = None
        self.queryNames = None
        self.image_dict_gt = None

        self.resultPose = list()
        self.resultRecon = list()
        self.map_type = "PPL"
        self.points_3D_recon = list()
        self.lines_3D_recon = list()

        super().__init__(dataset_path, output_path)

        self.pts_3d_ids = list(self.pts_3d_query.keys())
        np.random.shuffle(self.pts_3d_ids)

    def makeLineCloud(self):
        print("PPL: Two point line cloud")
        _pts_3d = np.array([v.xyz for v in self.pts_3d_query.values()])
        _pts_ids = np.array([k for k in self.pts_3d_query.keys()])

        self.points_3D, self.line_3d, self.ind_to_id, self.id_to_ind = line.drawlines_ppl(_pts_3d, _pts_ids)

        super().mapPointtoPPL()


    def maskSparsity(self, sparisty_level):
        new_shape = int(len(list(self.line_to_pts.keys())) * sparisty_level)
        self.sparse_line_3d_ids = list(self.line_to_pts.keys())[:new_shape]


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

        newIndex = []
        _x2 = []
        # Truncate using Two Point Line
        for _nk in self.sparse_line_3d_ids:
            _p1, _p2 = self.line_to_pts[_nk]

            if _p1 in pts_to_ind:
                newIndex.append(pts_to_ind[_p1])
                _x2.append(self.pts_to_line[_p1])
                
            if _p2 in pts_to_ind:
                # Both 3D Points Should be Visible from the Query Image
                newIndex.append(pts_to_ind[_p2])
                _x2.append(self.pts_to_line[_p2])

        if newIndex:
            newIndex = np.array(newIndex)
            # p1: 2D Point
            # x1: 2D Line 
            # p2: 3D Offset
            # x2: 3D Line
            self._x1 = x1[newIndex]
            self._p2 = p2[newIndex]
            self._x2 = np.array(_x2)
                    
        else:
            self._x1 = np.array([])
            self._p2 = np.array([])
            self._x2 = np.array([])

        print("Found correspondences: ", self._x1.shape[0])


    def addNoise(self, noise_level):
        super().addNoise(noise_level)


    def estimatePose(self, query_id):
        if self._x1.shape[0] >= 6:
            gt_img = pe.get_GT_image(query_id, self.pts_2d_query, self.image_dict_gt)
            cam_id = gt_img.camera_id
            cam_p6l = [pe.convert_cam(self.camera_dict_gt[cam_id])]

            start = time.time()
            res = poselib.estimate_p6l_relative_pose(self._x1, self._p2, self._x2, cam_p6l, cam_p6l, variable.RANSAC_OPTIONS, variable.BUNDLE_OPTIONS, variable.REFINE_OPTION)
            end = time.time()
            super().savePoseAccuracy(res, gt_img, cam_p6l[0],end-start)
        else:
            print("TOO sparse point cloud")


    def savePose(self, sparisty_level, noise_level):
        super().savePose(sparisty_level, noise_level)


    def saveAllPoseCSV(self):
        super().saveAllPoseCSV()


    def recoverPts(self, estimator, sparsity_level, noise_level):
        print("PPL recover image", "\n")
        super().recoverPPLbase(estimator, sparsity_level, noise_level)
        
    def reconTest(self,estimator):
        # Check recon index id match
        if estimator=='SPF':
            pass
        if estimator=='TPF':
            recontest.recontest_pt_idx(self.points_3D_recon,self.ind_to_id_recon,self.pts_3d_query)
            recontest.compare_LPtest_PPLbase(self.points_3D_recon,self.lines_3D_recon)

    def test(self,recover,esttype):
        # Check line point match
        super().test(recover,esttype)
