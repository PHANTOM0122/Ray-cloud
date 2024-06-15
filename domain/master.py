from abc import *
import time
import math
from collections import defaultdict
import numpy as np
import pandas as pd

from static import variable as VAR

import test_module.linecloud as lineCloudTest
import test_module.recontest as recontest

from utils.colmap.read_write_model import *
import utils.pose.pose_estimation as pe
import utils.pose.vector as vector
from utils.pose import dataset
from utils.pose import line
from utils.l2precon import save
from utils.l2precon import calculate
from utils.invsfm import reconstruction
from utils.remove_outliers import remove_outliers, remove_overlap_pts

from domain import *
np.random.seed(VAR.RANDOM_SEED)

class Master(metaclass=ABCMeta):
    """
     Load 3D Points/Lines from Colmap dataset.

     @param {string} relative path from main.py to colmap dataset (points3d.txt, images.txt, ...)
     @param {list[Point 2D]} 3D points.
     @param {list[Point 3D]} 2D points.
     @param {list[int]} query image ids.
     @param {dict} GT image dictionary. (GT pose)
     @param {dict} GT camera dictionary. 
    """

    def __init__(self, cur_dir, dataset):
        self.dataset = dataset
        self.curdir = cur_dir
        dataset_path = os.path.join(cur_dir, VAR.DATASET_MOUNT, VAR.getDatasetName(self.dataset), self.dataset)
        self.basepath = dataset_path
        self.output_path = os.path.join(cur_dir, "output", VAR.getDatasetName(self.dataset), self.dataset)

        self.pts_2d_query = read_images_text(os.path.join(dataset_path, "sparse_queryadded", "images.txt"))
        self.pts_3d_query = read_points3D_text(os.path.join(dataset_path, "sparse_queryadded","points3D.txt"))

        self.camera_dict_gt = read_cameras_text(os.path.join(dataset_path, "sparse_gt", "cameras.txt"))
        self.image_dict_gt = read_images_text(os.path.join(dataset_path, "sparse_gt", "images.txt"))

        query_txt_path = os.path.join(dataset_path, "query_imgs.txt")
        
        self.pts_3d_query, self.pts_2d_query = remove_outliers(self.pts_3d_query,self.pts_2d_query, VAR.THR_OUT_NN,VAR.THR_OUT_STD)
        self.pts_3d_query, self.pts_2d_query = remove_overlap_pts(self.pts_3d_query, self.pts_2d_query)

        self.queryNames, self.queryIds = pe.get_query_images(query_txt_path, self.pts_2d_query)

        self.scale = VAR.getScale(self.dataset)
        self.files = []
        self.checkedfiles = []
            
        print("Load dataset sample for debugging")
        print("--- Query Image ----")
        print(list(self.pts_2d_query.values())[0], "\n")
        print("--- Points 3D ----")
        print(list(self.pts_3d_query.values())[0], "\n")
        print("--- GT Image ----")
        print(list(self.image_dict_gt.values())[0], "\n")
        print("--- Images ----")
        print(list(self.camera_dict_gt.values())[0], "\n")
        print("--- Query Image IDS ----")
        print(self.queryIds[:3], "\n")
        print("Dataset loaded successfully", "\n")


    """
     List point cloud to line cloud.
     Construct point to line, line to point dictionary

     Variable.py: (PC, OLC, PPL, PPL+).
    """
    @abstractmethod
    def makeLineCloud(self):
        pass

    """
     Remove correspondences partially w.r.t. 3D line. 

     @param {string} sparsity level (list[float]).
    """
    @abstractmethod
    def maskSparsity(self, sparisty_level):
        pass


    """
     Match correspondences between 2D x 3D

     @param {int} query image ID.
    """
    @abstractmethod
    def matchCorrespondences(self, query_id):
        pass


    """
     Add noise to 2D x 3D points. 
    """
    @abstractmethod
    def addNoise(self, noise_level):
        self._x1 += np.random.rand(*self._x1.shape)*noise_level
        self._p2 += np.random.rand(*self._p2.shape)*noise_level


    """
     Estimate pose accuracy, and save R, t, time, errors

    """
    @abstractmethod
    def estimatePose(self):
        pass


    """     
     @param {dict} Pose etimation result.
     @param {dict} GT pose dictionary.
     @param {dict} GT cam dictionary.
     @param {string} output path.
    """
    def savePoseAccuracy(self, res, gt, cam, time):
        error_r, error_t = vector.calculate_loss(gt, res)
        
        # Convert to Degree
        error_r = math.degrees(error_r)
        
        # Get RANSAC results
        iterations = res[1]['iterations']
        inlier_ratio = res[1]['inlier_ratio']
        
        # Append localization results
        self.resultPose.append([res[0].q, res[0].t, error_r, error_t, time, iterations, inlier_ratio])

    """
    Save R, t, time, errors to specified output path
     --------------Output Format -------------
     query1: R | t | error(R) | error(t) | time | iterations | inlier ratio |
     query2: R | t | error(R) | error(t) | time | iterations | inlier ratio |
     ... 
     -----------------------------------------
    """
    @abstractmethod
    def savePose(self, sparsity_level, noise_level, solver=None, num_cluster=2):
        if VAR.REFINE_OPTION:
            pose_output_path = os.path.join(self.output_path, "PoseAccuracy","refined")
        else:    
            pose_output_path = os.path.join(self.output_path, "PoseAccuracy","notrefined")
        os.makedirs(pose_output_path, exist_ok=True)

        print("Saving pose estimation result.", self.dataset, "Sparsity: ", sparsity_level, " Noise: ", noise_level)
        # example filename: OLC_gerrard_hall_SPF_sp0.05_n0.0_sw0.txt
        filename = "_".join([self.map_type, self.dataset, "NA", "sp"+str(sparsity_level), "n"+str(noise_level), "sw0"]) + ".txt"
        if self.map_type.lower().startswith('raycloud'):
            filename = "_".join([self.map_type, solver, self.dataset, "NA", "sp" + str(sparsity_level), "n" + str(noise_level), "sw0"]) + ".txt"
        if self.map_type.lower() == 'raycloudvoxel':
            filename = "_".join([self.map_type, str(VAR.VOXEL_SIZE),'meters', solver, self.dataset, "NA", "sp" + str(sparsity_level), "n" + str(noise_level), "sw0"]) + ".txt"
            
        mean_r_error = -1
        median_r_error = -1
        mean_t_error = -1
        median_t_error = -1
        mean_time = -1
        mean_iter = -1
        mean_inliers = -1
        if self.resultPose:
            r_error_lst=np.empty(len(self.resultPose))
            t_error_lst=np.empty(len(self.resultPose))
            time_lst=np.empty(len(self.resultPose))
            iter_lst=np.empty(len(self.resultPose))
            inlier_lst=np.empty(len(self.resultPose))
            for i in range(len(self.resultPose)):
                r_error_lst[i] = self.resultPose[i][2]
                t_error_lst[i] = self.resultPose[i][3]
                time_lst[i] = self.resultPose[i][4]
                iter_lst[i] = self.resultPose[i][5]
                inlier_lst[i] = self.resultPose[i][6]
            mean_r_error = np.mean(r_error_lst)
            median_r_error = np.median(r_error_lst)
            mean_t_error = np.mean(t_error_lst) * self.scale
            median_t_error = np.median(t_error_lst) * self.scale
            mean_time = np.mean(time_lst)
            mean_iter = np.mean(iter_lst)
            mean_inliers = np.mean(inlier_lst)
        
        print("Scene scale:", self.scale)
        print("Final Mean Error R:",mean_r_error)
        print("Final Mean Error T:",mean_t_error)
        print("Final Mean Pose Inference time:",mean_time)
        print("Final Mean Iterations:",mean_iter)
        print("Final Mean inlier ratio:",mean_inliers)

        with open(os.path.join(pose_output_path, filename), "w") as f:
            f.write(f"R_Mean:{mean_r_error}, t_Mean:{mean_t_error}, R_Median:{median_r_error}, t_Meidan:{median_t_error} " + "\n")
            f.write(f"Mean_time:{mean_time}"+"\n")
            f.write(f"Mean_iteration:{mean_iter}"+"\n")
            f.write(f"Mean_inlier_ratio:{mean_inliers}"+"\n")
            f.write(f"Effective Queries: {len(self.resultPose)}/{len(self.queryIds)}" + "\n")
            f.write("qvec(4) tvec(3) error_r(1) error_t(1) time(1) iterations(1) inlier_ratio(1) solutions(1)" + "\n")
            for q, t, e_r, e_t, sec, _iter, _inlier in self.resultPose:
                q = " ".join(list(map(str, (q.tolist())))) + " "
                t = " ".join(list(map(str, (t.tolist())))) + " "
                _r = " ".join(list(map(str, ([e_r, (e_t * self.scale), sec, _iter, _inlier]))))
                f.write(q + t + _r + "\n")
                
        # Reset List
        self.resultPose = list()

    """
    Save R, t, time, errors to a single csv file
     --------------Output Format -------------
                    sparsity1 noise1                         sparsity2  noise2                   
     query2: R | t | error(R) | error(t) | time    query1: R | t | error(R) | error(t) | time
     query2: R | t | error(R) | error(t) | time    query2: R | t | error(R) | error(t) | time
     -----------------------------------------
    """
    @abstractmethod
    def saveAllPoseCSV(self):
        pose_output_path = os.path.join(self.output_path, "PoseAccuracy")

        n_query = len(self.queryIds)
        sparisty_noise_level_variations = len(VAR.SPARSITY_LEVEL) * len(VAR.NOISE_LEVEL)

        csv_result = np.zeros((n_query, 5 * sparisty_noise_level_variations), dtype=object)
        for _j in range(sparisty_noise_level_variations):
            for _i in range(n_query):
                q, t, e_r, e_t, sec = self.resultPose[_i + _j*n_query]
                csv_result[_i][_j*5] = ", ".join(list(map(str, (q.tolist()))))
                csv_result[_i][_j*5+1] = ", ".join(list(map(str, (t.tolist()))))
                csv_result[_i][_j*5+2] = e_r
                csv_result[_i][_j*5+3] = (e_t * self.scale)
                csv_result[_i][_j*5+4] = sec

        csv_columns = []
        for _i in range(len(VAR.SPARSITY_LEVEL)):
            for _j in range(len(VAR.NOISE_LEVEL)):
                _sl = VAR.SPARSITY_LEVEL[_i]
                _nl = VAR.NOISE_LEVEL[_j]
                csv_columns.append("Sparsity: " + str(_sl) + " Noise : " + str(_nl) + " qvec")
                csv_columns += ["tvec", "error_r", "error_t", "time"]

        _res_csv = pd.DataFrame(csv_result, columns=csv_columns)
        _fname = self.dataset + "_" + self.map_type + ".csv"
        _res_csv.to_csv(os.path.join(pose_output_path, _fname))


    """
     Create 2D points - 3D line mapper dictionary.
     The dictionary is used when constructing sparse line cloud
    """
    def mapPointtoPPL(self):
        line_val_to_int = defaultdict(lambda :1e9)

        half_size = len(self.points_3D[0])
        for idx_to_id in self.ind_to_id:
            for i, k in idx_to_id.items():
                self.pts_to_line[k] = self.line_3d[i]

                hashKey = line.getHash(self.line_3d[i])
                # Line Value : Index Integer 
                line_val_to_int[hashKey] = min(i, line_val_to_int[hashKey])

        for k, v in self.pts_to_line.items():
            self.line_to_pts[line_val_to_int[line.getHash(v)]].append(k)

        cnt_odd = 0
        cnt_left_over = 0
        remove_key = []
        add_key = []
        for k, v in self.line_to_pts.items():
            if len(v) == 1:
                # 처음 발생 -> Point 홀수 사용
                if cnt_odd == 0:
                    cnt_odd += 1
                    remove_key.append(k)
                    continue

                else:
                    raise Exception("Line With One Point Mapped ", len(v))
                    exit(1)

            if len(v) > 2:
                remove_key.append(k)

                if np.isclose(np.linalg.norm(self.pts_to_line[v[0]]), 0):
                    print("Skipping Line With Zero Vector")
                    continue

                # Check If Pairs are of identical points
                if len(v) % 2 == 0:
                    _temp = defaultdict(list)
                    pivot = line.getHash(self.pts_3d_query[v[0]].xyz)
                    _temp[pivot].append(v[0])
                    for i in range(1, len(v)):
                        cur = line.getHash(self.pts_3d_query[v[i]].xyz)
                        _temp[cur].append(v[i])

                    if len(_temp) != 2:
                        cnt_left_over += 1
                        continue
                    
                    _k1, _k2 = _temp.keys()
                    for _ii in range(len(_temp[_k1])):
                        add_key.append([_temp[_k1][_ii], _temp[_k2][_ii]])
                    
                    continue

                else:
                    raise Exception("Line with More than Two Points Mapped ", len(v))
                    exit(1)
            
        for r_key in remove_key:
            self.line_to_pts.pop(r_key)
            

    """
    Save recovered 3D point from line cloud using specified estimator.
    Swap features, modify sparsity.

    @param {list} estimated Points (3D Points).
    @param {list} info (Estimator type, Noise, Sparsity, Swap levels).
    
    """
    def saveReconpoints(self, estimatedPoints, info):
        sparsity_level,noise_level,_,estimate_type = info
        
        recon_output_path = os.path.join(self.output_path, "L2Precon")
        os.makedirs(recon_output_path, exist_ok=True)

        print(f"Saving L2P reconstruction result for {self.dataset} \n\
              with Sparsity: {sparsity_level}, Feature swap: No swap, Noise: {noise_level}")
        filename = "_".join([self.map_type,self.dataset,estimate_type,'sp'+str(sparsity_level),'n'+str(noise_level),'sw0.0']) + ".txt"
        
        # X list
        fout = os.path.join(recon_output_path,filename)
        save.save_colmap_spf(fout,estimatedPoints,self.id_to_ind_recon,self.pts_3d_query)


    def saveReconpointswithswap(self,estimatedPoints,info):
        sparsity_level,noise_level,swap_levels,estimate_type = info
        
        recon_output_path = os.path.join(self.output_path, "L2Precon")
        os.makedirs(recon_output_path, exist_ok=True)

        # list
        for i,swap_level in enumerate(swap_levels):
            print(f"Saving L2P reconstruction result for {self.dataset} \
                \n with Sparsity: {sparsity_level}, Feature swap: {swap_level}, Noise: {noise_level}")
            filename = "_".join([self.map_type, self.dataset, estimate_type,'sp'+str(sparsity_level),'n'+str(noise_level),'sw'+str(swap_level)]) + ".txt"
            fout = os.path.join(recon_output_path,filename)
            if estimate_type=='SPF':
                save.save_colmap_spf(fout,estimatedPoints,self.id_to_ind_recon[0][i],self.pts_3d_query)
            elif estimate_type=='TPF':
                save.save_colmap_tpf(fout,estimatedPoints[i],self.id_to_ind_recon,self.pts_3d_query)
                
    def saveReconpoints_raycloud(self, estimatedPoints, info):
        
        sparsity_level, noise_level, _, estimate_type, num_clusters, option = info
        filename = "_".join([self.map_type, str(num_clusters), 'clusters', self.dataset, estimate_type,'sp' + str(sparsity_level), 'n' + str(noise_level), 'sw0.0_' + option]) + ".txt"
        
        # if option.endswith("coarse"):
        #     filename = "_".join([self.map_type, str(num_clusters), 'clusters', self.dataset, estimate_type,'sp' + str(sparsity_level), 'n' + str(noise_level), 'sw0.0_' + option]) + ".txt"
        # elif option.endswith("refine"):
        #     filename = "_".join([self.map_type, str(num_clusters), 'clusters', self.dataset, estimate_type, 'sp' + str(sparsity_level), 'n' + str(noise_level), 'sw0.0_' + option]) + ".txt"
        
        recon_output_path = os.path.join(self.output_path, "L2Precon")
        os.makedirs(recon_output_path, exist_ok=True)

        print(f"Saving L2P reconstruction result for {self.dataset} \n\
              with Sparsity: {sparsity_level}, Feature swap: No swap, Noise: {noise_level}")
        print(f'Saved at {filename}')

        # X list
        fout = os.path.join(recon_output_path,filename)
        save.save_colmap_spf(fout,estimatedPoints,self.id_to_ind_recon,self.pts_3d_query)

    """
     Recover 3D points from line cloud using specified estimator.
     Swap features in case of PPL/PPL+.

     @param {string} estimator (SPF, TPF).
    """
    @abstractmethod
    def recoverPts(self):
        pass
    
    def recoverPPLbase(self,estimator, sparsity_level, noise_level):
        pts_A = []
        pts_B = []
        self.ind_to_id_recon = [{},{}]
        self.id_to_ind_recon = [{},{}]
        self.sparse_pts_3d_ids=[]
        self.points_3D_recon = []
        self.lines_3D_recon = []
        for i,_lk in enumerate(self.sparse_line_3d_ids):
            _pts_3d_id1, _pts_3d_id2 = self.line_to_pts[_lk]
            self.sparse_pts_3d_ids.extend([_pts_3d_id1,_pts_3d_id2])
            self.lines_3D_recon.append(self.pts_to_line[_pts_3d_id1])
            pts_A.append(self.pts_3d_query[_pts_3d_id1].xyz)
            pts_B.append(self.pts_3d_query[_pts_3d_id2].xyz)
            self.id_to_ind_recon[0][_pts_3d_id1] = i
            self.ind_to_id_recon[0][i] = _pts_3d_id1
            self.id_to_ind_recon[1][_pts_3d_id2] = i
            self.ind_to_id_recon[1][i] = _pts_3d_id2
        
        pts_A = np.array(pts_A)
        pts_B = np.array(pts_B)
        self.lines_3D_recon = np.array(self.lines_3D_recon)
        
        self.points_3D_recon.extend([pts_A, pts_B])
        print("Total recon line",len(self.lines_3D_recon))
        
        swap_level = VAR.SWAP_RATIO
        ref_iter = VAR.REFINE_ITER
        
        if estimator=='SPF':
            est = calculate.coarse_est_spf(self.points_3D_recon[0], self.lines_3D_recon)
            ests_pts, self.ind_to_id_recon, self.id_to_ind_recon = calculate.refine_est_spf_harsh(self.points_3D_recon, self.lines_3D_recon, self.ind_to_id_recon, swap_level, est, ref_iter)
        if estimator=='TPF':
            ests_pts = calculate.coarse_est_tpf(self.points_3D_recon, self.lines_3D_recon, swap_level)
            
        info = [sparsity_level, noise_level, swap_level, estimator]
        self.saveReconpointswithswap(ests_pts,info)
    
    """
     Test line & point matches.
     Test point id & index matches.

    """
    @abstractmethod
    def reconTest(self,estimator):
        pass

    @abstractmethod
    def test(self,recover,esttype):
        lineCloudTest.line_integrity_test(self.line_to_pts, self.pts_to_line, self.line_3d, self.pts_3d_query)
        if recover:
            self.reconTest(esttype)
    
    
    def append_filenames(self,sparsity_level,noise_level,esttype,swap_level):
        # For ray cloud variants
        if self.map_type.startswith('RayCloud'):
            if VAR.LINE_REJECTION: # With rejection
                if self.map_type == 'RayCloud':
                    # Coarse results
                    filename_coarse = "_".join(
                        [self.map_type, str(self.num_clusters), 'clusters', self.dataset, esttype, 'sp' + str(sparsity_level), 'n' + str(noise_level),
                         'sw' + str(swap_level), str(VAR.REJECTION_RATIO)+'rejectcoarse']) + ".txt"
                    print(f'File name coarse appended: {filename_coarse}')
                    self.files.append(filename_coarse)
                    
                    # Refined results
                    filename_refine = "_".join(
                        [self.map_type, str(self.num_clusters), 'clusters', self.dataset, esttype, 'sp' + str(sparsity_level), 'n' + str(noise_level),
                         'sw' + str(swap_level), str(VAR.REJECTION_RATIO)+'rejectrefine']) + ".txt"
                    print(f'File name refine appended: {filename_refine}')
                    self.files.append(filename_refine)

                elif self.map_type == 'RayCloudVoxel':
                    # Coarse results
                    filename_coarse = "_".join(
                        [self.map_type, str(self.num_clusters), 'clusters', self.dataset, esttype, 'sp' + str(sparsity_level),
                         'n' + str(noise_level), 'sw' + str(swap_level), str(VAR.VOXEL_SIZE)+'mvoxel',  str(VAR.REJECTION_RATIO)+'rejectcoarse']) + ".txt"
                    print(f'File name coarse appended: {filename_coarse}')
                    self.files.append(filename_coarse)

                    # Refined results
                    filename_refine = "_".join(
                        [self.map_type, str(self.num_clusters), 'clusters', self.dataset, esttype, 'sp' + str(sparsity_level),
                         'n' + str(noise_level), 'sw' + str(swap_level), str(VAR.VOXEL_SIZE)+'mvoxel',  str(VAR.REJECTION_RATIO)+'rejectrefine']) + ".txt"
                    print(f'File name refine appended: {filename_refine}')
                    self.files.append(filename_refine)
                    
            else: # No line rejection
                 # Coarse results
                filename_coarse = "_".join([self.map_type, str(self.num_clusters), 'clusters', self.dataset, esttype, 'sp' + str(sparsity_level), 'n' + str(noise_level), 'sw' + str(swap_level), 'norejectcoarse']) + ".txt"
                print(f'File name coarse appended: {filename_coarse}')
                self.files.append(filename_coarse)
                
                # Refined results
                filename_refine = "_".join([self.map_type, str(self.num_clusters), 'clusters', self.dataset, esttype, 'sp' + str(sparsity_level), 'n' + str(noise_level), 'sw' + str(swap_level), 'norejectrefine']) + ".txt"
                print(f'File name refine appended: {filename_refine}')
                self.files.append(filename_refine)
                
        else: 
            filename = "_".join([self.map_type, self.dataset, esttype,'sp'+str(sparsity_level),'n'+str(noise_level),'sw'+str(swap_level)]) + ".txt"
            print(f'File name appended: {filename}')
            self.files.append(filename)
    
    
    def checkexists(self):
        reconpts_path = os.path.join(self.output_path,'L2Precon')
        existing_files = os.listdir(reconpts_path)
        for file in self.files:
            if file in existing_files:
                self.checkedfiles.append(file)
        
    def reconImg(self,device):
        # Recover scene images from reconstructed 3D point cloud
        print("Inversion process starts.")
        self.output_path = os.path.join(self.curdir, "output", VAR.getDatasetName(self.dataset), self.dataset)
        recon_path = os.path.join(self.output_path,"L2Precon")
        inv_q_path = os.path.join(self.output_path,"Quality")
        vars = [VAR.INPUT_ATTR,VAR.SCALE_SIZE,VAR.CROP_SIZE,VAR.SAMPLE_SIZE,device]
        reconstruction.invsfm(self.checkedfiles,self.basepath,recon_path,inv_q_path,vars, self.output_path)
