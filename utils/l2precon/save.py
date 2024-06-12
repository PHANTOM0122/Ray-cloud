import os
import sys
from collections import defaultdict
from utils.colmap import read_write_model

def write_colmap_points(fout_name, points3D, pts_est, if_use_pt, id_to_ind):
    points = write_colmap_points_one_pair(points3D, pts_est, if_use_pt, id_to_ind)
    read_write_model.write_points3D_text(points, fout_name)
    

def write_colmap_points_two_pair(fout_name, points3D, pts_est1, pts_est2, if_use_pt1, if_use_pt2, id_to_ind, id_to_ind2):
    points3D_1 = write_colmap_points_one_pair(points3D, pts_est1, if_use_pt1, id_to_ind)
    points3D_2 = write_colmap_points_one_pair(points3D, pts_est2, if_use_pt2, id_to_ind2)
    points = dict(points3D_1)
    points.update(points3D_2)
    read_write_model.write_points3D_text(points, fout_name)

def write_colmap_points_one_pair(points3D, pts_est, if_use_pt, id_to_ind):
    points = points3D.copy()
    for pid in points3D.keys():
        if if_use_pt[pid]:
            pind = id_to_ind[pid]
            points[pid] = points[pid]._replace(xyz=pts_est[pind])
        else:
            del points[pid]
    
    return points

def save_colmap_spf(fout_name, pts_est, id_to_ind, Points):
    # ind_to_id , id_to_ind -  larger set including setB
    if_use_pt = defaultdict(int)
    for i in id_to_ind.keys():
        if_use_pt[i] = True
        
    write_colmap_points(fout_name, Points, pts_est, if_use_pt, id_to_ind)
    
    return 0
        
def save_colmap_tpf(fout_name, pts_ests, id_to_inds, Points):
    pts_est1, pts_est2 = pts_ests
    id_to_ind1, id_to_ind2 = id_to_inds
    
    # ind_to_id , id_to_ind -  larger set including setB
    if_use_pt1 = defaultdict(int)
    if_use_pt2 = defaultdict(int)
    for i in id_to_ind1.keys():
        if_use_pt1[i] = True
    for i in id_to_ind2.keys():
        if_use_pt2[i] = True
            
    write_colmap_points_two_pair(fout_name, Points, pts_est1, pts_est2, if_use_pt1,if_use_pt2, id_to_ind1, id_to_ind2)
    
    return 0