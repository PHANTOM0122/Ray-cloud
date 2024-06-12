import os
import shutil
import numpy as np
import open3d as o3d

def filter_output(points3D,th_nn, th_ratio):
    pts = np.array([v.xyz for v in points3D.values()])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors = th_nn, std_ratio = th_ratio)
    pcd2 = pcd.select_by_index(ind)

    cl2, ind2 = pcd2.remove_statistical_outlier(nb_neighbors = th_nn, std_ratio = th_ratio)
    pcd3 = pcd2.select_by_index(ind2)
    
    pt_keys = np.array(list(points3D.keys()))
    removed_outlier_id = []
    for i in ind2:
        removed_outlier_id.append(pt_keys[ind[i]])
    
    return removed_outlier_id

def remove_outliers(points3D, q_img, th_nn, th_ratio):
    # Points3D TXT 
    print("Execute outlier removal")
    pts_id_wo_outliers = filter_output(points3D, th_nn,th_ratio)

    print(len(points3D)-len(pts_id_wo_outliers),"Number of outliered removed")

    points3D_wo_outliers={}
    for k in pts_id_wo_outliers:
        points3D_wo_outliers[k] = points3D[k]
        
    pts_id_withoutliers = np.array(list(points3D.keys()))
    outlier_ids = np.setdiff1d(pts_id_withoutliers,pts_id_wo_outliers)
    
    q_img_wo_outliers = q_img.copy()
    for k in q_img_wo_outliers.keys():
        ids = q_img_wo_outliers[k].point3D_ids
        xys = q_img_wo_outliers[k].xys
        outliers = [i for i in range(len(ids)) if ids[i] in outlier_ids]
        del_ids = np.delete(ids,outliers)
        del_xys = np.delete(xys,outliers,axis=0)
        q_img_wo_outliers[k] = q_img_wo_outliers[k]._replace(point3D_ids = del_ids, xys = del_xys)
        
    return points3D_wo_outliers, q_img_wo_outliers

def remove_overlap_pts(txt_points3d, txt_images):
    print("Overlapping points removal")
    # make key with sum of 3d point xyz value
    pts_key = {sum(pt.xyz):k for k,pt in txt_points3d.items()}
    pts_key_check1 = {pt.xyz[0]-pt.xyz[1]+pt.xyz[2]:k for k,pt in txt_points3d.items()}
    pts_key_check2 = {pt.xyz[0]+pt.xyz[1]-pt.xyz[2]:k for k,pt in txt_points3d.items()}
    pts_key_union = set(pts_key.values()).union(pts_key_check1.values()).union(pts_key_check2.values())
    non_overlapping_ids = np.array(list(pts_key_union))
    overlapping_ids = np.setdiff1d(np.array(list(txt_points3d.keys())),non_overlapping_ids)

    non_overlapping_txt = {k:txt_points3d[k] for k in non_overlapping_ids}
    non_overlapping_images = txt_images.copy()

    for k in non_overlapping_images.keys():
        ids = non_overlapping_images[k].point3D_ids
        xys = non_overlapping_images[k].xys
        overlap = [i for i in range(len(ids)) if ids[i] in overlapping_ids]
        del_ids = np.delete(ids,overlap)
        del_xys = np.delete(xys,overlap,axis=0)
        non_overlapping_images[k] = non_overlapping_images[k]._replace(point3D_ids = del_ids, xys = del_xys)
        
    print(f"{len(overlapping_ids)} points ovelapping")
    
    return non_overlapping_txt, non_overlapping_images