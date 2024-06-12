import numpy as np
import os
import sys
import open3d as o3d

from tqdm import tqdm
import torch
import static.variable as VAR

device = torch.device(VAR.CUDA)
np.random.seed(VAR.RANDOM_SEED)

def getHash(lst):
    s = 0
    for l in lst:
        s += hash(l)
    return s

def id_ind_connect(pair_ind,ids):    
    ind_to_id=[]
    id_to_ind=[]
    for pind in pair_ind:
        ind_to_id_tmp = {}
        id_to_ind_tmp = {}
        for j,pt_id in enumerate(ids[pind]):
            ind_to_id_tmp[j] = pt_id
            id_to_ind_tmp[pt_id] = j
        ind_to_id.append(ind_to_id_tmp)
        id_to_ind.append(id_to_ind_tmp)
        
    return ind_to_id, id_to_ind

def drawlines_olc(pts,ids):
    num = len(pts)
    np.random.seed(91)
    lines = np.random.randn(num,3)
    lines /= np.linalg.norm(lines, axis=1, keepdims=True)+1e-7
    pair_ind =  np.arange(num)
    ind_to_id, id_to_ind = id_ind_connect([pair_ind],ids)
    
    return [pts], lines, ind_to_id, id_to_ind

def drawlines_tp_two_sets(pts,pre_ind): # lines from two peaks (PPL)
    if len(pts)%2==0:
        pass
    else:
        pts = pts[:-1]
    
    permut_ind = np.random.permutation(len(pts))
    pair_index = pre_ind[permut_ind]
    point_pair = pts[permut_ind].reshape(-1,2,3)
    tmplines = np.subtract(point_pair[:,1,:],point_pair[:,0,:]).reshape(-1,3)
    tmplines /= np.linalg.norm(tmplines,axis=1,keepdims=True) + 1e-7
    
    points_tp_a = point_pair[:,0,:].reshape(-1,3)
    points_tp_b = point_pair[:,1,:].reshape(-1,3)
    lines_tp = tmplines
    ind_tp_a = pair_index[::2]
    ind_tp_b = pair_index[1::2]
    
    return [points_tp_a,points_tp_b], lines_tp,[ind_tp_a,ind_tp_b]

def drawlines_ppl(pts3d,ids):
    # pts - list [ptsA, ptsB], pair_ind - list [pair_indA, pair_indB]
    pts, lines, pair_ind = drawlines_tp_two_sets(pts3d,np.arange(len(pts3d)))
    ind_to_id, id_to_ind = id_ind_connect(pair_ind,ids)
    
    return pts, lines, ind_to_id, id_to_ind

def drawlines_pplplus(pts3d,ids,THR_LOOP=1000, THR_ANGLE=20):
    pts, lines, pair_ind = drawlines_tp_reject_plane(pts3d,THR_LOOP, THR_ANGLE)
    
    ind_to_id, id_to_ind = id_ind_connect(pair_ind,ids)

    return pts, lines, ind_to_id, id_to_ind

def drawlines_rayclouds(pts3d, pts_ids, center_points):
    
    '''
    pts3d: 3D points
    pts_ids: 3D point ids
    center_points: center points
    '''
    np.random.seed(VAR.RANDOM_SEED)
    
    num = len(pts3d)
    lines = [] # Unit direction vector of rays
    center_pts = [] # Save paired center points of rays
    lines_center_indices = [] # Pairing arrangement between ray(3Dpoint) and ray center

    for i, pt in enumerate(pts3d):
        
        # Randomly select index of center point per 3D point
        arr = np.arange(len(center_points))
        center_point_idx = np.random.choice(arr)
        center_point = center_points[center_point_idx] # Randomly select ray center
        
        # For the ease of the point reconstruction, we set direction from 3D point -> virtual camera center direction.
        # The line directions will be inversed during localization.
        line = np.subtract(center_point, pt)
        lines.append(line)
        center_pts.append(center_point)
        lines_center_indices.append(center_point_idx)

    lines = np.array(lines, dtype=np.float64).reshape(-1,3)
    lines /= (np.linalg.norm(lines, axis=1, keepdims=True)) 

    center_pts = np.asarray(center_pts, dtype=np.float64).reshape(-1,3)
    pair_ind = np.arange(num)
    ind_to_id, id_to_ind = id_ind_connect([pair_ind], pts_ids)

    return pts3d, lines, ind_to_id, id_to_ind, center_pts, np.array(lines_center_indices)

def drawlines_ray_clouds_voxelize(pts3d, pts_ids, center_points, voxel_size):
    '''
    pts3d: 3D points
    pts_ids: 3D point ids
    center_points: center points
    voxel_size: size of voxels
    '''
    
    num = len(pts3d)
    lines = []
    center_pts = []
    lines_center_indices = []

    # Voxelize
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts3d))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    print('Number of voxels:', voxel_grid)

    for i, pt in enumerate(pts3d):

        # Select index of center point according to voxel_id
        # get_voxel(query pt) -> Returns voxel index given query point.
        center_point_idx = voxel_3d_to_1d(np.asarray(voxel_grid.get_voxel(pt))) % 2
        center_point = center_points[center_point_idx]

        # For the ease of the point reconstruction, we set direction from 3D point -> virtual camera center direction.
        # The line directions will be inversed during localization.
        line = np.subtract(center_point, pt)

        lines.append(line)
        center_pts.append(center_point)
        lines_center_indices.append(center_point_idx)

    lines = np.array(lines, dtype=np.float64).reshape(-1, 3)
    lines /= (np.linalg.norm(lines, axis=1, keepdims=True)) 

    center_pts = np.asarray(center_pts, dtype=np.float64).reshape(-1, 3)
    pair_ind = np.arange(num)
    ind_to_id, id_to_ind = id_ind_connect([pair_ind], pts_ids)

    return pts3d, lines, ind_to_id, id_to_ind, center_pts, np.array(lines_center_indices)

def get_vec_from_nn_torch(pt,pts,num_nn):
    dist = torch.norm(pts - pt,dim=1)
    _, ii_nn = torch.topk(dist, num_nn + 1, largest=False)
    nn_idx = ii_nn[1:num_nn+1]
    vec = pts[nn_idx]-pt
    return vec

def compare_normal_svd(lines,compare_rq):
    nn_vec,s_nn_vec,num_nn_p2p,thre_ang = compare_rq
    
    # cross over NN (must not include line)
    nn_normal_vec= np.cross(s_nn_vec, nn_vec) # shape : (num_pts//2, num_nn_p2p, 3)
    normal_angle = np.zeros((lines.shape[0]))
    for i,nnv in enumerate(nn_normal_vec):
        U, s, vt = np.linalg.svd(nnv)
        normal_vec = vt[0]
        normal_cos = np.abs(np.dot(lines[i],normal_vec)) # scalar
        normal_angle[i] = np.arccos(normal_cos)*180/np.pi
        
    # Find the point where most angle is smaller than threshold
    num_ortho_ang = np.where((normal_angle>=(90-thre_ang)),1,0) # (num_pts//2,)
    idx_tp_onplane = np.where(num_ortho_ang>=1)[0]
    
    return idx_tp_onplane

def test_in_plane(lines,nn_vec,s_nn_vec,num_nn_p2p,thre_ang):
        
    # Normalize vectors
    lines /= np.linalg.norm(lines,axis=1,keepdims=True)+1e-7 # already normalzied but to be sure
    
    compare_rq = [nn_vec,s_nn_vec, num_nn_p2p,thre_ang]
    ind_onplane = compare_normal_svd(lines, compare_rq)
    # onplane index matching
    ind_not_onplane = np.setdiff1d(np.arange(len(lines)),ind_onplane)
    
    return ind_onplane, ind_not_onplane

def list2array_append(pts_use,lines_use,ind_use):
    # list to array
    pts_tp = np.zeros((1,3))
    lines_tp = np.zeros((1,3))
    ind_tp = np.zeros(1,dtype=np.int32)
    for p,l,i in zip(pts_use,lines_use,ind_use):
        pts_tp = np.vstack((pts_tp,p))
        lines_tp = np.vstack((lines_tp,l))
        ind_tp = np.hstack((ind_tp,i))
    pts_tp = np.delete(pts_tp,0,axis=0)
    lines_tp = np.delete(lines_tp,0,axis=0)
    ind_tp = np.delete(ind_tp,0)
    
    return pts_tp, lines_tp, ind_tp

def drawlines_tp_reject_plane(pts, THR_LOOP, THR_ANGLE):
    # To pair up normal vectors, mk whole nn vector sets
    if len(pts)%2==0:
        pass
    else:
        pts = pts[:-1]
    num_pts = len(pts)
    num_pts -= num_pts&1
    num_nn_p2p = int(min(num_pts*0.01,100))
    num_nn_p2p -= num_nn_p2p&1
    nn_vec = np.zeros((num_pts, num_nn_p2p, 3))
    s_nn_vec = np.zeros((nn_vec.shape)) 
    print("Make Nearest Neighbor vector set")
    # for i in tqdm(range(num_pts)):
    #     nn_vec[i] = get_vec_from_nn(pts[i],pts,num_nn_p2p) # (n, nn_p2p,3)
    
    pts = torch.from_numpy(pts).to(device)
    nn_vec = torch.from_numpy(nn_vec).to(device) 
    s_nn_vec = torch.from_numpy(s_nn_vec).to(device)
    for i in tqdm(range(num_pts)):
        nn_vec[i] = get_vec_from_nn_torch(pts[i],pts,num_nn_p2p) # (n, nn_p2p,3)
        n_U, n_s, n_vt = torch.svd(nn_vec[i]) # (num_nn_p2p, 3)
        nn_normal = n_vt[0] # (3,)
        s_nn_vec[i] = nn_normal # if only fits for axis 2, values are duplicated and fill axis1
        
    pts = pts.cpu().numpy()
    nn_vec = nn_vec.cpu().numpy()
    s_nn_vec = s_nn_vec.cpu().numpy()

    pre_ind = np.arange(num_pts)
    # tp draw lines
    test_ptss, test_lines, test_inds = drawlines_tp_two_sets(pts,pre_ind)  
    
    count=0
    num_pts_onplane = []
    pts_tp_use_a = []
    pts_tp_use_b = []
    lines_tp_use = []
    ind_tp_use_a = []
    ind_tp_use_b = []
    print("Compare normal vectors over NN by loop")
    while count<THR_LOOP:
        # find indx on plane
        ind_half_onplane, ind_half_use = test_in_plane(test_lines,nn_vec[test_inds[0]],s_nn_vec[test_inds[0]],num_nn_p2p,THR_ANGLE) # max: num_pts//2
        
        pts_tp_use_a.append(test_ptss[0][ind_half_use])
        pts_tp_use_b.append(test_ptss[1][ind_half_use])
        lines_tp_use.append(test_lines[ind_half_use])
        ind_tp_use_a.append(test_inds[0][ind_half_use])
        ind_tp_use_b.append(test_inds[1][ind_half_use])
        
        # If he onplane points dosen't diminishs for 10 iteration
        repeated_onplane = False
        num_pts_onplane.append(len(ind_half_onplane))
        if len(num_pts_onplane)>100:
            rep_onplane = []
            for n in num_pts_onplane[-50:]:
                if n == num_pts_onplane[-50]:
                    rep_onplane.append(True)
                else:
                    rep_onplane.append(False)
            repeated_onplane = all(rep_onplane)

        # if less than 10 pts in plane, then break
        if len(ind_half_onplane)<10 or repeated_onplane:
            break
        
        pts_onplane = np.vstack((test_ptss[0][ind_half_onplane],test_ptss[1][ind_half_onplane]))
        pre_ind = np.hstack((test_inds[0][ind_half_onplane],test_inds[1][ind_half_onplane]))
        
        # make tp again using pts on plane
        test_ptss, test_lines, test_inds = drawlines_tp_two_sets(pts_onplane,pre_ind)  
        count+=1
        
    print()
    print(len(ind_half_onplane),"points left on plane")
    print(f"{count} iteration loop finished")
    
    
    pts_tp_a, lines_tp, ind_tp_a = list2array_append(pts_tp_use_a,lines_tp_use,ind_tp_use_a)
    pts_tp_b, _, ind_tp_b = list2array_append(pts_tp_use_b,lines_tp_use,ind_tp_use_b)
    
    print("Test index_a,b has no intersection :",all([True if i not in ind_tp_b else False for i in ind_tp_a]))
    
    return [pts_tp_a, pts_tp_b], lines_tp, [ind_tp_a, ind_tp_b]

def voxel_3d_to_1d(voxel_index):
    i, j, k = voxel_index
    return i + j + k
