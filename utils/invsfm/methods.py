import os
import sys
import torch
import torch.nn as nn
import math
import skimage
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
from PIL import Image

import utils.invsfm.load_data_edit as ld
from utils.invsfm import models

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utils.colmap.read_write_model as read_model
from utils.pose import line
from sklearn.cluster import KMeans
from static import variable

global device, start
start = 0

def set_vnet(vinp_ch, vwts_dir):
    print('Loading VisibNet...')
    vnet = models.VisibNet(vinp_ch)

    d = np.load(vwts_dir)
    weights = [val[1] for val in d.items()]
    wts = []
    bias = []
    mn = []
    var = []
    for i in range(int(len(weights)/4)):
        wts.append(weights[4*i])
        bias.append(weights[4*i+3])
        mn.append(weights[4*i+1])
        var.append(weights[4*i+2])
    wts.append(weights[int(len(weights)/4)*4])
    bias.append(weights[int(len(weights)/4)*4+1])
    #---------------------------------------------------------------------
    # Parameter transfer from tensorflow to pytorch
    #---------------------------------------------------------------------
    vnet.down1[0].weight = nn.Parameter(torch.from_numpy(wts[0]).to(torch.float32).permute((3,2,0,1)))
    vnet.down2[0].weight = nn.Parameter(torch.from_numpy(wts[1]).to(torch.float32).permute((3,2,0,1)))
    vnet.down3[0].weight = nn.Parameter(torch.from_numpy(wts[2]).to(torch.float32).permute((3,2,0,1)))
    vnet.down4[0].weight = nn.Parameter(torch.from_numpy(wts[3]).to(torch.float32).permute((3,2,0,1)))
    vnet.down5[0].weight = nn.Parameter(torch.from_numpy(wts[4]).to(torch.float32).permute((3,2,0,1)))
    vnet.down6[0].weight = nn.Parameter(torch.from_numpy(wts[5]).to(torch.float32).permute((3,2,0,1)))

    vnet.down1[1].running_mean = (torch.from_numpy(mn[0]).to(torch.float32))
    vnet.down2[1].running_mean = (torch.from_numpy(mn[1]).to(torch.float32))
    vnet.down3[1].running_mean = (torch.from_numpy(mn[2]).to(torch.float32))
    vnet.down4[1].running_mean = (torch.from_numpy(mn[3]).to(torch.float32))
    vnet.down5[1].running_mean = (torch.from_numpy(mn[4]).to(torch.float32))
    vnet.down6[1].running_mean = (torch.from_numpy(mn[5]).to(torch.float32))

    vnet.down1[1].running_var = (torch.from_numpy(var[0]).to(torch.float32))
    vnet.down2[1].running_var = (torch.from_numpy(var[1]).to(torch.float32))
    vnet.down3[1].running_var = (torch.from_numpy(var[2]).to(torch.float32))
    vnet.down4[1].running_var = (torch.from_numpy(var[3]).to(torch.float32))
    vnet.down5[1].running_var = (torch.from_numpy(var[4]).to(torch.float32))
    vnet.down6[1].running_var = (torch.from_numpy(var[5]).to(torch.float32))

    vnet.down1[1].bias = nn.Parameter(torch.from_numpy(bias[0]).to(torch.float32))
    vnet.down2[1].bias = nn.Parameter(torch.from_numpy(bias[1]).to(torch.float32))
    vnet.down3[1].bias = nn.Parameter(torch.from_numpy(bias[2]).to(torch.float32))
    vnet.down4[1].bias = nn.Parameter(torch.from_numpy(bias[3]).to(torch.float32))
    vnet.down5[1].bias = nn.Parameter(torch.from_numpy(bias[4]).to(torch.float32))
    vnet.down6[1].bias = nn.Parameter(torch.from_numpy(bias[5]).to(torch.float32))

    vnet.up1[0].weight = nn.Parameter(torch.from_numpy(wts[6]).to(torch.float32).permute((3,2,0,1)))
    vnet.up2[0].weight = nn.Parameter(torch.from_numpy(wts[7]).to(torch.float32).permute((3,2,0,1)))
    vnet.up3[0].weight = nn.Parameter(torch.from_numpy(wts[8]).to(torch.float32).permute((3,2,0,1)))
    vnet.up4[0].weight = nn.Parameter(torch.from_numpy(wts[9]).to(torch.float32).permute((3,2,0,1)))
    vnet.up5[0].weight = nn.Parameter(torch.from_numpy(wts[10]).to(torch.float32).permute((3,2,0,1)))
    vnet.up6[0].weight = nn.Parameter(torch.from_numpy(wts[11]).to(torch.float32).permute((3,2,0,1)))
    vnet.up7[0].weight = nn.Parameter(torch.from_numpy(wts[12]).to(torch.float32).permute((3,2,0,1)))
    vnet.up8[0].weight = nn.Parameter(torch.from_numpy(wts[13]).to(torch.float32).permute((3,2,0,1)))
    vnet.up9[0].weight = nn.Parameter(torch.from_numpy(wts[14]).to(torch.float32).permute((3,2,0,1)))
    vnet.up10[0].weight = nn.Parameter(torch.from_numpy(wts[15]).to(torch.float32).permute((3,2,0,1)))

    vnet.up1[1].running_mean = (torch.from_numpy(mn[6]).to(torch.float32))
    vnet.up2[1].running_mean = (torch.from_numpy(mn[7]).to(torch.float32))
    vnet.up3[1].running_mean = (torch.from_numpy(mn[8]).to(torch.float32))
    vnet.up4[1].running_mean = (torch.from_numpy(mn[9]).to(torch.float32))
    vnet.up5[1].running_mean = torch.from_numpy(mn[10]).to(torch.float32)
    vnet.up6[1].running_mean = torch.from_numpy(mn[11]).to(torch.float32)
    vnet.up7[1].running_mean = torch.from_numpy(mn[12]).to(torch.float32)
    vnet.up8[1].running_mean = torch.from_numpy(mn[13]).to(torch.float32)
    vnet.up9[1].running_mean = torch.from_numpy(mn[14]).to(torch.float32)

    vnet.up1[1].running_var = torch.from_numpy(var[6]).to(torch.float32)
    vnet.up2[1].running_var = torch.from_numpy(var[7]).to(torch.float32)
    vnet.up3[1].running_var = torch.from_numpy(var[8]).to(torch.float32)
    vnet.up4[1].running_var = torch.from_numpy(var[9]).to(torch.float32)
    vnet.up5[1].running_var = torch.from_numpy(var[10]).to(torch.float32)
    vnet.up6[1].running_var = torch.from_numpy(var[11]).to(torch.float32)
    vnet.up7[1].running_var = torch.from_numpy(var[12]).to(torch.float32)
    vnet.up8[1].running_var = torch.from_numpy(var[13]).to(torch.float32)
    vnet.up9[1].running_var = torch.from_numpy(var[14]).to(torch.float32)

    vnet.up1[1].bias = nn.Parameter(torch.from_numpy(bias[6]).to(torch.float32))
    vnet.up2[1].bias = nn.Parameter(torch.from_numpy(bias[7]).to(torch.float32))
    vnet.up3[1].bias = nn.Parameter(torch.from_numpy(bias[8]).to(torch.float32))
    vnet.up4[1].bias = nn.Parameter(torch.from_numpy(bias[9]).to(torch.float32))
    vnet.up5[1].bias = nn.Parameter(torch.from_numpy(bias[10]).to(torch.float32))
    vnet.up6[1].bias = nn.Parameter(torch.from_numpy(bias[11]).to(torch.float32))
    vnet.up7[1].bias = nn.Parameter(torch.from_numpy(bias[12]).to(torch.float32))
    vnet.up8[1].bias = nn.Parameter(torch.from_numpy(bias[13]).to(torch.float32))
    vnet.up9[1].bias = nn.Parameter(torch.from_numpy(bias[14]).to(torch.float32))
    vnet.up10[0].bias = nn.Parameter(torch.from_numpy(bias[15]).to(torch.float32))

    vnet = vnet.to(device)
    return vnet
def set_cnet(cinp_ch, cwts_dir):
    print('Loading CoarseNet...')
    # 2. CoarseNet
    cnet = models.CoarseNet(cinp_ch)

    d = np.load(cwts_dir)
    weights = [val[1] for val in d.items()]
    wts = []
    bias = []
    mn = []
    var = []
    for i in range(int(len(weights)/4)):
        wts.append(weights[4*i])
        bias.append(weights[4*i+3])
        mn.append(weights[4*i+1])
        var.append(weights[4*i+2])
    wts.append(weights[int(len(weights)/4)*4])
    bias.append(weights[int(len(weights)/4)*4+1])
    #---------------------------------------------------------------------
    # Parameter transfer from tensorflow to pytorch
    #---------------------------------------------------------------------
    cnet.down1[0].weight = nn.Parameter(torch.from_numpy(wts[0]).to(torch.float32).permute((3,2,0,1)))
    cnet.down2[0].weight = nn.Parameter(torch.from_numpy(wts[1]).to(torch.float32).permute((3,2,0,1)))
    cnet.down3[0].weight = nn.Parameter(torch.from_numpy(wts[2]).to(torch.float32).permute((3,2,0,1)))
    cnet.down4[0].weight = nn.Parameter(torch.from_numpy(wts[3]).to(torch.float32).permute((3,2,0,1)))
    cnet.down5[0].weight = nn.Parameter(torch.from_numpy(wts[4]).to(torch.float32).permute((3,2,0,1)))
    cnet.down6[0].weight = nn.Parameter(torch.from_numpy(wts[5]).to(torch.float32).permute((3,2,0,1)))

    cnet.down1[1].running_mean = (torch.from_numpy(mn[0]).to(torch.float32))
    cnet.down2[1].running_mean = (torch.from_numpy(mn[1]).to(torch.float32))
    cnet.down3[1].running_mean = (torch.from_numpy(mn[2]).to(torch.float32))
    cnet.down4[1].running_mean = (torch.from_numpy(mn[3]).to(torch.float32))
    cnet.down5[1].running_mean = (torch.from_numpy(mn[4]).to(torch.float32))
    cnet.down6[1].running_mean = (torch.from_numpy(mn[5]).to(torch.float32))

    cnet.down1[1].running_var = (torch.from_numpy(var[0]).to(torch.float32))
    cnet.down2[1].running_var = (torch.from_numpy(var[1]).to(torch.float32))
    cnet.down3[1].running_var = (torch.from_numpy(var[2]).to(torch.float32))
    cnet.down4[1].running_var = (torch.from_numpy(var[3]).to(torch.float32))
    cnet.down5[1].running_var = (torch.from_numpy(var[4]).to(torch.float32))
    cnet.down6[1].running_var = (torch.from_numpy(var[5]).to(torch.float32))

    cnet.down1[1].bias = nn.Parameter(torch.from_numpy(bias[0]).to(torch.float32))
    cnet.down2[1].bias = nn.Parameter(torch.from_numpy(bias[1]).to(torch.float32))
    cnet.down3[1].bias = nn.Parameter(torch.from_numpy(bias[2]).to(torch.float32))
    cnet.down4[1].bias = nn.Parameter(torch.from_numpy(bias[3]).to(torch.float32))
    cnet.down5[1].bias = nn.Parameter(torch.from_numpy(bias[4]).to(torch.float32))
    cnet.down6[1].bias = nn.Parameter(torch.from_numpy(bias[5]).to(torch.float32))

    cnet.up1[0].weight = nn.Parameter(torch.from_numpy(wts[6]).to(torch.float32).permute((3,2,0,1)))
    cnet.up2[0].weight = nn.Parameter(torch.from_numpy(wts[7]).to(torch.float32).permute((3,2,0,1)))
    cnet.up3[0].weight = nn.Parameter(torch.from_numpy(wts[8]).to(torch.float32).permute((3,2,0,1)))
    cnet.up4[0].weight = nn.Parameter(torch.from_numpy(wts[9]).to(torch.float32).permute((3,2,0,1)))
    cnet.up5[0].weight = nn.Parameter(torch.from_numpy(wts[10]).to(torch.float32).permute((3,2,0,1)))
    cnet.up6[0].weight = nn.Parameter(torch.from_numpy(wts[11]).to(torch.float32).permute((3,2,0,1)))
    cnet.up7[0].weight = nn.Parameter(torch.from_numpy(wts[12]).to(torch.float32).permute((3,2,0,1)))
    cnet.up8[0].weight = nn.Parameter(torch.from_numpy(wts[13]).to(torch.float32).permute((3,2,0,1)))
    cnet.up9[0].weight = nn.Parameter(torch.from_numpy(wts[14]).to(torch.float32).permute((3,2,0,1)))
    cnet.up10[0].weight = nn.Parameter(torch.from_numpy(wts[15]).to(torch.float32).permute((3,2,0,1)))

    cnet.up1[1].running_mean = (torch.from_numpy(mn[6]).to(torch.float32))
    cnet.up2[1].running_mean = (torch.from_numpy(mn[7]).to(torch.float32))
    cnet.up3[1].running_mean = (torch.from_numpy(mn[8]).to(torch.float32))
    cnet.up4[1].running_mean = (torch.from_numpy(mn[9]).to(torch.float32))
    cnet.up5[1].running_mean = torch.from_numpy(mn[10]).to(torch.float32)
    cnet.up6[1].running_mean = torch.from_numpy(mn[11]).to(torch.float32)
    cnet.up7[1].running_mean = torch.from_numpy(mn[12]).to(torch.float32)
    cnet.up8[1].running_mean = torch.from_numpy(mn[13]).to(torch.float32)
    cnet.up9[1].running_mean = torch.from_numpy(mn[14]).to(torch.float32)

    cnet.up1[1].running_var = torch.from_numpy(var[6]).to(torch.float32)
    cnet.up2[1].running_var = torch.from_numpy(var[7]).to(torch.float32)
    cnet.up3[1].running_var = torch.from_numpy(var[8]).to(torch.float32)
    cnet.up4[1].running_var = torch.from_numpy(var[9]).to(torch.float32)
    cnet.up5[1].running_var = torch.from_numpy(var[10]).to(torch.float32)
    cnet.up6[1].running_var = torch.from_numpy(var[11]).to(torch.float32)
    cnet.up7[1].running_var = torch.from_numpy(var[12]).to(torch.float32)
    cnet.up8[1].running_var = torch.from_numpy(var[13]).to(torch.float32)
    cnet.up9[1].running_var = torch.from_numpy(var[14]).to(torch.float32)

    cnet.up1[1].bias = nn.Parameter(torch.from_numpy(bias[6]).to(torch.float32))
    cnet.up2[1].bias = nn.Parameter(torch.from_numpy(bias[7]).to(torch.float32))
    cnet.up3[1].bias = nn.Parameter(torch.from_numpy(bias[8]).to(torch.float32))
    cnet.up4[1].bias = nn.Parameter(torch.from_numpy(bias[9]).to(torch.float32))
    cnet.up5[1].bias = nn.Parameter(torch.from_numpy(bias[10]).to(torch.float32))
    cnet.up6[1].bias = nn.Parameter(torch.from_numpy(bias[11]).to(torch.float32))
    cnet.up7[1].bias = nn.Parameter(torch.from_numpy(bias[12]).to(torch.float32))
    cnet.up8[1].bias = nn.Parameter(torch.from_numpy(bias[13]).to(torch.float32))
    cnet.up9[1].bias = nn.Parameter(torch.from_numpy(bias[14]).to(torch.float32))
    cnet.up10[0].bias = nn.Parameter(torch.from_numpy(bias[15]).to(torch.float32))

    cnet = cnet.to(device)
    return cnet
def set_rnet(rinp_ch, rwts_dir, input_attr):
    print('Loading RefineNet...')
    d = np.load(rwts_dir)

    weights = [val[1] for val in d.items()]
    wts = []
    bias = []
    for i in range(int(len(weights)/2)):
        wts.append(weights[i*2])
        bias.append(weights[i*2+1])
    #---------------------------------------------------------------------
    # Parameter transfer from tensorflow to pytorch
    #---------------------------------------------------------------------
    rnet = models.RefineNet(rinp_ch)

    rnet.down1[0].weight = nn.Parameter(torch.from_numpy(wts[0]).to(torch.float32).permute((3,2,0,1)))
    rnet.down2[0].weight = nn.Parameter(torch.from_numpy(wts[1]).to(torch.float32).permute((3,2,0,1)))
    rnet.down3[0].weight = nn.Parameter(torch.from_numpy(wts[2]).to(torch.float32).permute((3,2,0,1)))
    rnet.down4[0].weight = nn.Parameter(torch.from_numpy(wts[3]).to(torch.float32).permute((3,2,0,1)))
    rnet.down5[0].weight = nn.Parameter(torch.from_numpy(wts[4]).to(torch.float32).permute((3,2,0,1)))
    rnet.down6[0].weight = nn.Parameter(torch.from_numpy(wts[5]).to(torch.float32).permute((3,2,0,1)))

    rnet.down1[1].bias = nn.Parameter(torch.from_numpy(bias[0]).to(torch.float32))
    rnet.down2[1].bias = nn.Parameter(torch.from_numpy(bias[1]).to(torch.float32))
    rnet.down3[1].bias = nn.Parameter(torch.from_numpy(bias[2]).to(torch.float32))
    rnet.down4[1].bias = nn.Parameter(torch.from_numpy(bias[3]).to(torch.float32))
    rnet.down5[1].bias = nn.Parameter(torch.from_numpy(bias[4]).to(torch.float32))
    rnet.down6[1].bias = nn.Parameter(torch.from_numpy(bias[5]).to(torch.float32))

    rnet.up1[0].weight = nn.Parameter(torch.from_numpy(wts[6]).to(torch.float32).permute((3,2,0,1)))
    rnet.up2[0].weight = nn.Parameter(torch.from_numpy(wts[7]).to(torch.float32).permute((3,2,0,1)))
    rnet.up3[0].weight = nn.Parameter(torch.from_numpy(wts[8]).to(torch.float32).permute((3,2,0,1)))
    rnet.up4[0].weight = nn.Parameter(torch.from_numpy(wts[9]).to(torch.float32).permute((3,2,0,1)))
    rnet.up5[0].weight = nn.Parameter(torch.from_numpy(wts[10]).to(torch.float32).permute((3,2,0,1)))
    rnet.up6[0].weight = nn.Parameter(torch.from_numpy(wts[11]).to(torch.float32).permute((3,2,0,1)))
    rnet.up7[0].weight = nn.Parameter(torch.from_numpy(wts[12]).to(torch.float32).permute((3,2,0,1)))
    rnet.up8[0].weight = nn.Parameter(torch.from_numpy(wts[13]).to(torch.float32).permute((3,2,0,1)))
    rnet.up9[0].weight = nn.Parameter(torch.from_numpy(wts[14]).to(torch.float32).permute((3,2,0,1)))
    rnet.up10[0].weight = nn.Parameter(torch.from_numpy(wts[15]).to(torch.float32).permute((3,2,0,1)))

    rnet.up1[1].bias = nn.Parameter(torch.from_numpy(bias[6]).to(torch.float32))
    rnet.up2[1].bias = nn.Parameter(torch.from_numpy(bias[7]).to(torch.float32))
    rnet.up3[1].bias = nn.Parameter(torch.from_numpy(bias[8]).to(torch.float32))
    rnet.up4[1].bias = nn.Parameter(torch.from_numpy(bias[9]).to(torch.float32))
    rnet.up5[1].bias = nn.Parameter(torch.from_numpy(bias[10]).to(torch.float32))
    rnet.up6[1].bias = nn.Parameter(torch.from_numpy(bias[11]).to(torch.float32))
    rnet.up7[1].bias = nn.Parameter(torch.from_numpy(bias[12]).to(torch.float32))
    rnet.up8[1].bias = nn.Parameter(torch.from_numpy(bias[13]).to(torch.float32))
    rnet.up9[1].bias = nn.Parameter(torch.from_numpy(bias[14]).to(torch.float32))
    rnet.up10[0].bias = nn.Parameter(torch.from_numpy(bias[15]).to(torch.float32))

    rnet = rnet.to(device)
    return rnet

def eval_vnet(vinp, vnet, valid):
    with torch.no_grad():
        vnet.eval()
        vout = vnet(vinp)
    vpred = torch.logical_and(torch.gt(vout,.5),valid)
    vpredf = vpred.to(torch.float32)*0.+1.
    vpred = vpred.cpu()
    vpredf = vpredf.permute((0,2,3,1))
    return vpred, vpredf

def eval_cnet(cinp, cnet):
    with torch.no_grad():
        cnet.eval()
        cpred = cnet(cinp)
    cpred_ = (cpred+1.)*127.5
    cpred_ = cpred_.cpu()
    return cpred_, cpred

def eval_rnet(rinp, rnet):
    with torch.no_grad():
        rnet.eval()
        rpred = rnet(rinp)
    rpred = (rpred+1.)*127.5
    rpred_ = rpred.cpu()
    return rpred_

def v_eval(inp_ch, vinp, valid, vwts_dir):
    vnet = set_vnet(inp_ch, vwts_dir)
    vpred_ = torch.zeros([1,1,vinp.shape[2],vinp.shape[3]])
    vpredf_ = torch.zeros([1,vinp.shape[2],vinp.shape[3],1]).to(device)
    for i in range(vinp.shape[0]):
        vpred, vpredf = eval_vnet(vinp[i:i+1], vnet, valid[i:i+1])
        vpred_ = torch.cat([vpred_, vpred], dim=0)
        vpredf_ = torch.cat([vpredf_, vpredf], dim=0)
    vpred = vpred_[1:]
    vpredf = vpredf_[1:].to(device)
    return vpred, vpredf

def c_eval(inp_ch, cinp, cwts_dir):
    cnet = set_cnet(inp_ch, cwts_dir)
    cpred_ = torch.zeros([1,3,cinp.shape[2],cinp.shape[3]])
    cpredf_ = torch.zeros([1,3,cinp.shape[2],cinp.shape[3]]).to(device)
    for i in range(cinp.shape[0]):
        cpred, cpredf = eval_cnet(cinp[i:i+1], cnet)
        cpred_ = torch.cat([cpred_, cpred], dim=0)
        cpredf_ = torch.cat([cpredf_, cpredf], dim=0)
    cpred = cpred_[1:]
    cpredf = cpredf_[1:]
    return cpred, cpredf

def r_eval(inp_ch, rinp, rwts_dir, prm):
    rnet = set_rnet(inp_ch+3, rwts_dir, prm.input_attr)
    rpred_ = torch.zeros([1,3,rinp.shape[2],rinp.shape[3]])
    for i in range(rinp.shape[0]):
        rpred = eval_rnet(rinp[i:i+1], rnet)
        rpred_ = torch.cat([rpred_, rpred], dim=0)
    rpred = rpred_[1:]
    return rpred

def convert_to_matrix(q):
    q0, q1, q2, q3 = q
    R = np.zeros((3, 3),dtype=float)
    R[0][0] = 1 - 2*(q2*q2 + q3*q3)
    R[0][1] = 2*(q1*q2 - q3*q0)
    R[0][2] = 2*(q1*q3 + q2*q0)
    
    R[1][0] = 2*(q1*q2 + q3*q0)
    R[1][1] = 1 - 2*(q1*q1 + q3*q3)
    R[1][2] = 2*(q2*q3 - q1*q0)
    
    R[2][0] = 2*(q1*q3 - q2*q0)
    R[2][1] = 2*(q2*q3 + q1*q0)
    R[2][2] = 1 - 2*(q1*q1 + q2*q2)
    return R

def gen_projection(info,num_samples,scale_size,crop_size):
    pts, cam = info
    pcl_xyz, pcl_rgb, pcl_sift = pts
    K,R,T,h,w,src_img_list = cam

    # Generate projections
    proj_depth = []
    proj_sift = [] 
    proj_rgb = []
  
    for i in range(len(K))[start:start+num_samples]:

        proj_mat = K[i].dot(np.hstack((R[i],T[i])))
        
        pdepth, prgb, psift = ld.project_points(pcl_xyz, pcl_rgb, pcl_sift, proj_mat, h[i], w[i], scale_size, crop_size)    
        proj_depth.append((pdepth)[None,...])
        proj_sift.append((psift)[None,...])
        proj_rgb.append((prgb)[None,...])

    proj_depth = np.vstack(proj_depth)
    proj_sift = np.vstack(proj_sift)
    proj_rgb = np.vstack(proj_rgb)

    # Pytorch input
    pdepth = torch.tensor(proj_depth, dtype=torch.float32)
    psift = torch.tensor(proj_sift, dtype=torch.uint8)
    prgb = torch.tensor(proj_rgb, dtype=torch.uint8)

    pdepth = pdepth.to(device)
    psift = psift.to(device)
    prgb = prgb.to(device)

    psift = psift.to(torch.float32)
    prgb = prgb.to(torch.float32)

    valid = torch.greater(pdepth, 0.) # With cheirality

    return pdepth, psift, prgb, valid, src_img_list[start:start+num_samples]

def getQueryImageId(query_img_names, query_txt_fp):
    # Find query ID corresponding to the name
    images_gt = read_model.read_images_text(query_txt_fp)
    query_img_id = []
    for k, v in images_gt.items():
        if v.name in query_img_names:
            query_img_id.append(v.id)
    return query_img_id

def preprocess_load(db_dir, pts3d_dir, cam_dir, img_dir, imgID_list):
    # Load point cloud with per-point sift descriptors and rgb features from
    # colmap database and points3D.bin file from colmap sparse reconstruction
    print('Loading point cloud...')
    pcl_xyz, pcl_rgb, pcl_sift = ld.load_points_colmap(db_dir,pts3d_dir)
    pts_info = [pcl_xyz, pcl_rgb, pcl_sift]
    print('Done!')

    # Load camera matrices and from images.bin and cameras.bin files from
    # colmap sparse reconstruction
    print('Loading cameras...')
    K,R,T,h,w,src_img_list = ld.load_cameras_colmap_wID(img_dir,cam_dir, image_id=imgID_list)
    cam_info = [K,R,T,h,w,src_img_list]
    print('Done!')
    
    return [pts_info,cam_info]

def load_vinp_wtName(info, prm):
    input_attr = prm.input_attr
    pdepth, psift, prgb, valid, src_img_list = gen_projection(info,prm.num_samples, prm.scale_size, prm.crop_size)
    
    # set up visibnet
    if input_attr=='depth':
        vinp = pdepth
        inp_ch = 1
    elif input_attr=='depth_rgb':
        vinp = torch.cat((pdepth, prgb/127.5-1.), dim=3)
        inp_ch = 4
    elif input_attr=='depth_sift':
        vinp = torch.cat((pdepth, psift/127.5-1.), dim=3)
        inp_ch = 129
    elif input_attr=='depth_sift_rgb':
        vinp = torch.cat((pdepth, psift/127.5-1., prgb/127.5-1.), dim=3)
        inp_ch = 132

    vinp_ch = inp_ch
    valid = valid.permute(0,3,1,2)
    vinp = vinp.permute(0,3,1,2)
    return vinp, vinp_ch, valid, pdepth, psift, prgb, src_img_list

def sample_iter(prep_info,  prm, fp, imgsave_dir):
    vinp, inp_ch, valid, pdepth, psift, prgb, src_img_name = load_vinp_wtName(prep_info, prm)
    vnet_wts_fp, cnet_wts_fp, rnet_wts_fp= fp
    vpred, vpredf = v_eval(inp_ch, vinp, valid, vnet_wts_fp)

    # 2. CoarseNet
    # set up coarsenet 
    if prm.input_attr=='depth':
        cinp = pdepth*vpredf
    elif prm.input_attr=='depth_rgb':
        cinp = torch.cat((pdepth*vpredf, prgb*vpredf/127.5-1.), dim=3)
    elif prm.input_attr=='depth_sift':
        cinp = torch.cat((pdepth*vpredf, psift*vpredf/127.5-1.), dim=3)
    elif prm.input_attr=='depth_sift_rgb':
        cinp = torch.cat((pdepth*vpredf, psift*vpredf/127.5-1., prgb*vpredf/127.5-1.), dim=3)
    cinp = cinp.permute((0,3,1,2))
    cpred, cpredf = c_eval(inp_ch, cinp, cnet_wts_fp)

    # 3. RefineNet
    # set up refinenet
    rinp = torch.cat((cpredf, cinp), dim=1)
    rpred = r_eval(inp_ch, rinp, rnet_wts_fp, prm)

    os.makedirs(imgsave_dir, exist_ok=True)
    print('Saving post-RefineNet images to {}...'.format(imgsave_dir))
    for i in range(rpred.shape[0]):
        rpred_img = np.array(rpred[i])
        rpred_img = np.transpose(rpred_img, (1,2,0))
        rpred_img = Image.fromarray(rpred_img.astype(np.uint8))
        rpred_img.save(os.path.join(imgsave_dir,src_img_name[i][:-3]+'png'))
    print('Done!')

def load_name(cam_dir, img_dir, imgID_list):
    _,_,_,_,_,src_img_list = ld.load_cameras_colmap_wID(img_dir,cam_dir, image_id=imgID_list)
    return src_img_list

def get_query_image_names(query_txt):
    query_img_names = []
    with open(query_txt, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            query_img_names.append(line.strip())
    
    return query_img_names

def recon_scenes(prm, paths, fp, imgsave_dir, mydevice):
    
    global start
    global device
    device = mydevice
    start = 0
    original_numsamples = prm.num_samples
    
    path_db,path_pts3d,path_cameras,path_images,path_querytxt,path_queryfolder = paths

    imgname_list = get_query_image_names(path_querytxt)
    imgID_list = getQueryImageId(imgname_list, path_images)
    iter_num = len(imgID_list)//prm.num_samples
    last_iter_num = len(imgID_list)%prm.num_samples
    if last_iter_num != 0:
        iter_num = iter_num +1
    prep_info = preprocess_load(path_db, path_pts3d, path_cameras, path_images, imgID_list)
    for i in range(iter_num):
        prm.num_samples = original_numsamples
        if i == iter_num:
            prm.num_samples = last_iter_num
        sample_iter(prep_info, prm, fp,imgsave_dir)
        start += prm.num_samples
        if i == iter_num -1:
            print('{}/{} Done!'.format(len(imgID_list), len(imgID_list)))
        else:
            print('{}/{} Done!'.format(start, len(imgID_list)))

def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255. / math.sqrt(mse))

def cal_mae(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mae = np.mean(np.abs(img1-img2))
    return mae

def cal_psnr_ssim_mae(ori_img_dir, img_dir):
    ori_img = np.array(Image.open(ori_img_dir))
    img = np.array(Image.open(img_dir))
    psnr = calculate_psnr(ori_img, img)
    ssim = skimage.metrics.structural_similarity(ori_img, img, channel_axis=2, data_range=255)
    mae = cal_mae(ori_img, img)
    return psnr, ssim, mae

def savetxt(save_fp, eval_opt, ptsname, quality_values):
    with open(os.path.join(save_fp,f'{eval_opt}_{ptsname}'), 'w', encoding='UTF-8') as f:
        f.write('MEAN: '+str(sum(quality_values)/len(quality_values))+'\n')
        f.write('MEADIAN: '+str(np.median(np.array(quality_values)).item())+'\n')
        for value in quality_values:
            f.write(str(value) + '\n')

def eval_imgs(src_imgs_dir, imgs_dir, txtname_pts, eval_fp):
    src_img_list = os.listdir(src_imgs_dir)
    src_imgs_list = [img for img in src_img_list if img[-3:]=='png']
    img_list = os.listdir(imgs_dir)
    imgs_list = [img for img in img_list if img[-3:]=='png']
    
    for i in imgs_list:
        if i in src_imgs_list:
            continue
        else:
            print("Src & recon query do not match")
            exit()

    ssim_list = []
    psnr_list = []
    mae_list = []
    
    os.makedirs(eval_fp,exist_ok=True)
    for img in imgs_list:
        src_img = os.path.join(src_imgs_dir,img)
        img_dir = os.path.join(imgs_dir,img)
        psnr, ssim, mae = cal_psnr_ssim_mae(src_img, img_dir)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        mae_list.append(mae)
        
    savetxt(eval_fp,'PSNR', txtname_pts, psnr_list)
    savetxt(eval_fp,'SSIM', txtname_pts, ssim_list)
    savetxt(eval_fp,'MAE', txtname_pts, mae_list)
    print("Evaluation Finished!\n")

def mk_src(src_dir,paths,prm):
    if os.path.exists(src_dir):
        print("Scr already exists")
        return 1
    else:
        path_db,path_pts3d,path_cameras,path_images,path_querytxt,path_queryfolder = paths
        os.makedirs(src_dir,exist_ok=True)
        print(f'Saving src files to {src_dir}...')
        query_files =  get_query_image_names(path_querytxt)
        for file in query_files:
            src_img = np.array(Image.open(os.path.join(path_queryfolder,file)))
            src_img = ld.scale_crop(src_img, prm.scale_size, prm.crop_size)
            src_img_ = Image.fromarray(src_img.astype(np.uint8))
            src_img_.save(os.path.join(src_dir,file[:-3]+'png'))
        print('Done!')
        
        return 0

def get_files_invsfm(data_path,recon_pts_path,recontxt):
    path_pts3d = os.path.join(recon_pts_path,recontxt)
    path_cameras = os.path.join(data_path,'sparse_queryadded','cameras.txt')
    path_images = os.path.join(data_path,'sparse_queryadded','images.txt')
    path_db = os.path.join(data_path,'sparse_queryadded','database.db')
    path_querytxt = os.path.join(data_path,'query_imgs.txt')
    path_queryfolder = os.path.join(data_path,'query')

    return [path_db,path_pts3d,path_cameras,path_images,path_querytxt,path_queryfolder]

class OPTIONS():
    def __init__(self,input_attr, scale_size, crop_size, sample_size):
        self.input_attr = input_attr
        self.scale_size = scale_size
        self.crop_size = crop_size
        self.num_samples = sample_size
        
        