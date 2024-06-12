# Copyright (c) Microsoft Corporation.
# Copyright (c) University of Florida Research Foundation, Inc.
# Licensed under the MIT License.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in 
# the Software without restriction, including without limitation the rights to 
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
# of the Software, and to permit persons to whom the Software is furnished to do 
# so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
# IN THE SOFTWARE.
#
# demo_colmap.py
# Demo script for running pre-trained models on data loaded directly from colmap sparse reconstruction files
# Author: Francesco Pittaluga

'''
    Last Edit Date  : 22.01.09
    Original Editor : Chanhyeok Yoon
    Last Editor     : Chunghwan Lee
'''
import os
import sys
import argparse
import torch

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)
from methods import *

def invsfm(inputfiles,src_path,recon_pts_path,eval_path,params,output_path):
    input_attr, scale_size, crop_size, sample_size, device = params

    # set paths for model wts
    weights_folder = os.path.join(curdir,'wts','pretrained', input_attr)
    vmodel_path = os.path.join(weights_folder,'visibnet.model.npz')
    cmodel_path = os.path.join(weights_folder,'coarsenet.model.npz')
    rmodel_path = os.path.join(weights_folder,'refinenet.model.npz')

    #-------------------------------------------------------------------------------
    # Codes for pytorch : preparing data
    #-------------------------------------------------------------------------------
    '''
    Data in tensorflow  : NHWC
    Data in pytorch     : NCHW
    data_torch = np.transpose( data_tf.numpy() , (0,3,1,2) )
    '''
    # Set device
    fp = [vmodel_path,cmodel_path,rmodel_path]
    for file in inputfiles:
        prm = OPTIONS(input_attr, scale_size, crop_size, sample_size)
        src_img_dir = os.path.join(src_path,'inv_src')
        
        # Recon image path
        recon_imgs_dir = os.path.join(output_path,'invsfmIMG',file.rstrip('.txt'))
        # recon_imgs_dir = os.path.join(output_path, 'invsfmIMG(line_rejection)', file.rstrip('.txt')) 

        paths = get_files_invsfm(src_path,recon_pts_path,file)
        mk_src(src_img_dir,paths,prm)
        recon_scenes(prm, paths, fp, recon_imgs_dir,device)
        print('Evaluation module initiated.')
        eval_imgs(src_img_dir, recon_imgs_dir, file, eval_path)


