import numpy as np
import os
import sys
import glob

cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath("."))

from static import variable
from static import graph_var
from domain.loader.result_obj import ResultObject

def load_result_obj(_pose, _recover, _quality):
    result_obj = []

    if _pose:
        for pose_text in graph_var.POSE_TEXT:
            ro = ResultObject(pose_text)
            base_dir, dataset_dir, dataset_name, filename = findTextPath(pose_text, "PoseAccuracy", ro)
            if graph_var.REFINE_OPTION:
                ro.loadPoseResult(os.path.join(base_dir, "output", dataset_dir, dataset_name, "PoseAccuracy","refined", filename))
            else:
                ro.loadPoseResult(os.path.join(base_dir, "output", dataset_dir, dataset_name, "PoseAccuracy","notrefined", filename))

            result_obj.append(ro)
        
        return result_obj

    elif _recover:
        for recover_text in graph_var.POINT_TEXT:
            ro = ResultObject(recover_text)

            base_dir, dataset_dir, dataset_name, filename = findTextPath(recover_text, "L2Precon", ro)
            gt_img_path = os.path.join(base_dir, graph_var.DATASET_MOUNT, dataset_dir, dataset_name, "sparse_queryadded", "points3D.txt")
            ro.loadReconResult(os.path.join(base_dir, "output", dataset_dir, dataset_name, "L2Precon", filename), gt_img_path)

            result_obj.append(ro)

        return result_obj

    elif _quality:
        for q_type in graph_var.QUALITY_METIC:
            result_temp = []
            for temp_text_name in graph_var.QUALITY_TEXT:
                result_temp2 =[]
                for sparisity_level in graph_var.QUALITY_SPARSITY:
                    pose_text, quality_text, est_type = graph_var.getPoseQualityText(temp_text_name, sparisity_level,q_type)
                    pose_text_temp = pose_text.replace("NA", est_type)
                    ro = ResultObject(pose_text_temp)

                    base_dir, dataset_dir, dataset_name, filename = findTextPath(pose_text, "PoseAccuracy", ro)                    
                    if graph_var.REFINE_OPTION:
                        ro.loadPoseResult(os.path.join(base_dir, "output", dataset_dir, dataset_name, "PoseAccuracy","refined", filename))
                    else:
                        ro.loadPoseResult(os.path.join(base_dir, "output", dataset_dir, dataset_name, "PoseAccuracy","notrefined", filename))
                    
                    base_dir, dataset_dir, dataset_name, filename = findTextPath(quality_text, "Quality", ro)
                    ro.loadQualityResult(os.path.join(base_dir, "output", dataset_dir, dataset_name, "Quality", filename))

                    result_temp2.append(ro)
                result_temp.append(result_temp2)
            result_obj.append(result_temp)
            
        return result_obj
    
    else:
        raise Exception("No Text File Name is Provided, Please privode at least one text file")


def findTextPath(text_name, result_type, ro):
    base_dir = os.path.abspath(os.curdir)
    dataset_dir = variable.getDatasetName(ro.dataset_name)
    dataset_name = ro.dataset_name
    filename = text_name
    
    filepath = os.path.join(base_dir,'output', dataset_dir, dataset_name, result_type, text_name)
    if os.path.isfile(filepath):
        print(f"Processing {dataset_dir} : {dataset_name} : {filename} \n")
    
    else:
        raise Exception("No file exists", filepath)
    
    return base_dir, dataset_dir, dataset_name, text_name

    # PPLplus_gerrard_hall_TPF_sp0.01_n0.0_sw0.2
