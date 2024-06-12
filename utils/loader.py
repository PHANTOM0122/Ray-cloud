import numpy as np
import os
import sys
import glob

cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath("."))

from static import PoseCDFVAR
from domain.loader.output_obj import PoseAccuracyLoader


def loadPoseAccuracyResult():
    dataloader_obj = []
    base_dir = os.path.abspath(os.curdir)

    for dataset_dir in os.listdir(os.path.join(base_dir, "output")):
        for dataset_name in os.listdir(os.path.join(base_dir, "output", dataset_dir)):
            for filename in os.listdir(os.path.join(base_dir, "output", dataset_dir, dataset_name, "PoseAccuracy")):
                if filename.endswith(".txt"):
                    if dataset_name.lower() in PoseCDFVAR.LOAD_DATASET:
                        print(f"Processing {dataset_dir} : {dataset_name} : {filename} \n")
                        pa_obj = PoseAccuracyLoader(base_dir, dataset_dir, dataset_name, filename)
                        if pa_obj.sparsity_level in PoseCDFVAR.SPARSITY_LEVEL and pa_obj.noise_level in PoseCDFVAR.NOISE_LEVEL:

                            if pa_obj.line_type.lower():
                                dataloader_obj.append(pa_obj)


    return dataloader_obj

def loadReconResult():
    dataloader_obj = []
    base_dir = os.path.abspath(os.curdir)

    for dataset_dir in os.listdir(os.path.join(base_dir, "output")):
        for dataset_name in os.listdir(os.path.join(base_dir, "output", dataset_dir)):
            for filename in os.listdir(os.path.join(base_dir, "output", dataset_dir, dataset_name, "PoseAccuracy")):
                if filename.endswith(".txt"):
                    if dataset_name.lower() in PoseCDFVAR.LOAD_DATASET:
                        print(f"Processing {dataset_dir} : {dataset_name} : {filename} \n")
                        pa_obj = ReconLoader(base_dir, dataset_dir, dataset_name, filename)
                        if pa_obj.sparsity_level in PoseCDFVAR.SPARSITY_LEVEL and pa_obj.noise_level in PoseCDFVAR.NOISE_LEVEL:

                            if pa_obj.line_type.lower():
                                dataloader_obj.append(pa_obj)


    return dataloader_obj

if __name__ == "__main__":
    loadPoseAccuracyResult()