from static import variable
from utils.colmap import read_write_model
import collections
import os
import sys
import numpy as np

BaseImage = collections.namedtuple(
    "Image", ["name", "qvec", "tvec"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def loadDatasetCambridge(base_dir):
    datasetText = ["dataset_test.txt", "dataset_train.txt"]
    images = {}

    for txt in datasetText:
        with open(os.path.join(base_dir, txt), "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break

                line = line.strip()
                if len(line) > 0 and line[0:3] == "seq":
                    elems = line.split()
                    qvec = np.array(tuple(map(float, elems[4:8])))
                    qvec /= np.linalg.norm(qvec)
                    tvec = np.array(tuple(map(float, elems[1:4])))
                    image_name = elems[0].replace("/", "_")
                    images[image_name] = Image(
                        qvec=qvec, tvec=tvec, name=image_name)

    return images


def loadDatasetEnergy(base_dir):
    pose_dir = os.path.join(base_dir, "pose")
    images = {}

    cnt = 0
    for txt in os.listdir(pose_dir):
        if txt[-4:] == ".txt":
            with open(os.path.join(pose_dir, txt), "r") as fid:
                PMatrix = np.empty((0, 4), float)
                tvec = np.array([])
                while True:
                    line = fid.readline()
                    if not line:
                        break

                    line = line.strip()
                    if len(line) > 0:
                        elems = line.split()
                        PMatrix = np.vstack([PMatrix, np.array(tuple(map(float, elems)))])

                image_name = txt[:-4].replace("pose", "color.jpg")
                qvec = model.rotmat2qvec(PMatrix[:3, :3])
                qvec /= np.linalg.norm(qvec)
                qvec = np.array([-qvec[0], qvec[1], qvec[2], qvec[3]])
                tvec = PMatrix[:, -1][:-1]

            images[image_name] = Image(
                qvec=qvec, tvec=tvec, name=image_name)
            cnt += 1


    return images


def loadDatasetColmap(base_dir):
    return read_write_model.read_images_text(os.path.join(base_dir, "sparse_gt", "images.txt"))


def loadDataset(base_dir, dataset):
    if dataset in variable.CAMBRIDGE_DATASET:
        return loadDatasetCambridge(base_dir)
    
    if dataset in variable.ENERGY_DATASET:
        return loadDatasetEnergy(base_dir)
    
    if dataset in variable.COLMAP_DATASET:
        return loadDatasetColmap(base_dir)
    
    raise Exception("Not Supported Dataset")


# Normalize target dataset w.r.t. source dataset
def normalize(src_dataset, tar_dataset):
    # Select any image as pivot
    source_keys = list(src_dataset.keys())

    img_names = []

    src_lst = np.zeros((3*len(source_keys), 4))
    tar_lst = np.zeros((3*len(source_keys), 4))

    for i in range(len(source_keys)):
        _name = src_dataset[source_keys[i]].name
        img_names.append(_name)
        
        P_source = np.zeros((3, 4))
        P_source[:, :3] = model.qvec2rotmat(src_dataset[source_keys[i]].qvec)
        P_source[:, 3] = src_dataset[source_keys[i]].tvec

        src_lst[(i*3):(i*3)+3, :] = P_source

        P_target = np.zeros((3, 4))
        P_target[:, :3] = model.qvec2rotmat(tar_dataset[_name].qvec)
        P_target[:, 3] = tar_dataset[_name].tvec

        tar_lst[(i*3):(i*3)+3, :] = P_target

    src_inv_matrix = np.zeros((4, 4))
    src_inv_matrix[:3, :3] = src_lst[:3, :3].T
    src_inv_matrix[:3, 3] = -src_lst[:3, :3].T@src_lst[:3, 3]
    src_inv_matrix[3, 3] = 1

    tar_inv_matrix = np.zeros((4, 4))
    tar_inv_matrix[:3, :3] = tar_lst[:3, :3].T
    tar_inv_matrix[:3, 3] = -tar_lst[:3, :3].T@tar_lst[:3, 3]
    tar_inv_matrix[3, 3] = 1

    res_src = src_lst@src_inv_matrix
    res_tar = tar_lst@tar_inv_matrix

    rot_error = 0
    trans_error = 0
    for i in range(len(source_keys)):
        r_e = Vector.error_r(res_tar[(i*3):(i*3)+3, :3], res_src[(i*3):(i*3)+3, :3])
        rot_error += r_e

        t_e = Vector.error_t(res_tar[(i*3):(i*3)+3, :3], res_src[(i*3):(i*3)+3, :3], res_tar[(i*3):(i*3)+3, 3], res_src[(i*3):(i*3)+3, 3])
        trans_error += t_e

    return rot_error/len(source_keys), trans_error/len(source_keys)
