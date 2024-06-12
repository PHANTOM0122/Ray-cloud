import numpy as np
import poselib
import math
import cv2

pi = math.pi

def error_r(r_pred, r_gt):
    return np.arccos((np.trace(np.dot(np.transpose(r_pred), r_gt)) - 1) / 2)

def error_t(r_pred, r_gt, t_pred, t_gt):
    return np.linalg.norm(np.subtract(np.dot(np.transpose(r_pred), t_pred), np.dot(np.transpose(r_gt), t_gt)))

def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return [qw, qx, qy, qz]

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

def projection(pts,q,center):
  R = convert_to_matrix(q)
  t = - R.T@center.T
  n = pts.shape[0]
  pts = (R@pts.T).T + np.tile(t,[n,1])

  return np.array([pts[:,0]/pts[:,2],pts[:,1]/pts[:,2]]).T, R, t

def homogeneous(x):
    n = x.shape[0]
    homo = np.zeros((n,3))
    homo[:,0] = x[:,0]
    homo[:,1] = x[:,1]
    homo[:,2] = 1
    homo /= np.linalg.norm(homo, axis=1, keepdims=True)

    return homo

def get_query_image_names(query_txt):
    query_img_names = []
    with open(query_txt, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            query_img_names.append(line.strip())
    
    return query_img_names

def get_query_images(query_txt, img_txt):
    query_img_names = get_query_image_names(query_txt)
    
    # Find query ID corresponding to the name
    query_img_id = []
    for k, v in img_txt.items():
        if v.name in query_img_names:
            query_img_id.append(v.id)
  
    return query_img_names, query_img_id

def get_GT_image(img_id, img_txt, img_gt):
    gt_img_name = img_txt[img_id].name
    for k, v in img_gt.items():
        if v.name == gt_img_name:
            return v

def convert_cam(camobj):
    c = {}
    c["model"] = camobj.model
    c["width"] = camobj.width
    c["height"] = camobj.height
    c["params"] = camobj.params
    return c