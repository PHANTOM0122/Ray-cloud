import numpy as np
import math
import poselib

def calculate_loss(image_info, res):
    # print("-"*35)
    q_gt = image_info.qvec
    t_gt = image_info.tvec
    
    q_pred = np.array(res[0].q)
    t_pred = np.array(res[0].t)
    # print('GT Rotation :', q_gt)
    # print('GT translation :', t_gt)
    # print('Pred Rotation :', q_pred)
    # print('Pred translation :', t_pred)

    # Normalized for 
    q_gt /= np.linalg.norm(q_gt)
    q_pred /= np.linalg.norm(q_pred)

    R_gt = convert_to_matrix(q_gt)
    R_pred = convert_to_matrix(q_pred)
    rot_error = error_r(R_pred, R_gt)
    # print("Rotational Error:", rot_error)

    trans_error = error_t(R_pred, R_gt, t_pred, t_gt)
    # print("Transitional Error", trans_error)
    
    return rot_error, trans_error

def error_r(r_pred, r_gt):
  return np.arccos((np.trace(np.dot(np.transpose(r_pred), r_gt)) - 1) / 2)

def error_t(r_pred, r_gt, t_pred, t_gt):
  return np.linalg.norm(np.subtract(np.dot(np.transpose(r_pred), t_pred), np.dot(np.transpose(r_gt), t_gt)))

def convert_to_matrix(q):
    # Rotation is represented as a unit quaternion
    # with real part first, i.e. QW, QX, QY, QZ

    q0, q1, q2, q3 = q

    R = np.zeros((3, 3), dtype=float)

    R[0][0] = 1 - 2*(q2*q2 + q3*q3)
    R[0][1] = 2*(q1*q2 - q0*q3)
    R[0][2] = 2*(q1*q3 + q0*q2)

    R[1][0] = 2*(q1*q2 + q0*q3)
    R[1][1] = 1 - 2*(q1*q1 + q3*q3)
    R[1][2] = 2*(q2*q3 - q0*q1)
    
    R[2][0] = 2*(q1*q3 - q0*q2)
    R[2][1] = 2*(q2*q3 + q0*q1)
    R[2][2] = 1 - 2*(q1*q1 + q2*q2)

    return R


def homogeneous(x):
    n = x.shape[0]
    homo = np.zeros((n,3))
    homo[:,0] = x[:,0]
    homo[:,1] = x[:,1]
    homo[:,2] = 1
    homo /= np.linalg.norm(homo, axis=1, keepdims=True)

    return homo

def rotate_quartenion(q, p):
    q1, q2, q3, q4 = q
    p1, p2, p3 = p
    px1 = -p1 * q2 - p2 * q3 - p3 * q4
    px2 = p1 * q1 - p2 * q4 + p3 * q3
    px3 = p2 * q1 + p1 * q4 - p3 * q2
    px4 = p2 * q2 - p1 * q3 + p3 * q1
    return np.array([px2 * q1 - px1 * q2 - px3 * q4 + px4 * q3, px3 * q1 - px1 * q3 + px2 * q4 - px4 * q2,
                           px3 * q2 - px2 * q3 - px1 * q4 + px4 * q1])


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

def projection(pts,q,center):
    R = convert_to_matrix(q)
    t = - R.T@center.T
    n = pts.shape[0]
    pts = (R@pts.T).T + np.tile(t,[n,1])

    return np.array([pts[:,0]/pts[:,2],pts[:,1]/pts[:,2]]).T, R, t


def check_cheirality(q,t,p1,x1,p2,x2):
    min_depth = 0.01
    Rx1 = rotate_quartenion(q, x1)
    rhs = t+rotate_quartenion(q, p1) - p2
    a = -Rx1@x2
    b1 = -Rx1@rhs
    b2 = x2@rhs
    
    lambda1 = b1 -a*b2
    lambda2 = -a*b1 + b2
    
    min_depth *= (1 - a*a)
    
    return lambda2 > min_depth
  
def make_matches(points2D, lines3D, virtual_cam_id):

  pair_matches = poselib.PairwiseMatches()

  # 3D rays to virtual normalized plane, assuming no distortion
  points2D_ray = lines3D / lines3D[:, 2].reshape(-1,1)

  pair_matches.x1 = points2D_ray[:, :2]
  pair_matches.x2 = points2D

  pair_matches.cam_id1 = virtual_cam_id
  pair_matches.cam_id2 = 0

  return pair_matches
