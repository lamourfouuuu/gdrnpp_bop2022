import numpy as np
import math
from scipy import spatial
from scipy.linalg import logm
from numpy import linalg as LA
import open3d as o3d
import argparse
import os
import sys
from plyfile import PlyData, PlyElement
import json
import pandas as pd


def read_diameter(path, obj_number):
    filename = f'{path}/models/models_info.json'
    with open(filename, 'r') as f:
        models_info = json.load(f)
    
    if str(obj_number) in models_info:
        diameter_in_cm = models_info[str(obj_number)]["diameter"]
        return diameter_in_cm
    
    #return diameter_in_cm * 0.01
    return diameter_in_cm


def transform_pts_Rt(pts, R, t):
    """Applies a rigid transformation to 3D points.
  :param pts: nx3 ndarray with 3D points.
  :param R: 3x3 ndarray with a rotation matrix.
  :param t: 3x1 ndarray with a translation vector.
  :return: nx3 ndarray with transformed 3D points.
  """
    assert (pts.shape[1] == 3)
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T


def project_pts(pts, K, R, t):
    """Projects 3D points.
  :param pts: nx3 ndarray with the 3D points.
  :param K: 3x3 ndarray with an intrinsic camera matrix.
  :param R: 3x3 ndarray with a rotation matrix.
  :param t: 3x1 ndarray with a translation vector.
  :return: nx2 ndarray with 2D image coordinates of the projections.
  """
    assert (pts.shape[1] == 3)
    P = K.dot(np.hstack((R, t)))
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_im = P.dot(pts_h.T)
    pts_im /= pts_im[2, :]
    return pts_im[:2, :].T


def add(R_est, t_est, R_gt, t_gt, pts):
    """Average Distance of Model Points for objects with no indistinguishable
  views - by Hinterstoisser et al. (ACCV'12).
  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :param pts: nx3 ndarray with 3D model points.
  :return: The calculated error.
  """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e


def adi(R_est, t_est, R_gt, t_gt, pts):
    """Average Distance of Model Points for objects with indistinguishable views
  - by Hinterstoisser et al. (ACCV'12).
  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :param pts: nx3 ndarray with 3D model points.
  :return: The calculated error.
  """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

    # Calculate distances to the nearest neighbors from vertices in the
    # ground-truth pose to vertices in the estimated pose.
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e


def re(R_est, R_gt):
    """Rotational Error.
  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :return: The calculated error.
  """
    error_cos = float(0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0))

    # Avoid invalid values due to numerical errors.
    error_cos = min(1.0, max(-1.0, error_cos))

    error = math.acos(error_cos)
    error = 180.0 * error / np.pi  # Convert [rad] to [deg].
    return error


def te(t_est, t_gt):
    """Translational Error.
  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :return: The calculated error.
  """
    assert (t_est.size == t_gt.size == 3)
    error = np.linalg.norm(t_gt - t_est)
    return error


def proj(R_est, t_est, R_gt, t_gt, K, pts):
    """Average distance of projections of object model vertices [px]
  - by Brachmann et al. (CVPR'16).
  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :param K: 3x3 ndarray with an intrinsic camera matrix.
  :param pts: nx3 ndarray with 3D model points.
  :return: The calculated error.
  """
    proj_est = project_pts(pts, K, R_est, t_est)
    proj_gt = project_pts(pts, K, R_gt, t_gt)
    e = np.linalg.norm(proj_est - proj_gt, axis=1).mean()
    return e


def compute_add_score(pts3d, diameter, R_pred, t_pred, R_gt, t_gt, percentage=0.1):
    # R_gt, t_gt = pose_gt
    # R_pred, t_pred = pose_pred
    count = R_gt.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]
        pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
        distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
        mean_distances[i] = np.mean(distance)

    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score


def compute_adds_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):
    R_gt, t_gt = pose_gt
    R_pred, t_pred = pose_pred

    count = R_gt.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        if np.isnan(np.sum(t_pred[i])):
            mean_distances[i] = np.inf
            continue
        pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]
        pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
        kdt = spatial.KDTree(pts_xformed_gt.transpose(), metric='euclidean')
        distance, _ = kdt.query(pts_xformed_pred.transpose(), k=1)
        mean_distances[i] = np.mean(distance)
    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score


def compute_pose_error(diameter, pose_gt, pose_pred):
    R_gt, t_gt = pose_gt
    R_pred, t_pred = pose_pred

    count = R_gt.shape[0]
    R_err = np.zeros(count)
    t_err = np.zeros(count)
    for i in range(count):
        if np.isnan(np.sum(t_pred[i])):
            continue
        r_err = logm(np.dot(R_pred[i].transpose(), R_gt[i])) / 2
        R_err[i] = LA.norm(r_err, 'fro')
        t_err[i] = LA.norm(t_pred[i] - t_gt[i])
    return np.median(R_err) * 180 / np.pi, np.median(t_err) / diameter


def cou_mask(mask_est, mask_gt):
    """Complement over Union of 2D binary masks.
  :param mask_est: hxw ndarray with the estimated mask.
  :param mask_gt: hxw ndarray with the ground-truth mask.
  :return: The calculated error.
  """
    mask_est_bool = mask_est.astype(bool)
    mask_gt_bool = mask_gt.astype(bool)

    inter = np.logical_and(mask_gt_bool, mask_est_bool)
    union = np.logical_or(mask_gt_bool, mask_est_bool)

    union_count = float(union.sum())
    if union_count > 0:
        e = 1.0 - inter.sum() / union_count
    else:
        e = 1.0
    return e


def find_nn_idx(pc_src, pc_target):
    """
        pc_src: (N1, 3) array
        pc_target: (N2, 3) array
    """
    dist_sq = -np.dot(pc_src, pc_target.T) * 2 + \
            np.square(np.linalg.norm(pc_src, axis=-1, keepdims=True)) + \
            np.square(np.linalg.norm(pc_target.T, axis=0, keepdims=True))
    idx_min = np.argmin(dist_sq, axis=0)
    return idx_min


def add_metric(pose_pred, pose_targets, obj_points, diameter, symm=False, percentage=0.1, gpu=False):

    diam = diameter * percentage
    model_pred = np.dot(obj_points, pose_pred[:, :3].T) + pose_pred[:, 3]
    model_targets = np.dot(obj_points, pose_targets[:, :3].T) + pose_targets[:, 3]

    if symm:
        # if gpu:
        #     idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
        # else:
        idxs = find_nn_idx(model_pred, model_targets)
        mean_dist = np.mean(np.linalg.norm(model_pred[idxs] - model_targets, 2, 1))
    else:
        mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))

    return mean_dist < diam, (mean_dist, diam)


def eval_pose(r_est, t_est, r_gt, t_gt, pc, k, diameter, sym=False):
    add_res = add(r_est, t_est, r_gt, t_gt, pc)
    is_add = add(r_est, t_est, r_gt, t_gt, pc) < diameter * 0.1
    proj_res = proj(r_est, t_est, r_gt, t_gt, k, pc)
    is_proj = proj(r_est, t_est, r_gt, t_gt, k, pc) < 5
    adi_res = 0
    is_adi = False
    if sym:
        add_res = adi(r_est, t_est, r_gt, t_gt, pc)
        is_add = adi(r_est, t_est, r_gt, t_gt, pc) < diameter * 0.1
        # print("add_res", add_res, "is_add", is_add)

    return add_res, is_add, proj_res, is_proj, adi_res, is_adi


def load_predicted_csv(fname):
    df = pd.read_csv(fname)
    info_list = df.to_dict("records")
    return info_list   


def parse_Rt_in_csv(_item):
    return np.array([float(i) for i in _item.strip(" ").split(" ")])


def load_gt(json_path, obj_ids):
    with open(json_path, 'r') as f:
        gt_data = json.load(f)
    
    filtered_gt_data = {}
    for obj_id in obj_ids:
        if obj_id in gt_data:
            filtered_gt_data[obj_id] = gt_data[obj_id]
        else:
            print(f"Warning: {obj_id} not in gt_data")
    return filtered_gt_data


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_data", type=str, required=True)
    ap.add_argument("--pred_path", type=str, required=True)
    ap.add_argument("-cls_name", "--class_name", type=str,
                    default='kiwi1',
                    help="[apple2, apricot, banana1, kiwi1, lemon2, orange2, peach1, pear2]")
    ap.add_argument("-sym", "--symmetry", type=bool,
                    default=False)

    args = vars(ap.parse_args())
 
    class_name = args["class_name"]
    symmetry = args["symmetry"]
    path_data = args["path_data"]
    pred_path = args["pred_path"]

    id2obj = {
    1: "apple2",
    2: "apricot",
    3: "banana1",
    4: "kiwi1",
    5: "lemon2",
    6: "orange2",
    7: "peach1",
    8: "pear2"
    }

    obj_number = list(id2obj.keys())[list(id2obj.values()).index(class_name)]

    basePath = args["path_data"] + "/test/{:06d}/".format(obj_number - 1)
    print(basePath)

    pc_path = '/gdrnpp_bop2022/datasets/BOP_DATASETS/fruitbin/models/obj_{:06d}.ply'.format(obj_number)

    plydata = PlyData.read(pc_path)
    elm = plydata.elements
    data = np.asarray(elm[0][:])
    pc = np.zeros((len(data), 3))
    print("len(data)", len(data))
    print("len(pc)", len(pc))

    diameter = read_diameter(path_data, obj_number)
    for i in range(len(data)):
        pc[i][0], pc[i][1], pc[i][2] = data[i][0], data[i][1], data[i][2]


    add_res_ls = []
    proj_res_ls = []
    count_add = 0
    count_iadd = 0
    count_proj = 0

    #  ============== Loading Pose ===============
    preds_csv = load_predicted_csv(pred_path)
    preds = {}
    obj_ids = []
    for item in preds_csv:
        if item["obj_id"] == obj_number:
            im_key = "{}".format(item["im_id"])
            item["R"] = parse_Rt_in_csv(item["R"]).reshape(3, 3)
            item["t"] = parse_Rt_in_csv(item["t"]).reshape(3, 1)
            if im_key not in preds:
                preds[im_key] = []
            preds[im_key].append(item)
            obj_ids.append(str(item["im_id"]))

    length_data=len(preds)
    print("number of evaluating data :", length_data)

    gt_data = load_gt(f"{basePath}scene_gt.json", obj_ids)

    for im_id, items in preds.items():
        gt_items = gt_data.get(im_id, [])
        for item in items:
            if gt_items:
                item["gt_R"] = np.array(gt_items[0]["cam_R_m2c"]).reshape(3, 3)
                item["gt_t"] = np.array(gt_items[0]["cam_t_m2c"]).reshape(3, 1)

    #  ============== Evaluation ===============

    k = np.array([[543.25272224, 0., 320.25],
                    [0., 724.33696299, 240.33333333],
                    [0., 0., 1.]])    

    for im_id, items in preds.items():
        for item in items:
            r_est, t_est = item["R"], item["t"]
            r_gt, t_gt = item.get("gt_R"), item.get("gt_t")
            
            add_res, is_add, proj_res, is_proj, adi_res, is_adi = eval_pose(r_est, t_est, r_gt, t_gt, pc, k, diameter, symmetry)
            if is_add:
                count_add += 1
                # print("count_add", count_add, "/", length_data)
            if is_proj:
                count_proj += 1
            if is_adi:
                count_iadd += 1

    print("results for class : ", class_name)
    print(f"ADD_Res: {count_add / length_data}")
    print(f"ADI_Res: {count_iadd / length_data}")
    print(f"Proj_Res: {count_proj / length_data}")
    print(f"======================")

    print("Done")
