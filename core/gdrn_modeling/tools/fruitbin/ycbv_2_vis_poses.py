import mmcv
import os.path as osp
import numpy as np
import sys
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import torch
import pandas as pd

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))

from lib.vis_utils.colormap import colormap
from lib.utils.mask_utils import mask2bbox_xyxy, cocosegm2mask, get_edge
from core.utils.data_utils import crop_resize_by_warp_affine, read_image_mmcv
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
from transforms3d.quaternions import quat2mat
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer


out_size = 512
score_thr = 0.3
colors = colormap(rgb=False, maximum=255)

# object info
id2obj = {
        3: "banana1",
}
objects = list(id2obj.values())


def load_predicted_csv(fname):
    df = pd.read_csv(fname)
    info_list = df.to_dict("records")
    return info_list


def parse_Rt_in_csv(_item):
    return np.array([float(i) for i in _item.strip(" ").split(" ")])


tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
image_tensor = torch.empty((out_size, out_size, 4), **tensor_kwargs).detach()
seg_tensor = torch.empty((out_size, out_size, 4), **tensor_kwargs).detach()
# image_tensor = torch.empty((480, 640, 4), **tensor_kwargs).detach()

model_dir = "datasets/BOP_DATASETS/fruitbin/models/"

model_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.ply") for obj_id in id2obj]
texture_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.png") for obj_id in id2obj]

ren = EGLRenderer(
    model_paths,
    texture_paths=texture_paths,
    vertex_scale=0.001,
    use_cache=True,
    width=out_size,
    height=out_size,
)

# NOTE: this is for fruitbin_bop_test
pred_path = "/gdrnpp_bop2022/output/gdrn/fruitbin/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_fruitbin/inference_/fruitbin_test/convnext-a6-AugCosyAAEGray-BG05-mlL1-DMask-amodalClipBox-classAware-fruitbin-iter0_fruitbin-test.csv"

vis_dir = "/gdrnpp_bop2022/output/gdrn/fruitbin/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_fruitbin/inference_/vis_gt_pred"
mmcv.mkdir_or_exist(vis_dir)

print(pred_path)
preds_csv = load_predicted_csv(pred_path)
preds = {}
for item in preds_csv:
    im_key = "{}/{}".format(item["scene_id"], item["im_id"])
    item["time"] = float(item["time"])
    item["score"] = float(item["score"])
    item["R"] = parse_Rt_in_csv(item["R"]).reshape(3, 3)
    item["t"] = parse_Rt_in_csv(item["t"]) / 1000
    item["obj_name"] = id2obj[item["obj_id"]]
    if im_key not in preds:
        preds[im_key] = []
    preds[im_key].append(item)

dataset_name = "fruitbin_test"
print(dataset_name)
register_datasets([dataset_name])

meta = MetadataCatalog.get(dataset_name)
print("MetadataCatalog: ", meta)
objs = meta.objs

dset_dicts = DatasetCatalog.get(dataset_name)
for d in tqdm(dset_dicts):
    K = d["cam"]
    file_name = d["file_name"]
    scene_im_id = d["scene_im_id"]
    img = read_image_mmcv(file_name, format="BGR")

    scene_im_id_split = d["scene_im_id"].split("/")
    scene_id = scene_im_id_split[0]
    im_id = int(scene_im_id_split[1])

    imH, imW = img.shape[:2]
    annos = d["annotations"]
    masks = [cocosegm2mask(anno["segmentation"], imH, imW) for anno in annos]
    fg_mask = sum(masks).astype("bool").astype("uint8")
    minx, miny, maxx, maxy = mask2bbox_xyxy(fg_mask)

    bboxes = [anno["bbox"] for anno in annos]
    bbox_modes = [anno["bbox_mode"] for anno in annos]
    bboxes_xyxy = np.array(
        [BoxMode.convert(box, box_mode, BoxMode.XYXY_ABS) for box, box_mode in zip(bboxes, bbox_modes)]
    )

    quats = [anno["quat"] for anno in annos]
    transes = [anno["trans"] for anno in annos]
    Rs = [quat2mat(quat) for quat in quats]
    # 0-based label
    cat_ids = [anno["category_id"] for anno in annos]
    obj_names = [objs[cat_id] for cat_id in cat_ids]

    gt_Rs = []
    gt_ts = []
    gt_labels = []

    for anno_i, anno in enumerate(annos):
        obj_name = obj_names[anno_i]
        gt_labels.append(objects.index(obj_name))  # 0-based label

        gt_Rs.append(Rs[anno_i])
        gt_ts.append(transes[anno_i])

    if scene_im_id not in preds:
        print(scene_im_id, "not detected")
        continue
    cur_preds = preds[scene_im_id]
    kpts_2d_est = []
    est_Rs = []
    est_ts = []
    est_labels = []
    for pred_i, pred in enumerate(cur_preds):
        try:
            R_est = pred["R"]
            t_est = pred["t"]
            score = pred["score"]
            obj_name = pred["obj_name"]
        except:
            continue
        if score < score_thr:
            continue

        est_Rs.append(R_est)
        est_ts.append(t_est)
        est_labels.append(objects.index(obj_name))  # 0-based label

    center = np.array([(minx + maxx) / 2, (miny + maxy) / 2])
    scale = max(maxx - minx, maxy - miny) * 1.5  # + 10
    crop_minx = max(0, center[0] - scale / 2)
    crop_miny = max(0, center[1] - scale / 2)
    crop_maxx = min(imW - 1, center[0] + scale / 2)
    crop_maxy = min(imH - 1, center[1] + scale / 2)
    scale = min(scale, min(crop_maxx - crop_minx, crop_maxy - crop_miny))

    zoomed_im = crop_resize_by_warp_affine(img, center, scale, out_size)
    im_zoom_gray = mmcv.bgr2gray(zoomed_im, keepdim=True)
    im_zoom_gray_3 = np.concatenate([im_zoom_gray, im_zoom_gray, im_zoom_gray], axis=2)
    # print(im_zoom_gray.shape)
    K_zoom = K.copy()
    K_zoom[0, 2] -= center[0] - scale / 2
    K_zoom[1, 2] -= center[1] - scale / 2
    K_zoom[0, :] *= out_size / scale
    K_zoom[1, :] *= out_size / scale

    gt_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(gt_Rs, gt_ts)]
    est_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(est_Rs, est_ts)]

    ren.render(
        est_labels,
        est_poses,
        K=K_zoom,
        image_tensor=image_tensor,
        background=im_zoom_gray_3,
    )
    ren_bgr = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype("uint8")

    for gt_label, gt_pose in zip(gt_labels, gt_poses):
        ren.render([gt_label], [gt_pose], K=K_zoom, seg_tensor=seg_tensor)
        gt_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")
        gt_edge = get_edge(gt_mask, bw=3, out_channel=1)
        ren_bgr[gt_edge != 0] = np.array(mmcv.color_val("blue"))

    for est_label, est_pose in zip(est_labels, est_poses):
        ren.render([est_label], [est_pose], K=K_zoom, seg_tensor=seg_tensor)
        est_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")
        est_edge = get_edge(est_mask, bw=3, out_channel=1)
        ren_bgr[est_edge != 0] = np.array(mmcv.color_val("green"))

    vis_im = ren_bgr

    # vis_im_add = (im_zoom_gray_3 * 0.3 + ren_bgr * 0.7).astype("uint8")

    save_path_0 = osp.join(vis_dir, "{}_{:06d}_vis0.png".format(scene_id, im_id))
    mmcv.imwrite(zoomed_im, save_path_0)

    save_path_1 = osp.join(vis_dir, "{}_{:06d}_vis1.png".format(scene_id, im_id))
    mmcv.imwrite(vis_im, save_path_1)

    # if True:
    #     # grid_show([zoomed_im[:, :, ::-1], vis_im[:, :, ::-1]], ["im", "est"], row=1, col=2)
    #     # im_show = cv2.hconcat([zoomed_im, vis_im, vis_im_add])
    #     im_show = cv2.hconcat([zoomed_im, vis_im])
    #     cv2.imshow("im_est", im_show)
    #     if cv2.waitKey(0) == 27:
    #         break  # esc to quit
