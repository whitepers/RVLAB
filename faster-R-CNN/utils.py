import numpy as np

"""
tools to convert specified type
"""
import torch as t
import numpy as np


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        return data.detach().cpu().numpy()


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
    if isinstance(data, t.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, t.Tensor):
        return data.item()


def generate_anchors(image_size, sub_sample=16, anchor_scale=[8, 16, 32], ratio=[0.5, 1, 2]):
    len_ratio = len(ratio)
    anchor_base = np.zeros((len(anchor_scale) * len_ratio, 4))  # 9x4

    for idx, scale in enumerate(anchor_scale):
        w = scale / np.sqrt(ratio) * sub_sample
        h = scale * np.sqrt(ratio) * sub_sample
        x1, y1, x2, y2 = -w / 2, -h / 2, w / 2, h / 2

        anchor_base[idx * len_ratio:(idx + 1) * len_ratio] = np.c_[x1, y1, x2, y2]

    feature_map_size = image_size[0] // sub_sample, image_size[1] // sub_sample
    ctr_x = np.arange(sub_sample // 2, image_size[0], sub_sample)
    ctr_y = np.arange(sub_sample // 2, image_size[1], sub_sample)

    ctr = np.zeros((*feature_map_size, 2))
    for idx, y in enumerate(ctr_y):
        ctr[idx, :, 0] = ctr_x
        ctr[idx, :, 1] = y

    anchors = np.zeros((*feature_map_size, *anchor_base.shape))
    for idx_x in range(feature_map_size[0]):
        for idx_y in range(feature_map_size[1]):
            anchors[idx_x, idx_y] = (ctr[idx_x, idx_y] + anchor_base.reshape(-1, 2, 2)).reshape(-1, 4)

    return anchors.reshape(-1, 4)


# bbox iou 계산, (num_of_boxes1, 4) x (num_of_boxes2, 4)
# bboxes_1: anchor, bboxes_2: target box
# shape : x1 x2 y1 y2
def bbox_iou(bboxes_1, bboxes_2):
    len_bboxes_1 = bboxes_1.shape[0]
    len_bboxes_2 = bboxes_2.shape[0]
    ious = np.zeros((len_bboxes_1, len_bboxes_2))

    for idx, bbox_1 in enumerate(bboxes_1):
        yy1_max = np.maximum(bbox_1[1], bboxes_2[:, 1])
        xx1_max = np.maximum(bbox_1[0], bboxes_2[:, 0])
        yy2_min = np.minimum(bbox_1[3], bboxes_2[:, 3])
        xx2_min = np.minimum(bbox_1[2], bboxes_2[:, 2])

        height = np.maximum(0.0, yy2_min - yy1_max)
        width = np.maximum(0.0, xx2_min - xx1_max)

        eps = np.finfo(np.float32).eps
        inter = height * width
        union = (bbox_1[3] - bbox_1[1]) * (bbox_1[2] - bbox_1[0]) + \
                (bboxes_2[:, 3] - bboxes_2[:, 1]) * (bboxes_2[:, 2] - bboxes_2[:, 0]) - inter + eps
        iou = inter / union
        ious[idx] = iou

    return ious  # ious (num_of_boxes1, num_of_boxes2)


# (x1, y1, x2, y2) -> (x, y, w, h) -> (dx, dy, dw, dh)
'''
t_{x} = (x - x_{a})/w_{a}
t_{y} = (y - y_{a})/h_{a}
t_{w} = log(w/ w_a)
t_{h} = log(h/ h_a)
anchors are the anchors
base_anchors are the boxes
'''


def format_loc(anchors, base_anchors):
    width = anchors[:, 2] - anchors[:, 0]
    height = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + width * 0.5
    ctr_y = anchors[:, 1] + height * 0.5

    base_width = base_anchors[:, 2] - base_anchors[:, 0]
    base_height = base_anchors[:, 3] - base_anchors[:, 1]
    base_ctr_x = base_anchors[:, 0] + base_width * 0.5
    base_ctr_y = base_anchors[:, 1] + base_height * 0.5

    eps = np.finfo(np.float32).eps
    height = np.maximum(eps, height)
    width = np.maximum(eps, width)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    anchor_loc_target = np.stack((dx, dy, dw, dh), axis=1)
    return anchor_loc_target


# (dx, dy, dw, dh) -> (x, y, w, h) -> (x1, y1, x2, y2)
'''
anchors are the default anchors
formatted_base_anchors are the boxes with (dy, dx, dh, dw)
'''


def deformat_loc(anchors, formatted_base_anchor):
    width = anchors[:, 2] - anchors[:, 0]
    height = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + width * 0.5
    ctr_y = anchors[:, 1] + height * 0.5

    dx, dy, dw, dh = formatted_base_anchor.T
    base_width = np.exp(dw) * width
    base_height = np.exp(dh) * height
    base_ctr_x = dx * width + ctr_x
    base_ctr_y = dy * height + ctr_y

    base_anchors = np.zeros_like(anchors)
    base_anchors[:, 0] = base_ctr_x - base_width * 0.5
    base_anchors[:, 1] = base_ctr_y - base_height * 0.5
    base_anchors[:, 2] = base_ctr_x + base_width * 0.5
    base_anchors[:, 3] = base_ctr_y + base_height * 0.5

    return base_anchors


# non-maximum-suppression
def nms(rois, scores, nms_thresh):
    # print(scores, scores.shape)
    order = (-scores).argsort().cpu().data.numpy()  # [::-1]
    # x1, y1, x2, y2 = rois.T
    rois = rois.cpu().data.numpy()

    keep_index = []
    # print(order.size)
    while order.size > 0:
        i = order[0]
        keep_index.append(i)
        ious = bbox_iou(rois[i][np.newaxis, :], rois[order[1:]])
        inds = np.where(ious <= nms_thresh)[1]
        order = order[inds + 1]
    return np.asarray(keep_index)