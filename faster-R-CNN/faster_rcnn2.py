import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import RoIPool

import numpy as np

from utils import *

# Backbone
from backbone import get_bb_clf

# bbox = torch.FloatTensor([[30,20,500,400], [400,300,600,500]])
# labels = torch.LongTensor([6, 8])
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np

from utils import *
from creator_tools import *


# bbox = torch.FloatTensor([[30,20,500,400], [400,300,600,500]])
# labels = torch.LongTensor([6, 8])

# RPN
class RPN(nn.Module):
    def __init__(
            self, in_c=512, mid_c=512,
            image_size=(800, 800), sub_sample=16,
            anchor_scale=[8, 16, 32], ratio=[0.5, 1, 2],
    ):
        super(RPN, self).__init__()

        self.rpn = nn.Conv2d(in_c, mid_c, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        n_anchor = len(anchor_scale) * len(ratio)

        self.reg = nn.Conv2d(mid_c, n_anchor * 4, 1, 1, 0)
        self.cls = nn.Conv2d(mid_c, n_anchor * 2, 1, 1, 0)

        self.anchor_base = generate_anchors(image_size, sub_sample=sub_sample,
                                            anchor_scale=anchor_scale, ratio=ratio)
        self.proposal_layer = ProposalCreator(self)

        weight_init(self.rpn)
        weight_init(self.reg)
        weight_init(self.cls)

    # x : feature map
    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape
        anchor = self.anchor_base
        n_anchor = anchor.shape[0] // (h * w)  # 9

        x = self.rpn(x)
        x = self.relu(x)
        pred_loc = self.reg(x)  # batch, anchor*4, height, width
        pred_cls = self.cls(x)  # batch, anchor*2, height, width

        pred_loc = pred_loc.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  # batch anchors (coor)
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous()  # batch anchors (obj)
        pred_sfmax_cls = F.softmax(pred_cls.view(n, h, w, n_anchor, 2), dim=4)
        pred_fg_cls = pred_sfmax_cls[:, :, :, :, 1].contiguous()
        pred_fg_cls = pred_fg_cls.view(n, -1)
        pred_cls = pred_cls.view(n, -1, 2)

        pred_object = pred_cls[:, :, 1]
        rois = []
        roi_indices = []
        for i in range(n):
            roi = self.proposal_layer(
                pred_loc[i].cpu().data.numpy(),
                pred_fg_cls[i].cpu().data.numpy(),
                anchor, img_size, scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return pred_loc, pred_cls, rois, roi_indices, anchor


# target_loc, target_cls = assign_cls_loc로 만든 것
def rpn_loss(pred_loc, pred_cls, target_loc, target_cls, rpn_lamda=10):
    # cls loss
    # print(pred_cls.shape)
    gt_rpn_cls = torch.from_numpy(target_cls).long().to('cuda:0')
    pred_rpn_cls = pred_cls[0].to('cuda:0')
    # print(pred_rpn_cls.shape, gt_rpn_cls.shape)
    rpn_cls_loss = F.cross_entropy(pred_rpn_cls, gt_rpn_cls, ignore_index=-1)

    # reg loss
    gt_rpn_loc = torch.from_numpy(target_loc).to('cuda:0')
    pred_rpn_loc = pred_loc[0].to('cuda:0')

    mask = gt_rpn_cls > 0
    mask_gt_loc = gt_rpn_loc[mask]
    mask_pred_loc = pred_rpn_loc[mask]

    x = torch.abs(mask_gt_loc - mask_pred_loc)
    rpn_loc_loss = ((x < 0.5).float() * (x ** 2) * 0.5 + (x > 0.5).float() * (x - 0.5)).sum()

    N_reg = mask.float().sum()
    rpn_loss = rpn_cls_loss + rpn_lamda / N_reg * rpn_loc_loss

    return rpn_cls_loss, rpn_loc_loss, rpn_loss


# class RoIHead(nn.Module):
#     def __init__(self, n_class, roi_size, spatial_scale, classifier):
#         super(RoIHead, self).__init__()

#         self.classifier = classifier
#         self.cls_loc

class FastRCNN(nn.Module):
    def __init__(self, classifier, n_class=21, size=(7, 7), spatial_scale=(1. / 16)):
        super(FastRCNN, self).__init__()
        self.roi = RoIPool(size, spatial_scale)
        self.roi_pool = nn.AdaptiveMaxPool2d(size)
        self.classifier = classifier

        self.reg = nn.Linear(4096, n_class * 4)
        weight_init(self.reg)
        self.cls = nn.Linear(4096, n_class)
        weight_init(self.cls)

    def forward(self, feature_map, rois, roi_indices):
        # correspond to feature map
        roi_indices = totensor(roi_indices).float()
        rois = totensor(rois).float()
        indices_rois = t.cat([roi_indices[:, None], rois], dim=1).contiguous()

        pool = self.roi(feature_map, indices_rois)
        pool = pool.view(pool.size(0), -1)

        x = self.classifier(pool)
        roi_loc = self.reg(x)
        roi_cls = self.cls(x)

        return roi_loc, roi_cls


# gt_loc = torch.from_numpy(final_rois).float()
# gt_cls = torch.from_numpy(final_cls).long()
def fastrcnn_loss(roi_loc, roi_cls, gt_loc, gt_cls):  # [128, 84], [128, 21], [128, 4] torch float, [128, 1] torch long
    roi_cls = roi_cls.to('cuda:0')
    gt_cls = gt_cls.to('cuda:0')
    roi_loc = roi_loc.to('cuda:0')
    gt_loc = torch.from_numpy(gt_loc).float().to('cuda:0')
    # print(roi_cls)
    # print(roi_cls.shape, gt_cls.shape, roi_loc.shape, gt_loc.shape)
    cls_loss = F.cross_entropy(roi_cls, gt_cls)
    # print(cls_loss)

    num_roi = roi_loc.size(0)
    roi_loc = roi_loc.view(-1, 21, 4)
    roi_loc = roi_loc[torch.arange(num_roi), gt_cls]

    mask = gt_cls > 0
    mask_loc_pred = roi_loc[mask]
    mask_loc_target = gt_loc[mask]

    x = torch.abs(mask_loc_pred - mask_loc_target)
    loc_loss = ((x < 0.5).float() * x ** 2 * 0.5 + (x > 0.5).float() * (x - 0.5)).sum()

    # print(loc_loss)

    roi_lamda = 10
    N_reg = (gt_cls > 0).float().sum()
    roi_loss = cls_loss + roi_lamda / N_reg * loc_loss

    return cls_loss, loc_loss, roi_loss


class FasterRCNN(nn.Module):
    def __init__(self, backbone, rpn, head):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.head = head  # Fast R-CNN
        self.proposal_target_creator = ProposalTargetCreator()

    def forward(self, img, bboxes, labels):
        b, c, h, w = img.shape

        ##### backbone
        feature_map = self.backbone(img)

        ##### RPN
        # anchors = generate_anchors((w, h))
        # target_cls, target_loc = assign_cls_loc(bboxes, anchors, (w, h))
        pred_loc, pred_cls, rois, roi_indices, anchor = self.rpn(feature_map, (w, h), scale=1.)
        target_cls, target_loc = assign_cls_loc(bboxes, anchor, (w, h))
        rpn_loc_loss, rpn_cls_loss, t_rpn_loss = rpn_loss(pred_loc, pred_cls, target_loc, target_cls)
        # pred_loc, pred_cls, pred_object =
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(rois, bboxes, labels)
        sample_roi_index = t.zeros(len(sample_roi))
        ##### HEAD - Fast RCNN
        final_loc, final_cls = self.head(feature_map, sample_roi, sample_roi_index)

        roi_cls_loss, roi_loc_loss, t_roi_loss = fastrcnn_loss(final_loc, final_cls, gt_roi_loc, gt_roi_label)
        t_loss = torch.sum(t_roi_loss + t_rpn_loss)
        return rpn_loc_loss, rpn_cls_loss, roi_cls_loss, roi_loc_loss, t_loss

        # post_train_rois, post_train_scores = generate_proposal(anchors, pred_loc, pred_cls, pred_object, (w, h))
        # final_rois, final_cls = assign_targets(post_train_rois, post_train_scores, bboxes, labels)
        # final_rois, final_cls = torch.from_numpy(final_rois).float(), torch.from_numpy(final_cls).long()
        # rois = torch.from_numpy(final_rois).float()
        # roi_loc, roi_cls = self.fastrcnn(feature_map, final_rois)

        # gt_loc = final_rois
        # gt_cls = final_cls

        # return final_loc, final_cls, rois, roi_indices


def fasterrcnn_loss(rpn_loss, roi_loss):
    return torch.sum(rpn_loss + roi_loss)


class FasterRCNNSEMob(FasterRCNN):
    down_size = 16

    def __init__(self, n_fg_class=20, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        backbone, classifier = get_bb_clf()
        rpn = RPN()
        head = FastRCNN(classifier, n_class=n_fg_class + 1, spatial_scale=(1. / 16))

        super(FasterRCNNSEMob, self).__init__(backbone, rpn, head)


def assign_cls_loc(bboxes, anchors, image_size, pos_thres=0.7, neg_thres=0.3, n_sample=256, pos_ratio=0.5):
    valid_idx = np.where((anchors[:, 0] >= 0)
                         & (anchors[:, 1] >= 0)
                         & (anchors[:, 2] <= image_size[0])
                         & (anchors[:, 3] <= image_size[1]))[0]
    # print(valid_idx.shape)

    valid_cls = np.empty((valid_idx.shape[0],), dtype=np.int32)
    valid_cls.fill(-1)

    valid_anchors = anchors[valid_idx]

    ious = bbox_iou(valid_anchors, bboxes.numpy())
    # print(ious.shape) # 8940, 2

    # valid cls에 positive로 판단하는 것이 총 두 시나리오에 의해 생성됨
    # a
    iou_by_anchor = np.amax(ious, axis=1)  # anchor별 최대값
    pos_idx = np.where(iou_by_anchor >= pos_thres)[0]
    neg_idx = np.where(iou_by_anchor < neg_thres)[0]
    valid_cls[pos_idx] = 1
    valid_cls[neg_idx] = 0
    # b
    iou_by_gt = np.amax(ious, axis=0)  # gt box별 최대 값
    gt_idx = np.where(ious == iou_by_gt)[0]
    # print(gt_idx)
    valid_cls[gt_idx] = 1

    total_n_pos = len(np.where(valid_cls == 1)[0])
    n_pos = int(n_sample * pos_ratio) if total_n_pos > n_sample * pos_ratio else total_n_pos
    n_neg = n_sample - n_pos

    # valid label에서 256개 넘는 것은 제외
    pos_index = np.where(valid_cls == 1)[0]
    # print(pos_index, len(pos_index, n_pos))
    if len(pos_index) > n_sample * pos_ratio:
        disable_index = np.random.choice(pos_index, size=len(pos_index) - n_pos, replace=False)
        valid_cls[disable_index] = -1
    neg_index = np.where(valid_cls == 0)[0]
    disable_index = np.random.choice(neg_index, size=len(neg_index) - n_neg, replace=False)
    valid_cls[disable_index] = -1

    # 최종 valid class (object or not)
    # print(len(np.where(valid_cls==1)[0]), len(np.where(valid_cls==0)[0]))

    # valid loc
    # Anchor별로 iou가 더 높은쪽으로 loc 분배
    argmax_iou = np.argmax(ious, axis=1)
    max_iou_box = bboxes[argmax_iou].numpy()  # valid_anchors와 shape 같아야함

    valid_loc = format_loc(valid_anchors, max_iou_box)
    # print(valid_loc.shape) # 8940, 4 dx dy dw dh

    # 기존 anchor에서 valid index에 지금까지 구한 valid label (pos, neg 18, 238) 할당
    target_cls = np.empty((len(anchors),), dtype=np.int32)
    target_cls.fill(-1)
    target_cls[valid_idx] = valid_cls

    # 기존 anchor에서 valid index에 지금까지 구한 dx, dy, dw, dh 할당
    target_loc = np.zeros((len(anchors), 4), dtype=np.float32)
    target_loc[valid_idx] = valid_loc

    # print(target_cls.shape)
    # print(target_loc.shape)

    return target_cls, target_loc


# for Fast RCNN
def generate_proposal(anchors, pred_loc, pred_cls, pred_object, image_size,
                      n_train_pre_nms=12000,
                      n_train_post_nms=2000,
                      n_test_pre_nms=6000,
                      n_test_post_nms=300,
                      min_size=16, nms_thresh=0.7):
    rois = deformat_loc(anchors=anchors, formatted_base_anchor=pred_loc[0].cpu().data.numpy())
    np.where(rois[:, 0])
    rois[:, [0, 2]] = np.clip(rois[:, [0, 2]], a_min=0, a_max=image_size[0])  # x [0 ~ 800] width
    rois[:, [1, 3]] = np.clip(rois[:, [1, 3]], a_min=0, a_max=image_size[1])  # y [0 ~ 800] height
    w = rois[:, 2] - rois[:, 0]
    h = rois[:, 3] - rois[:, 1]

    valid_idx = np.where((h > min_size) & (w > min_size))[0]
    valid_rois = rois[valid_idx]
    valid_scores = pred_object[0][valid_idx].cpu().data.numpy()

    order_idx = valid_scores.ravel().argsort()[::-1]

    pre_train_idx = order_idx[:n_train_pre_nms]

    pre_train_rois = valid_rois[pre_train_idx]
    pre_train_scores = valid_scores[pre_train_idx]

    keep_index = nms(rois=pre_train_rois, scores=pre_train_scores, nms_thresh=nms_thresh)
    post_train_rois = pre_train_rois[keep_index][:n_train_post_nms]
    post_train_scores = pre_train_scores[keep_index][:n_train_post_nms]

    return post_train_rois, post_train_scores


def assign_targets(post_train_rois, post_train_scores, bboxes, labels,
                   n_sample=128,
                   pos_ratio=0.25,
                   pos_thresh=0.5,
                   neg_thresh_hi=0.5,
                   neg_thresh_lo=0.0):
    ious = bbox_iou(post_train_rois, bboxes.numpy())

    # cls
    bbox_idx = ious.argmax(axis=1)
    box_max_ious = ious.max(axis=1)
    final_cls = labels[bbox_idx]  # 2000, Object Class 값 들어감

    total_n_pos = len(np.where(box_max_ious >= pos_thresh)[0])
    n_pos = int(n_sample * pos_ratio) if total_n_pos > n_sample * pos_ratio else total_n_pos
    n_neg = n_sample - n_pos

    pos_index = np.where(box_max_ious >= pos_thresh)[0]
    pos_index = np.random.choice(pos_index, size=n_pos, replace=False)

    neg_index = np.where((box_max_ious < neg_thresh_hi) & (box_max_ious >= neg_thresh_lo))[0]
    neg_index = np.random.choice(neg_index, size=n_neg, replace=False)

    keep_index = np.append(pos_index, neg_index)
    final_cls = final_cls[keep_index].data.numpy()
    final_cls[len(pos_index):] = 0

    final_rois = post_train_rois[keep_index]
    post_sample_bbox = bboxes[bbox_idx[keep_index]]

    d_rois = format_loc(anchors=final_rois, base_anchors=post_sample_bbox.data.numpy())

    return final_rois, final_cls


def weight_init(l):
    if type(l) in [nn.Conv2d]:
        l.weight.data.normal_(0, 0.01)
        l.bias.data.zero_()


if __name__ == "__main__":
    pass