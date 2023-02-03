import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import *
import matplotlib.pyplot as plt
import cv2


#샘플 이미지를 통한 특징 추출
image = torch.zeros((1,3,800,800)).float()
image_size = (800,800)

#bbox -> x1,x2,y1, y2 = x,y,w,h
bbox = torch.FloatTensor([[20, 30, 400, 500], [300, 400 ,500, 600]])
labels = torch.LongTensor([6,8])

sub_sample = 16 #down sampling이 얼마나 되는지 800에서 50으로 줄어 듦 -> sub)sample =16

vgg16 = torchvision.models.vgg16(pretrained=True)
req_features = vgg16.features[:30] #vgg를 통해 나온 특징 저장
print(req_features)
output_map = req_features(image) # Conv Layer을 지나서 나와 사전에 만든 sample image의 특징 추출
print(output_map.shape)
plt.scatter(1000, 1000, marker='^')
plt.show()




#Anchor 박스 생성 3가지 scale , Aspect Ratio(H/W)비율 사용
anchor_scale = [8,16,32]
ratio = [0.5, 1, 2] #H/W

len_anchor_scale = len(anchor_scale)
len_ratio = len(ratio)
len_anchor_template = len_anchor_scale * len_ratio   # 3x3
anchor_template = np.zeros((9,4)) #anchor_template 생성

#Anchor scale과 ratio로 box 판단 모양 만들어준다. anchor 하나당 scale3 ratio3 =9개
for idx, scale in enumerate(anchor_scale):
    h = scale * np.sqrt(ratio) * sub_sample
    w = scale / np.sqrt(ratio) * sub_sample
    y1 = -h/2
    x1 = -w/2
    y2 = h/2
    x2 = w/2
    anchor_template[idx*len_ratio:(idx+1)*len_ratio, 0] = y1
    anchor_template[idx*len_ratio:(idx+1)*len_ratio, 1] = x1
    anchor_template[idx*len_ratio:(idx+1)*len_ratio, 2] = y2
    anchor_template[idx*len_ratio:(idx+1)*len_ratio, 3] = x2
print(anchor_template)





# anchor_template가 위치한 점들을 가져옴 50*50
feature_map_size = (50,50)
ctr_y = np.arange(8, 800, 16) # 50개를 8*1, 8*3, 8*5 ...
ctr_x = np.arange(8, 800, 16) # 50개를 8*1, 8*3, 8*5 ...

# 50 * 50개의 anchor center 생성
ctr = np.zeros((*feature_map_size, 2))
for idx, y in enumerate(ctr_y):
    ctr[idx, :, 0] = y
    ctr[idx, :, 1] = ctr_x




#위의 점들을 50x50개의 anchor center 생성
anchors = np.zeros((*feature_map_size, 9, 4))
for idx_y in range(feature_map_size[0]):
    for idx_x in range(feature_map_size[1]):
        anchors[idx_y, idx_x] = (ctr[idx_y, idx_x] + anchor_template.reshape(-1, 2, 2)).reshape(-1, 4)

anchors = anchors.reshape(-1, 4)
print(anchors.shape) # (22500, 4)







#anchor box labeling for RPN / 이미지를 넘어가는 박스 제외
valid_index = np.where((anchors[:, 0] >= 0)
                      &(anchors[:, 1] >= 0)
                      &(anchors[:, 2] <= 800)
                      &(anchors[:, 3] <= 800))[0]
print(valid_index.shape) # 8940

valid_labels = np.empty((valid_index.shape[0],), dtype=np.int32)
valid_labels.fill(-1)
valid_anchors = anchors[valid_index]
print(valid_anchors.shape) # (8940,4)
print(bbox.shape) # torch.Size([2,4])







ious = bbox_iou(valid_anchors, bbox.numpy()) # anchor 8940 : bbox 2
pos_iou_thres = 0.7
neg_iou_thred = 0.3

# Scenario A
anchor_max_iou = np.amax(ious, axis=1)
pos_iou_anchor_label = np.where(anchor_max_iou >= pos_iou_thres)[0]
neg_iou_anchor_label = np.where(anchor_max_iou < neg_iou_thred)[0]
valid_labels[pos_iou_anchor_label] = 1
valid_labels[neg_iou_anchor_label] = 0

# Scenario B
gt_max_iou = np.amax(ious, axis=0)
gt_max_iou_anchor_label = np.where(ious == gt_max_iou)[0]
print(gt_max_iou_anchor_label)
valid_labels[gt_max_iou_anchor_label] = 1






n_sample_anchors = 256
pos_ratio = 0.5

total_n_pos = len(np.where(valid_labels == 1)[0])
n_pos_sample = n_sample_anchors*pos_ratio if total_n_pos > n_sample_anchors*pos_ratio else total_n_pos
n_neg_sample = n_sample_anchors - n_pos_sample

pos_index = np.where(valid_labels == 1)[0]
if len(pos_index) > n_sample_anchors*pos_ratio:
    disable_index = np.random.choice(pos_index, size=len(pos_index)-n_pos_sample, replace=False)
    valid_labels[disable_index] = -1

neg_index = np.where(valid_labels == 0)[0]
disable_index = np.random.choice(neg_index, size=len(neg_index) - n_neg_sample, replace=False)
valid_labels[disable_index] = -1







# Each anchor corresponds to a box
argmax_iou = np.argmax(ious, axis=1)
max_iou_box = bbox[argmax_iou].numpy()
print(max_iou_box.shape) # 8940, 4
print(valid_anchors.shape) # 8940, 4

anchor_loc_format_target = format_loc(valid_anchors, max_iou_box)
print(anchor_loc_format_target.shape) # 8940, 4







anchor_target_labels = np.empty((len(anchors),), dtype=np.int32)
anchor_target_format_locations = np.zeros((len(anchors), 4), dtype=np.float32)
anchor_target_labels.fill(-1)
anchor_target_labels[valid_index] = valid_labels
anchor_target_format_locations[valid_index] = anchor_loc_format_target
print(anchor_target_labels.shape) # 22500,
print(anchor_target_format_locations.shape) # 22500, 4







mid_channel = 512
in_channel = 512
n_anchor = 9

conv1 = nn.Conv2d(in_channel, mid_channel, 3, 1, 1)
reg_layer = nn.Conv2d(mid_channel, n_anchor*4, 1, 1, 0)
cls_layer = nn.Conv2d(mid_channel, n_anchor*2, 1, 1, 0)


#########################input###########################################
x = conv1(output_map)
anchor_pred_format_locations = reg_layer(x)
anchor_pred_scores = cls_layer(x)

print(anchor_pred_format_locations.shape) # torch.Size([1, 36, 50, 50])
print(anchor_pred_scores.shape) # torch.Size([1, 18, 50, 50])





#ground truth로 만든 anchor과 비교하기 위해, 형태를 맞춰준다.
anchor_pred_format_locations = anchor_pred_format_locations.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
anchor_pred_scores = anchor_pred_scores.permute(0, 2, 3, 1).contiguous().view(1, -1, 2)
objectness_pred_scores = anchor_pred_scores[:, :, 1]
print(anchor_target_labels.shape)
print(anchor_target_format_locations.shape)
print(anchor_pred_scores.shape)
print(anchor_pred_format_locations.shape)



gt_rpn_format_locs = torch.from_numpy(anchor_target_format_locations)
gt_rpn_scores = torch.from_numpy(anchor_target_labels)
rpn_format_locs = anchor_pred_format_locations[0]
rpn_scores = anchor_pred_scores[0]



#object or not loss
rpn_cls_loss = F.cross_entropy(rpn_scores, gt_rpn_scores.long(), ignore_index=-1)
print(rpn_cls_loss)

## location loss
mask = gt_rpn_scores > 0
mask_target_format_locs = gt_rpn_format_locs[mask]
mask_pred_format_locs = rpn_format_locs[mask]


x = torch.abs(mask_target_format_locs - mask_pred_format_locs)
rpn_loc_loss = ((x<0.5).float()*(x**2)*0.5 + (x>0.5).float()*(x-0.5)).sum()
print(rpn_loc_loss)




rpn_lambda = 10
N_reg = mask.float().sum()
rpn_loss = rpn_cls_loss + rpn_lambda / N_reg * rpn_loc_loss
print(rpn_loss)
nms_thresh = 0.7
n_train_pre_nms = 12000
n_train_post_nms = 2000
n_test_pre_nms = 6000
n_test_post_nms = 300
min_size = 16




print(anchors.shape) # 22500, 4
print(anchor_pred_format_locations.shape) # 22500, 4

rois = deformat_loc(anchors=anchors, formatted_base_anchor=anchor_pred_format_locations[0].data.numpy())
print(rois.shape) # 22500, 4
print(rois)




rois[:, 0:4:2] = np.clip(rois[:, 0:4:2], a_min=0, a_max=image_size[0])
rois[:, 1:4:2] = np.clip(rois[:, 1:4:2], a_min=0, a_max=image_size[1])
print(rois)
h = rois[:, 2] - rois[:, 0]
w = rois[:, 3] - rois[:, 1]
valid_index = np.where((h>min_size)&(w>min_size))[0]
valid_rois = rois[valid_index]
valid_scores = objectness_pred_scores[0][valid_index].data.numpy()





valid_score_order = valid_scores.ravel().argsort()[::-1]

pre_train_valid_score_order = valid_score_order[:n_train_pre_nms]
pre_train_valid_rois = valid_rois[pre_train_valid_score_order]
pre_train_valid_scores = valid_scores[pre_train_valid_score_order]

print(pre_train_valid_rois.shape) # 12000, 4
print(pre_train_valid_scores.shape) # 12000,
print(pre_train_valid_score_order.shape) # 12000,


keep_index = nms(rois=pre_train_valid_rois, scores=pre_train_valid_scores, nms_thresh=nms_thresh)
post_train_valid_rois = pre_train_valid_rois[keep_index][:n_train_post_nms] #################################################
post_train_valid_scores = pre_train_valid_scores[keep_index][:n_train_post_nms]
print(post_train_valid_rois.shape) # 2000, 4
print(post_train_valid_scores.shape) # 2000,





n_sample = 128
pos_ratio = 0.25
pos_iou_thresh = 0.5
neg_iou_thresh_hi = 0.5
neg_iou_thresh_lo = 0.0

ious = bbox_iou(post_train_valid_rois, bbox)
print(ious.shape) # 2000, 2

bbox_assignments = ious.argmax(axis=1)
roi_max_ious = ious.max(axis=1)
roi_target_labels = labels[bbox_assignments]
print(roi_target_labels.shape) # 2000






total_n_pos = len(np.where(roi_max_ious >= pos_iou_thresh)[0])
n_pos_sample = n_sample*pos_ratio if total_n_pos > n_sample*pos_ratio else total_n_pos
n_neg_sample = n_sample - n_pos_sample

print(n_pos_sample) # 10
print(n_neg_sample) # 118




pos_index = np.where(roi_max_ious >= pos_iou_thresh)[0]
pos_index = np.random.choice(pos_index, size=n_pos_sample, replace=False)

neg_index = np.where((roi_max_ious < neg_iou_thresh_hi) & (roi_max_ious > neg_iou_thresh_lo))[0]
neg_index = np.random.choice(neg_index, size=n_neg_sample, replace=False)

print(pos_index.shape) # 10
print(neg_index.shape) # 118

keep_index = np.append(pos_index, neg_index)
post_sample_target_labels = roi_target_labels[keep_index].data.numpy()
post_sample_target_labels[len(pos_index):] = 0
post_sample_rois = post_train_valid_rois[keep_index]



post_sample_bbox = bbox[bbox_assignments[keep_index]]
post_sample_format_rois = format_loc(anchors=post_sample_rois, base_anchors=post_sample_bbox.data.numpy())
print(post_sample_format_rois.shape)








rois = torch.from_numpy(post_sample_rois).float()
print(rois.shape) # 128, 4
# roi_indices = torch.zeros((len(rois),1), dtype=torch.float32)
# print(rois.shape, roi_indices.shape)

# indices_and_rois = torch.cat([roi_indices, rois], dim=1)
# print(indices_and_rois.shape)













size = (7, 7)
adaptive_max_pool = nn.AdaptiveMaxPool2d(size)

# correspond to feature map
rois.mul_(1/16.0)
rois = rois.long()




output = []
num_rois = len(rois)
for roi in rois:
    roi_feature = output_map[..., roi[0]:roi[2]+1, roi[1]:roi[3]+1]
    output.append(adaptive_max_pool(roi_feature))
output = torch.cat(output, 0)
print(output.shape) # 128, 512, 7, 7




output_ROI_pooling = output.view(output.size(0), -1)
print(output_ROI_pooling.shape) # 128, 25088



roi_head = nn.Sequential(nn.Linear(25088, 4096),
                        nn.Linear(4096, 4096))
cls_loc = nn.Linear(4096, 21*4)
cls_loc.weight.data.normal_(0, 0.01)
cls_loc.bias.data.zero_()

cls_score = nn.Linear(4096, 21)
cls_score.weight.data.normal_(0, 0.01)
cls_score.bias.data.zero_()

x = roi_head(output_ROI_pooling)
roi_cls_loc = cls_loc(x)
roi_cls_score = cls_score(x)

print(roi_cls_loc.shape, roi_cls_score.shape) # 128, 84 / 128, 21



print(roi_cls_loc.shape) # 128, 84
print(roi_cls_score.shape) # 128, 21


print(post_sample_format_rois.shape) # 128, 4
print(post_sample_target_labels.shape) # 128,

gt_roi_cls_loc = torch.from_numpy(post_sample_format_rois).float()
gt_roi_cls_label = torch.from_numpy(post_sample_target_labels).long()





roi_cls_loss = F.cross_entropy(roi_cls_score, gt_roi_cls_label)
print(roi_cls_loss)




num_roi = roi_cls_loc.size(0)
roi_cls_loc = roi_cls_loc.view(-1, 21, 4)
roi_cls_loc = roi_cls_loc[torch.arange(num_roi), gt_roi_cls_label]
print(roi_cls_loc.shape)

mask = gt_roi_cls_label>0
mask_loc_pred = roi_cls_loc[mask]
mask_loc_target = gt_roi_cls_loc[mask]

print(mask_loc_pred.shape) # 10, 4
print(mask_loc_target.shape) # 10, 4

x = torch.abs(mask_loc_pred-mask_loc_target)
roi_loc_loss = ((x<0.5).float()*x**2*0.5 + (x>0.5).float()*(x-0.5)).sum()
print(roi_loc_loss)






roi_lambda = 10
N_reg = (gt_roi_cls_label>0).float().sum()
roi_loss = roi_cls_loss + roi_lambda / N_reg * roi_loc_loss
print(roi_loss)

total_loss = rpn_loss + roi_loss