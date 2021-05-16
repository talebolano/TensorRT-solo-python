#!/usr/bin/env python3
'''
SOLO for non-commercial purposes

Copyright (c) 2019 the authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn


seg_num_grids = [40, 36, 24, 16, 12]
self_strides = [8, 8, 16, 32, 32]
mask_thr = 0.5
update_thr = 0.05
nms_pre =500
max_per_img = 100
class_num = 1000 # ins
colors = [(np.random.random((1, 3)) * 255).tolist()[0] for i in range(class_num)]
class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus",
               "train", "truck", "boat", "traffic_light", "fire_hydrant",
               "stop_sign", "parking_meter", "bench", "bird", "cat", "dog",
               "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
               "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
               "skis", "snowboard", "sports_ball", "kite", "baseball_bat",
               "baseball_glove", "skateboard", "surfboard", "tennis_racket",
               "bottle", "wine_glass", "cup", "fork", "knife", "spoon", "bowl",
               "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
               "hot_dog", "pizza", "donut", "cake", "chair", "couch",
               "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop",
               "mouse", "remote", "keyboard", "cell_phone", "microwave",
               "oven", "toaster", "sink", "refrigerator", "book", "clock",
               "vase", "scissors", "teddy_bear", "hair_drier", "toothbrush"]


def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss' 
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay 
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


def get_seg_single(cate_preds,
                    kernel_preds,
                    seg_preds
                    ):
    inds = (cate_preds > 0.1) 
    cate_scores = cate_preds[inds]
    if len(cate_scores) == 0:
        return None

    # cate_labels & kernel_preds
    inds = inds.nonzero()
    cate_labels = inds[:, 1]
    kernel_preds = kernel_preds[inds[:, 0]]

    size_trans = cate_labels.new_tensor(seg_num_grids).pow(2).cumsum(0)
    strides = kernel_preds.new_ones(size_trans[-1])
    n_stage = len(seg_num_grids)
    strides[:size_trans[0]] *= self_strides[0]
    for ind_ in range(1, n_stage):
        strides[size_trans[ind_-1]:size_trans[ind_]] *= self_strides[ind_]
    strides = strides[inds[:, 0]] 
    
    I, N = kernel_preds.shape
    kernel_preds = kernel_preds.view(I, N, 1, 1)
    seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid() 


    seg_masks = seg_preds > mask_thr
    sum_masks = seg_masks.sum((1, 2)).float()

    keep = sum_masks > strides 
    if keep.sum() == 0:
        return None

    seg_masks = seg_masks[keep, ...] 
    seg_preds = seg_preds[keep, ...]
    sum_masks = sum_masks[keep]
    cate_scores = cate_scores[keep]
    cate_labels = cate_labels[keep]
    # mask scoring.
    seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
    cate_scores *= seg_scores

    sort_inds = torch.argsort(cate_scores, descending=True) 
    if len(sort_inds) > max_per_img: 
        sort_inds = sort_inds[:max_per_img]
    seg_masks = seg_masks[sort_inds, :, :]
    seg_preds = seg_preds[sort_inds, :, :]
    sum_masks = sum_masks[sort_inds]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]

    cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                kernel='gaussian',sigma=2.0, sum_masks=sum_masks) #

    return seg_preds, cate_labels, cate_scores    


def vis_seg(image_raw, result, score_thresh):
    '''
    draw mask in image
    '''

    img_show = image_raw # no pad
    seg_show = img_show.copy()

    ori_h,ori_w,_ = image_raw.shape

    if result!=None:
        seg_label = result[0].cpu().numpy() # seg
        output_scale = [ ori_w/seg_label.shape[2] , ori_h/seg_label.shape[1] ]
        #seg_label = seg_label.astype(np.uint8) # 变成int8
        cate_label = result[1] # cate_label
        cate_label = cate_label.cpu().numpy()
        score = result[2].cpu().numpy() # cate_scores
        
        vis_inds = score > score_thresh 
        seg_label = seg_label[vis_inds]
        num_mask = seg_label.shape[0]
        cate_label = cate_label[vis_inds]
        cate_score = score[vis_inds]

        for idx in range(num_mask):
            mask = seg_label[idx, :,:]
            # cur_mask = cv2.resize(cur_mask,(ori_w,ori_h))
            cur_mask = (mask> mask_thr).astype(np.uint8)
            if cur_mask.sum() == 0:
                continue
            mask_roi = cv2.boundingRect(cur_mask)

            draw_roi = (int(output_scale[0]*mask_roi[0]),int(output_scale[1]*mask_roi[1]),
                        int(output_scale[0]*mask_roi[2]),int(output_scale[1]*mask_roi[3]))
            
            now_mask = cv2.resize(mask[mask_roi[1]:mask_roi[1]+mask_roi[3],mask_roi[0]:mask_roi[0]+mask_roi[2]],(draw_roi[2],draw_roi[3]))
            now_mask = (now_mask> mask_thr).astype(np.uint8)
            color_mask = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))

            contours,_ = cv2.findContours(now_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

            draw_roi_mask = seg_show[ draw_roi[1]:draw_roi[1]+ draw_roi[3] , draw_roi[0]:draw_roi[0]+ draw_roi[2] ,:]
            cv2.drawContours(draw_roi_mask,contours,-1,color_mask,2)

            cur_cate = cate_label[idx]
            cur_score = cate_score[idx]

            label_text = class_names[cur_cate]

            vis_pos = (max(int(draw_roi[0]) - 10, 0), int(draw_roi[1])) #1ms
            #vis_pos = (max(int(center_x) - 10, 0), int(center_y)) #1ms
            cv2.rectangle(seg_show,(draw_roi[0],draw_roi[1]),(draw_roi[0]+ draw_roi[2],draw_roi[1]+ draw_roi[3]),(0,0,0),thickness=2)
            cv2.putText(seg_show, label_text, vis_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))  # green 0.1ms
    
    return  seg_show
