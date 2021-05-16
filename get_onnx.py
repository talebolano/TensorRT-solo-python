#!/usr/bin/env python3

import argparse
import mmcv
import torch
from mmcv.runner import load_checkpoint
import torch.nn.functional as F
from mmdet.models import build_detector
import cv2
import torch.onnx as onnx
import numpy as np
import torch.nn as nn


numclass=80

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


def fpn_forward(self, inputs):
    assert len(inputs) == len(self.in_channels)

    # build laterals
    laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
    ]
    # build top-down path
    used_backbone_levels = len(laterals)
    for i in range(used_backbone_levels - 1, 0, -1):

        sh = torch.tensor(laterals[i].shape)
        laterals[i - 1] += F.interpolate(
            laterals[i], size=(sh[2]*2,sh[3]*2), mode='nearest')

    outs = [
        self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
    ]
    # part 2: add extra levels
    if self.num_outs > len(outs):
        if not self.add_extra_convs:
            for i in range(self.num_outs - used_backbone_levels):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        # add conv layers on top of original feature maps (RetinaNet)
        else:
            if self.extra_convs_on_inputs:
                orig = inputs[self.backbone_end_level - 1]
                outs.append(self.fpn_convs[used_backbone_levels](orig))
            else:
                outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
            for i in range(used_backbone_levels + 1, self.num_outs):
                if self.relu_before_extra_convs:
                    outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                else:
                    outs.append(self.fpn_convs[i](outs[-1]))
    return tuple(outs)


def forward_single(self, x, idx, eval=False, upsampled_size=None): # bbox head
    ins_kernel_feat = x
    y_range = np.linspace(-1, 1, ins_kernel_feat.shape[-1],dtype=np.float32)#h
    x_range = np.linspace(-1, 1, ins_kernel_feat.shape[-2],dtype=np.float32)#w
    x, y = np.meshgrid(y_range, x_range)
    y = y[None][None]
    x = x[None][None]
    coord_feat =np.concatenate([x, y], 1)

    coord_feat__ = torch.tensor(coord_feat)

    seg_num_grid = self.seg_num_grids[idx]
    cate_feat = F.interpolate(ins_kernel_feat, size=seg_num_grid, mode='bilinear')
    #ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)
    kernel_feat = torch.cat([ins_kernel_feat, coord_feat__], 1)
    # kernel branch
    kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear')

    kernel_feat = kernel_feat.contiguous()
    for i, kernel_layer in enumerate(self.kernel_convs):

        kernel_feat = kernel_layer.conv(kernel_feat)

        num_group = torch.tensor(kernel_layer.gn.num_groups)
        sh = torch.tensor(kernel_feat.shape)

        kernel_feat = kernel_feat.view(1,num_group,-1)
        insta_weight = torch.ones(num_group)
        insta_bias = torch.zeros(num_group)
        kernel_feat = F.instance_norm(kernel_feat,weight=insta_weight)
        kernel_feat = kernel_feat.view(sh[0],sh[1],sh[2],sh[3])

        gn_weight = kernel_layer.gn.weight.data.view(1,-1,1,1)

        gn_bias = kernel_layer.gn.bias.data.view(1,-1,1,1)
        kernel_feat = gn_weight*kernel_feat
        kernel_feat = gn_bias+kernel_feat

        kernel_feat = F.relu(kernel_feat)
        #kernel_feat = kernel_layer(kernel_feat)
    kernel_pred = self.solo_kernel(kernel_feat)

        # cate branch
    cate_feat = cate_feat.contiguous()
    for i, cate_layer in enumerate(self.cate_convs):

        cate_feat = cate_layer.conv(cate_feat)

        num_group = torch.tensor(cate_layer.gn.num_groups)
        sh = torch.tensor(cate_feat.shape)
        cate_feat = cate_feat.view(1,num_group,-1)
        
        insta_weight = torch.ones(num_group)
        insta_bias = torch.zeros(num_group)
        cate_feat = F.instance_norm(cate_feat,weight=insta_weight)
        cate_feat = cate_feat.view(sh[0],sh[1],sh[2],sh[3])

        gn_weight = cate_layer.gn.weight.data.view(1,-1,1,1)

        gn_bias = cate_layer.gn.bias.data.view(1,-1,1,1)
        cate_feat = gn_weight*cate_feat
        cate_feat = gn_bias+cate_feat

        cate_feat = F.relu(cate_feat)
        #cate_feat = cate_layer(cate_feat)

    cate_pred = self.solo_cate(cate_feat)

    if eval:
       cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
    return cate_pred, kernel_pred


def split_feats(self, feats):
    sh1 = torch.tensor(feats[0].shape)
    sh2 = torch.tensor(feats[3].shape)

    return (F.interpolate(feats[0], size=(int(sh1[2]*0.5),int(sh1[3]*0.5)), mode='bilinear'), #从da到xiao
            feats[1],
            feats[2],
            feats[3],
            F.interpolate(feats[4], size=(sh2[2],sh2[3]), mode='bilinear'))


def reshap_gn_mask_nead(layer,inputs):

    inputs = layer.conv(inputs)

    num_group = torch.tensor(layer.gn.num_groups)
    sh = torch.tensor(inputs.shape)

    inputs = inputs.view(1,num_group,-1)
        
    insta_weight = torch.ones(num_group)
    insta_bias = torch.zeros(num_group)
    inputs = F.instance_norm(inputs,weight=insta_weight)
    inputs = inputs.view(sh[0],sh[1],sh[2],sh[3])

    gn_weight = layer.gn.weight.data.view(1,-1,1,1)

    gn_bias = layer.gn.bias.data.view(1,-1,1,1)
    inputs = gn_weight*inputs
    inputs = gn_bias+inputs

    outputs = F.relu(inputs)

    return outputs


def forward(self, inputs): #mask head
    feature_add_all_level = reshap_gn_mask_nead(self.convs_all_levels[0].conv0,inputs[0])
 
    x = reshap_gn_mask_nead(self.convs_all_levels[1].conv0,inputs[1])
    sh = torch.tensor(x.shape)
    feature_add_all_level += F.interpolate(x, size=(sh[2]*2,sh[3]*2), mode='bilinear')

    x = reshap_gn_mask_nead(self.convs_all_levels[2].conv0,inputs[2])
    sh = torch.tensor(x.shape)
    x = F.interpolate(x, size=(sh[2]*2,sh[3]*2), mode='bilinear')
    x = reshap_gn_mask_nead(self.convs_all_levels[2].conv1,x)
    sh = torch.tensor(x.shape)
    feature_add_all_level += F.interpolate(x, size=(sh[2]*2,sh[3]*2), mode='bilinear')

    y_range = np.linspace(-1, 1, inputs[3].shape[-1],dtype=np.float32)#h
    x_range = np.linspace(-1, 1, inputs[3].shape[-2],dtype=np.float32)#w
    x, y = np.meshgrid(y_range, x_range)
    y = y[None][None]
    x = x[None][None]
    coord_feat =np.concatenate([x, y], 1)
    coord_feat__ = torch.tensor(coord_feat)
    input_p = torch.cat([inputs[3], coord_feat__], 1)
    x = reshap_gn_mask_nead(self.convs_all_levels[3].conv0,input_p)
    sh = torch.tensor(x.shape)
    x = F.interpolate(x, size=(sh[2]*2,sh[3]*2), mode='bilinear')
    x = reshap_gn_mask_nead(self.convs_all_levels[3].conv1,x)
    sh = torch.tensor(x.shape)
    x = F.interpolate(x, size=(sh[2]*2,sh[3]*2), mode='bilinear')
    x = reshap_gn_mask_nead(self.convs_all_levels[3].conv2,x)
    sh = torch.tensor(x.shape)
    feature_add_all_level += F.interpolate(x, size=(sh[2]*2,sh[3]*2), mode='bilinear')

    feature_pred = reshap_gn_mask_nead(self.conv_pred[0],feature_add_all_level)

    return feature_pred


def main_forward(self,x):
    x = self.extract_feat(x)
    outs = self.bbox_head(x, eval=True)
    mask_feat_pred = self.mask_feat_head(
               x[self.mask_feat_head.
                 start_level:self.mask_feat_head.end_level + 1])
    cate_pred_list = [outs[0][i].view(-1, numclass) for i in range(5)]
    kernel_pred_list = [
    outs[1][i].squeeze(0).permute(1, 2, 0).view(-1, 256) for i in range(5)] # if use light solo may change this 256

    cate_pred_list = torch.cat(cate_pred_list, dim=0)
    kernel_pred_list = torch.cat(kernel_pred_list, dim=0)

    return (cate_pred_list,kernel_pred_list,mask_feat_pred)


def parse_args():
    parser = argparse.ArgumentParser(description='get solo onnx model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--outputname',help="output name")
    parser.add_argument('--numclass', type=int,default=80)
    parser.add_argument('--inputh', type=int,default=800)
    parser.add_argument('--inputw', type=int,default=1344)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    global numclass
    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    from types import MethodType
    model.bbox_head.forward_single = MethodType(forward_single,model.bbox_head)
    model.bbox_head.split_feats = MethodType(split_feats,model.bbox_head)
    model.mask_feat_head.forward = MethodType(forward,model.mask_feat_head)
    model.neck.forward = MethodType(fpn_forward, model.neck)

    img = torch.randn(1,3,args.inputh,args.inputw)
    model.forward = MethodType(main_forward,model)

    outs = model(img)
    print(len(outs))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    outputname = ["output1","output2","output3"]
    onnx.export(model,img,args.outputname,verbose=True,opset_version=10,input_names=["input"],output_names=outputname)


if __name__ == '__main__':
    main()
