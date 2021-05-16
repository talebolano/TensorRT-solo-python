#!/usr/bin/env python3

import cv2
import os
import numpy as np
import tensorrt as trt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
import time
import argparse
import torch
import torch.nn as nn
from utils import get_seg_single,vis_seg


def torch_dtype_from_trt(dtype):
    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)


class Preprocessimage(object):
    '''
        do pre-processing:
        1. imread
        2. bgr --> rgb
        3. hwc --> chw
        4. div 255
        5. normalize
    '''
    def __init__(self,inszie):
        self.inszie = (inszie[3],inszie[2])
		self.Normalize = Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225] ) 
		
    def process(self,image_path):
        start = time.time()
        image = cv2.imread(image_path)#[...,::-1] # bgr rgb
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        H,W,_ = image.shape

        image = cv2.resize(image,self.inszie) #10ms
        new_H,new_W,_ = image.shape

        image_raw =  cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        image = torch.form_numpy(image).float().cuda()
        image = image.permute(2,0,1) # chw
        image = self.Normalize(image/255.)
        image = image.unsqueeze(0)

        return image,image_raw


class TRT_model(nn.Module):
    '''
    genrate and inference tensorrt engine
    '''
    def __init__(self,
                input_size,
                onnx_path,
                engine_path,
                mode="fp16"):
        super(TRT_model, self).__init__()
        self._register_state_dict_hook(TRT_model._on_state_dict)
        self.TRT_LOGGER = trt.Logger()
        self.onnx_path = onnx_path
        self.engine_path = engine_path
        self.input_size = input_size
        self.mode = mode
        
        if os.path.exists(engine_path):
            print("loading engine file {} ...".format(engine_path))
            trt.init_libnvinfer_plugins(self.TRT_LOGGER,"")
            with open(engine_path,"rb") as f,\
                trt.Runtime(self.TRT_LOGGER) as runtime:
                    self.engine = runtime.deserialize_cuda_engine(f.read())
        else:
            self.engine = build_engine()

        self.context = self.engine.create_execution_context()  

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        state_dict[prefix + 'engine'] = bytearray(self.engine.serialize())

    def build_engine():
        EXPLICIT_BATCH = 1<<(int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(self.TRT_LOGGER) as builder,\
            builder.create_network(EXPLICIT_BATCH) as network,\
            trt.OnnxParser(network,self.TRT_LOGGER) as parser:

            builder.max_workspace_size =1<<20
            builder.max_batch_size = 1
            if self.mode =="fp16":
                print("build fp16 mode")
                builder.fp16_mode = True
            if not os.path.exists(self.onnx_path):
                print("onnx file {} not found".format(self.onnx_path))
                exit(0)
            print("loading onnx file {} .....".format(self.onnx_path))

            with open(self.onnx_path,'rb') as model:
                print("Begining parsing....")
                parser.parse(model.read())
            print("completed parsing")
            print("Building an engine from file {}".format(onnx_path))

            network.get_input(0).shape = self.input_size 
            engine = builder.build_cuda_engine(network)

            print("completed build engine")
            with open(self.engine_path,"wb") as f:
                f.write(engine.serialize())
            return engine

    def forward(self,inputs):
        #start = time.time()
        bindngs = [None]*(1+3)
        bindngs[0]= inputs.contiguous().data_ptr()

        outputs = [None]*3
        for i in range(1,4):
            output_shape = tuple(self.context.get_binding_shape(i))
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(i))
            device = torch_device_from_trt(self.engine.get_location(i))

            output = torch.empty(size=output_shape,dtype=dtype,device=device)
            outputs[i-1] = output
            bindngs[i] = output.data_ptr()
        
        self.context.execute_async_v2(bindngs,
                torch.cuda.current_stream().cuda_stream)

        cate_preds = outputs[1]
        kernel_preds = outputs[2]
        seg_pred = outputs[0]
        # do post-processing in pytorch
        result = get_seg_single(cate_preds,kernel_preds,seg_pred)

        return result


def main():

    args = argparse.ArgumentParser(description="trt pose predict")
    args.add_argument("--onnx_path",type=str)
    args.add_argument("--engine_path",type=str)
    args.add_argument("--mode",type=str,choices=["fp32","fp16"])
    args.add_argument("--image_path",type=str,default="demo/demo.jpg")
    args.add_argument("--h",type=int,default=800)
    args.add_argument("--w",type=int,default=1344)
    args.add_argument("--mode",type=str,default="fp16")
    args.add_argument('--score_thr', type=float, default=0.3, help='score threshold for visualization')
    args.add_argument("--save",type=str,default="result.jpg")
    args.add_argument("--show",action="store_true")
    opt = args.parse_args()

    insize = [1,3,opt.h,opt.w]
    model = TRT_model(insize,opt.onnx_path,opt.engine_path,opt.mode)
    preprocesser = Preprocessimage(insize)
    if opt.show:
        cv2.namedWindow("output",0)
    ############start inference##############
    image, image_raw = preprocesser.process(opt.image_path)
    start = time.time()
    with torch.no_grad():
        result = model(image)
    print("inference time {:.3f} ms".format((time.time() - start) * 1000))
    if opt.save or opt.show:
        result_image = vis_seg(image_raw, result, score_thresh=opt.score_thr)
        if opt.save:
            cv2.imwrite(opt.save,result_image)
        if opt.show:
            cv2.imshow("output",result_image)
            cv2.waitKey(0)


if __name__=="__main__":
    main()
