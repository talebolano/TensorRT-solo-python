# TensorRT SOLO (Python)

<a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg" alt="996.icu" /></a>


TensorRT for SOLO(use python)
## Enviroments
    TensorRT >=7.2
    Ubuntu 18.04

## A quick demo

### 1. Convert solo model form pytorch to onnx

    python3 get_onnx.py --config ${SOLO_path}/configs/solov2_r101_fpn_8gpu_3x.py --checkpoint ${SOLO_path}/work_dirs/SOLOv2_R101_3x.pth --outputname solov2_r101.onnx 

### 2.Genrate tensorRT engine and inference

    python3 inference.py --onnx_path solov2_r101.onnx --engine_path solov2_101.engine --mode fp16 --image_path ${your_picture_path} --save --show

## Inference performance(only inference time in GPU)

GPU|Model|Mode|Inference time
:--: | :--: | :--: | :--: |
V100| solov2 r101 | fp16 | 35ms
Xavier | solov2 r101 | fp16 | 150ms