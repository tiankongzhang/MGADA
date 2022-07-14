# Multi-Granularity Alignment Domain Adaptation for Object Detection---FCOS


## Installation 

Check [INSTALL.md] for installation instructions. 

The implementation of our anchor-free detector is heavily based on FCOS ([\#f0a9731](https://github.com/tianzhi0549/FCOS/tree/f0a9731dac1346788cc30d5751177f2695caaa1f)).


## Dataset

refer to  EveryPixelMatters(https://github.com/chengchunhsu/EveryPixelMatters) for all details of dataset construction.


## Training

To reproduce our experimental result, we recommend training the model by following steps.

Let's take Cityscapes -> Foggy Cityscapes as an example.


**1. Pre-training with only GA module**

export PYTHONPATH=$PWD:$PYTHONPATH

- Using VGG-16 as backbone with 2 GPUs

export CUDA_VISIBLE_DEVICES=0,1

[first stage]

 ```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2434 tools/train_net_da_mgad.py --config-file ./configs/detector/cityscapes_to_foggy/VGG/S0/da_ga_cityscapes_VGG_16_FPN_4x.yaml OUTPUT_DIR /your_path/ MODEL.ISSAMPLE True
```

[second stage]

 ```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2434 tools/train_net_da_mgad.py --config-file ./configs/detector/cityscapes_to_foggy/VGG/S1/da_ga_cityscapes_VGG_16_FPN_4x.yaml OUTPUT_DIR ./ MODEL.ISSAMPLE True
```


## Evaluation

The trained model can be evaluated by the following command.

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2300 tools/test_net_mgad.py --config-file configs/detector/cityscapes_to_foggy/VGG/S1/da_ga_cityscapes_VGG_16_FPN_4x.yaml MODEL.WEIGHT ./model_rs.pth
```

**Environments**

- Hardware
  - 2 NVIDIA 3090 GPUs

- Software
  - PyTorch 1.3.1
  - Torchvision 0.4.2
  - CUDA 11.4



## Citations

Please consider citing our paper in your publications if the project helps your research.

```
@InProceedings{Zhou_2022_CVPR,
    author    = {Zhou, Wenzhang and Du, Dawei and Zhang, Libo and Luo, Tiejian and Wu, Yanjun},
    title     = {Multi-Granularity Alignment Domain Adaptation for Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {9581-9590}
}
```
