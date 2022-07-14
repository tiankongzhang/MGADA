# Multi-Granularity Alignment Domain Adaptation for Object Detection---FCOS


## Installation 

The implementation of our anchor-based detector is heavily based on Faster-RCNN ([\#f0a9731](https://isrc.iscas.ac.cn/gitlab/research/domain-adaption)).


## Dataset

refer to  SAPNet(https://isrc.iscas.ac.cn/gitlab/research/domain-adaption) for all details of dataset construction.


## Training

To reproduce our experimental result, we recommend training the model by following steps.

Let's take Cityscapes -> Foggy Cityscapes as an example.


**1. Pre-training with only GA module**


- Using VGG-16 as backbone with 2 GPUs

export CUDA_VISIBLE_DEVICES=0,1

[first stage]

 ```
python-m torch.distributed.launch --nproc_per_node=2 --master_port=2900 dis_train.py --config-file configs/mgad/cityscape_to_foggy/VGG/s0/adv_vgg16_cityscapes_2_foggy.yaml
```

[second stage]

 ```
python-m torch.distributed.launch --nproc_per_node=2 --master_port=2900 dis_train.py --config-file configs/mgad/cityscape_to_foggy/VGG/s1/adv_vgg16_cityscapes_2_foggy.yaml --resume /your_path/
```


## Evaluation

The trained model can be evaluated by the following command.

```
python-m torch.distributed.launch --nproc_per_node=2 --master_port=2900 dis_train.py --config-file configs/mgad/cityscape_to_foggy/VGG/s1/adv_vgg16_cityscapes_2_foggy.yaml --resume /your_path/ --test-only
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
