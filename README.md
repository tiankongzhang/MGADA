# Multi-Granularity Alignment Domain Adaptation for Object Detection

**[[Project Page]](https://github.com/tiankongzhang/MGAD) [[PDF]](https://arxiv.org/abs/2203.16897)**

This project hosts the code for the implementation of **[Multi-Granularity Alignment Domain Adaptation for Object Detection](https://arxiv.org/abs/2203.16897)** (CVPR 2022).

## Introduction
Domain adaptive object detection is challenging due to distinctive data distribution between source domain and target domain. In this paper, we propose a unified multi-granularity alignment based object detection framework towards domain-invariant feature learning. To this end, we encode the dependencies across different granularity perspectives including pixel-, instance-, and category-levels simultaneously to align two domains. Based on pixel-level feature maps from the backbone network, we first develop the omni-scale gated fusion module to aggregate discriminative representations of instances by scale-aware convolutions, leading to robust multi-scale object detection. Meanwhile, the multi-granularity discriminators are proposed to identify which domain different granularities of samples (\ie, pixels, instances, and categories) come from. Notably, we leverage not only the instance discriminability in different categories but also the category consistency between two domains. Extensive experiments are carried out on multiple domain adaptation scenarios, demonstrating the effectiveness of our framework over state-of-the-art algorithms on top of anchor-free FCOS and anchor-based Faster R-CNN detectors with different backbones.

![](/figs/relations.png)

## Installation 

Check FCOS and Faster-RCNN for installation instructions. 


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
