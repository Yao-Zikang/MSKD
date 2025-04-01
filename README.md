# MSKD
This repository is the official implementation of our paper '[Multi-Scale Knowledge Distillation]'.
# Guidance
Our implementation is based on [MDistiller](https://github.com/megvii-research/mdistiller). Here we introduce the guidance for reproducing the experiments reported in the paper, more detailed usage of the framework please refer to [MDistiller](https://github.com/megvii-research/mdistiller).

## Preparation
1. Download [ImageNet](https://image-net.org/) and move them to MSKD/data/imagenet.
2. Download [pre-trained teachers](https://github.com/megvii-research/mdistiller/releases/tag/checkpoints) and untar them to MSKD/download_ckpts/cifar_teachers.

## Reproduction
All reported experiments can be easily reproduced by selecting/modifying our preset configuration file.
``` python
python tools/train.py --cfg configs/cifar100/MSKD/res32x4_shuv1.yaml
```
