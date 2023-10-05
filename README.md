<div align="center">    
 
# SurfaceNet: Adversarial SVBRDF Estimation from a Single Image     
[Giuseppe Vecchio](https://github.com/giuvecchio), [Simone Palazzo](https://github.com/simopal6) and [Concetto Spampinato](https://github.com/cspampin)

[![Paper](http://img.shields.io/badge/paper-arxiv.2107.11298-B31B1B.svg)](https://arxiv.org/abs/2107.11298)
[![Conference](http://img.shields.io/badge/ICCV-2021-4b44ce.svg)](https://openaccess.thecvf.com/content/ICCV2021/html/Vecchio_SurfaceNet_Adversarial_SVBRDF_Estimation_From_a_Single_Image_ICCV_2021_paper.html)

<!--  
Conference   
-->   
</div>
 
## Overview   
This is the official PyTorch implementation for paper __"SurfaceNet: Adversarial SVBRDF Estimation from a Single Image"__.

<br/>

![alt text](https://github.com/perceivelab/surfacenet/blob/main/imgs/hd_sample.jpg?raw=true)

<!--![alt text](https://github.com/perceivelab/surfacenet/blob/main/imgs/figures/framework.png?raw=true)-->

## Abstract

In this paper we present **SurfaceNet**, an approach for estimating spatially-varying bidirectional reflectance distribution function (SVBRDF) material properties from a single image.
We pose the problem as an image translation task and propose a novel patch-based generative adversarial network (GAN) that is able to produce high-quality, high-resolution surface reflectance maps. The employment of the GAN paradigm has a twofold objective: 1) allowing the model to recover finer details than standard translation models; 2) reducing the domain shift between synthetic and real data distributions in an unsupervised way.

An extensive evaluation, carried out on a public benchmark of synthetic and real images under different illumination conditions, shows that **SurfaceNet** largely outperforms existing SVBRDF reconstruction methods, both quantitatively and qualitatively.
Furthermore, **SurfaceNet** exhibits a remarkable ability in generating high-quality maps from real samples without any supervision at training time. 

## Method
![alt text](https://github.com/perceivelab/surfacenet/blob/main/imgs/figures/framework.png?raw=true)

## Instructions   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/perceivelab/surfacenet

# install requirements 
cd surfacenet/src
pip install -r requirements.txt
 ```   
 Next, navigate to the main and run it to strat the training.   
 ```bash
# src folder
cd surfacenet/src

# run training   
accelerate launch train.py --tag ... --dataset ... --logdir ...
```


### Citation   
```
@inproceedings{vecchio2021surfacenet,
  title={SurfaceNet: Adversarial SVBRDF Estimation from a Single Image},
  author={Vecchio, Giuseppe and Palazzo, Simone and Spampinato, Concetto},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12840--12848},
  year={2021}
}
```   

