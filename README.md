<div align="center">    
 
# SurfaceNet: Adversarial SVBRDF Estimation from a Single Image     
[Giuseppe Vecchio](https://github.com/giuvecchio), [Simone Palazzo](https://github.com/simopal6) and Concetto Spampinato

[![Conference](http://img.shields.io/badge/ICCV-2021-4b44ce.svg)](http://iccv2021.thecvf.com/home)
[![Conference](http://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.11298)

<!--  
Conference   
-->   
</div>
 
## Overview   
This is the repo where the official PyTorch implementation for paper __"SurfaceNet: Adversarial SVBRDF Estimation from a Single Image"__ will be released.

Our super trained monkeys 🐒 are working night and day to clean up the code, stay tuned...

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
<!--
## Instructions   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/perceivelab/surfacenet

# install requirements 
cd surfacenet
pip install -r requirements.txt
 ```   
 Next, navigate to the main and run it to strat the training.   
 ```bash
# module folder
cd surfacenet

# run training   
python main.py    
```
-->


### Citation   
```
@article{DBLP:journals/corr/abs-2107-11298,
  author    = {Giuseppe Vecchio and
               Simone Palazzo and
               Concetto Spampinato},
  title     = {SurfaceNet: Adversarial {SVBRDF} Estimation from a Single Image},
  journal   = {CoRR},
  volume    = {abs/2107.11298},
  year      = {2021},
  url       = {https://arxiv.org/abs/2107.11298},
  archivePrefix = {arXiv},
  eprint    = {2107.11298},
  timestamp = {Thu, 29 Jul 2021 16:14:15 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2107-11298.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```   

