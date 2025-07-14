# CloudRuler

This repo contains information and code of the transofrmer-based cloud removal method for remote sensing images presented in the work.

Li, J., Wang, Y., Sheng, Q., Wu, Z., Wang, B., Ling, X., Du, Y., Gao, F., Camps-Vall, G., Molinier, M., 2025. A hybrid generative adversarial network for weakly-supervised cloud detection in multispectral images. Remote Sens. Environ. xx, 114913.

## Abstract
Clouds are a key factor influencing transmission of the radiance signal in optical remote sensing images. For mapping or monitoring the Earth’s surface, it is inevitable to mask or remove clouds before applying optical remote sensing images. Nowadays, deep learning (DL) based thin cloud removal methods far outperform traditional methods. Yet these DL-based methods often overlook position information or the physical cloud model in thermal bands. Moreover, most existing cloud physical models for cloud removal overlook the down-transmittance of the cloud in optical bands and do not account for the radiance of thermal bands. This work proposes a novel transformer network, CloudRuler, coupled with three rules in remote sensing domain for cloud removal. The proposed CloudRuler can distinguish the semantic meanings between similar features in different pixel positions by utilizing the Half-Spherical Coordinate System, aggregating features from local neighborhood windows with remote sensing mosaicking, and solving the parameters of the cloud physical model without limitations. Experimental results on 20 paired Landsat 8 and 9 images demonstrate that CloudRuler outperforms seven baseline methods, based on GAN, CNN, and transformer, both visually and quantitatively. Ablation experiments demonstrate that the proposed rule-based modules are highly effective in improving CloudRuler's performance for thin cloud removal. This work demonstrates that the joint use of Landsat 8 and 9 images for cloud removal is effective, producing more reliable data for downstream applications than methods that utilize only one satellite with a longer revisit period. For future research of the field, the code and dataset for reproducing the reported results are available on: https://github.com/Neooolee/CloudRuler.

## KEYWORDS: Cloud removal, transformer, cloud physical model, Landsat, deep learning.

# Code

Code snippets and demos can be found in this repository. 

# Data
The training dataset is on:
Distributions of training and testing data. Training areas are marked in white, testing areas are marked in black. The landcover background is derived from 300 m annual global land cover time series from 1992 to 2015 (Defourny et al., 2017)  
![WHUS2-CR.png]([https://i.loli.net/2020/12/23/XSh6YCA23fnMQiZ.png](https://github.com/Neooolee/CloudRuler/blob/main/Data.jpg)) 


# How to cite our work
If you find this useful, consider citing our work:
Li, J., Wang, Y., Sheng, Q., Wu, Z., Wang, B., Ling, X., Du, Y., Gao, F., Camps-Vall, G., Molinier, M., 2025. A hybrid generative adversarial network for weakly-supervised cloud detection in multispectral images. Remote Sens. Environ. xx, 114913.
