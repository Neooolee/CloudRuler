# CloudRuler

This repo contains information and code of the Rule-based cloud removal transformer for remote sensing images presented in the work.

Li, J., Wang, Y., Sheng, Q., Wu, Z., Wang, B., Ling, X., Liu, X., Du, Y., Gao, F., Camps-valls, G., Molinier, M., 2025. CloudRuler : Rule-based transformer for cloud removal in Landsat images. Remote Sens. Environ. 328, 114913. https://doi.org/10.1016/j.rse.2025.114913

![CloudRuler.jpg](https://github.com/Neooolee/CloudRuler/blob/main/Model.jpg).

## Abstract
Clouds are a key factor influencing transmission of the radiance signal in optical remote sensing images. For mapping or monitoring the Earth’s surface, it is inevitable to mask or remove clouds before applying optical remote sensing images. Nowadays, deep learning (DL) based thin cloud removal methods far outperform traditional methods. Yet these DL-based methods often overlook position information or the physical cloud model in thermal bands. Moreover, most existing cloud physical models for cloud removal overlook the down-transmittance of the cloud in optical bands and do not account for the radiance of thermal bands. This work proposes a novel transformer network, CloudRuler, coupled with three rules in remote sensing domain for cloud removal. The proposed CloudRuler can distinguish the semantic meanings between similar features in different pixel positions by utilizing the Half-Spherical Coordinate System, aggregating features from local neighborhood windows with remote sensing mosaicking, and solving the parameters of the cloud physical model without limitations. Experimental results on 20 paired Landsat 8 and 9 images demonstrate that CloudRuler outperforms seven baseline methods, based on GAN, CNN, and transformer, both visually and quantitatively. Ablation experiments demonstrate that the proposed rule-based modules are highly effective in improving CloudRuler's performance for thin cloud removal. This work demonstrates that the joint use of Landsat 8 and 9 images for cloud removal is effective, producing more reliable data for downstream applications than methods that utilize only one satellite with a longer revisit period. For future research of the field, the code and dataset for reproducing the reported results are available on: https://github.com/Neooolee/CloudRuler.

## KEYWORDS: Cloud removal, transformer, cloud physical model, Landsat, deep learning.

# Code

Code snippets and demos can be found in this repository. 

# Data
The NUAA-CR4L8/9 dataset is on:

Distributions of training and testing data. Training areas are marked in white, testing areas are marked in red. The land cover basemap is Annual_NLCD_LndCov_2023_CU_C1V0 product from U.S. Geological Survey (USGS). 
![NUAA-CR4L8/9.jpg](https://github.com/Neooolee/CloudRuler/blob/main/Data.jpg) 


# How to cite our work
If you find this useful, consider citing our work:

Li, J., Wang, Y., Sheng, Q., Wu, Z., Wang, B., Ling, X., Liu, X., Du, Y., Gao, F., Camps-valls, G., Molinier, M., 2025. CloudRuler : Rule-based transformer for cloud removal in Landsat images. Remote Sens. Environ. 328, 114913.

```
@article {CloudRuler: Rule-based transformer for cloud removal in Landsat images,
  author = {Jun Li, Yihui Wang, Qinghong Sheng, Zhaocong Wu, Bo Wang, Xiao Ling, Xiang Liu, Yang Du, Fan Gao, Gustau Camps-Valls, Matthieu Molinier<br>},
  title = {CloudRuler: Rule-based transformer for cloud removal in Landsat images},
  volume = {328},
  number = {},
  elocation-id = {},
  year = {2025},
  doi = {10.1016/j.rse.2025.114913},
  publisher = {Elsevier Inc.},
  URL = {},
  eprint = {},
  journal = {Remote Sensing of Environment}
}
```
