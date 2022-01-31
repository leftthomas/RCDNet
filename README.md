# RCDNet: A Model-driven Deep Neural Network for Single Image Rain Removal (CVPR 2020)

## Requirements
- [Anaconda](https://www.anaconda.com/download/)

- [PyTorch](https://pytorch.org)
```
conda install pytorch=1.10.1 torchvision cudatoolkit -c pytorch
```

- [PIQ](https://piq.readthedocs.io/en/latest/)
```
pip install piq
```

## Dataset 
[Rain100L](https://mega.nz/file/MpgnwYDS#jqyDEyL1U9srLBbEFCPnAOZb2HZTsSrwSvRGQ6m6Dzc), 
[Rain100H](https://www.dropbox.com/s/kzbzer5wem37byg/rain100H.zip?dl=0), 
[Rain1400](https://mega.nz/file/XSxSEajb#6ZwCOSeFqAnErIg6bIjs_bUFOKcs7HoZ2rwXCP8htZc) 
and [SPA-Data]() are used, download these datasets and make sure the directory 
like this:
```                           
|-- data     
    |-- rain100L
        |-- train
            |-- rain
                norain-1.png
                ...
            `-- norain
                norain-1.png
                ...
        `-- test                                                        
    |-- rain100H
        same as rain100L
    `-- rain1400
        same as rain100L                       
    |-- spa
        same as rain100L
```

## Evaluation
All PSNR and SSIM results are computed by using [Matlab code](eval) based on Y channel of YCbCr space.

## Citation
```
@InProceedings{Wang_2020_CVPR,  
author = {Wang, Hong and Xie, Qi and Zhao, Qian and Meng, Deyu},  
title = {A Model-Driven Deep Neural Network for Single Image Rain Removal},  
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},  
month = {June},  
year = {2020}  
}
```
