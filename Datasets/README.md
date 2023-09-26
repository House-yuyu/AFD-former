Your testing data directory structure should look like this 

`H264` <br/>
&nbsp;`└──`hall <br/>
&emsp;&emsp;`└──`gt <br/>
&emsp;&emsp;&emsp;&emsp;`├──001.bmp` <br/>
&emsp;&emsp;&emsp;&emsp;`├──002.bmp` <br/>
&emsp;&emsp;&emsp;&emsp;`├── ...    ` <br/>
&emsp;&emsp;&emsp;&emsp;`└──0200.bmp` <br/>
&nbsp;`└──`kendo <br/>
&nbsp;`└──`lovebird <br/>
&nbsp;`└──`newspaper <br/>

`H265` <br/>
&nbsp;`└──`hall <br/>
&nbsp;`└──`kendo <br/>
&nbsp;`└──`lovebird <br/>
&nbsp;`└──`newspaper <br/>
&nbsp;`└──`pantomime <br/>
  
`IETR` <br/>

`MCL-3D` <br/>

## Train datasets
[**Patch-training**](链接：https://pan.baidu.com/s/16TNGBexDyRwWlrrr8Wn23g 
提取码：23pt) or [**Full-image-training**](链接：https://pan.baidu.com/s/1l_gii2wHIEShsQL1CeyAdQ 
提取码：23ft)    
  
### Datasets
| Setting   | Weather Types          | Datasets                           | Training Configurations  |
| :---------: | :----------------------: | :----------------------------------: | :---------------------------------------------------: |
| Setting 1 | (Rain, RainDrop, Snow) | ([Outdoor-Rain](https://github.com/liruoteng/HeavyRainRemoval), [RainDrop](https://github.com/rui1996/DeRaindrop), [Snow100K](https://sites.google.com/view/yunfuliu/desnownet)) | Uniformly sampling 9000 images pairs                |
| Setting 2 | (Rain, Haze, Snow)     | ([Rain1400](https://xueyangfu.github.io/projects/cvpr2017.html), [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0), [Snow100K](https://sites.google.com/view/yunfuliu/desnownet))       | Uniformly sampling 5000 images pairs                |
| Setting 3 | (Rain, Haze, Snow)     | (SPA+, [REVIDE](https://github.com/BookerDeWitt/REVIDE_Dataset), RealSnow)            | Uniformly sampling 160000 images patches            |
