## Train datasets
[**Patch-training**](链接：https://pan.baidu.com/s/16TNGBexDyRwWlrrr8Wn23g 提取码：23pt)| [**Full-image-training**](链接：https://pan.baidu.com/s/1l_gii2wHIEShsQL1CeyAdQ 
提取码：23ft)    
  
### Details
| Setting   | Dataset Name          | Index                          |
| :-------: | :-------------------: | :----------------------------: |
| NO. 1 | Balloons              | l1~l5:1-20                     | 
| NO. 2 | Bookarrival           | l1~l5:21-30    | 
| NO. 3 | Undodancer            | l1~l5:31-50            |   
| NO. 4 | Gtfly                 | l1~l5:51-70            |             
| NO. 5 | Outdoor               | l1~l5:71-80            |           
| NO. 6 | Poznancarpark         | l1~l5:81-100            |            
| NO. 7 | PoznanStreet          | l1~l5:-120            |            
| NO. 8 | Shark                 | l1~l5:121-140            |            

Note: l1~l5 represent the distortion levels from 1 to 5, ordered by severity.

## Test datasets
### Download
|  Types          | Datasets                            | 
| :-------------: | :---------------------------------: |
| H.264           | [Outdoor-Rain](链接：https://pan.baidu.com/s/186wDsBHZ84RVGx_PNWFamQ 提取码：23nt) |
| H.264           | ([Rain1400](https://xueyangfu.github.io/projects/cvpr2017.html), [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) | 
| H.264           |         | 
| H.264           |         | 
| H.265           |         | 
| H.265           |         | 
| H.265           |         | 
| H.265           |         | 
| H.265           |         | 
| MCL-3D          |         | 
| IETR            |    [Data](https://vaader-data.insa-rennes.fr/data/stian/ieeetom/IETR_DIBR_Database.zip) | 

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
