## Train datasets
[**Patch-training**](链接：https://pan.baidu.com/s/16TNGBexDyRwWlrrr8Wn23g 
提取码：23pt) or [**Full-image-training**](链接：https://pan.baidu.com/s/1l_gii2wHIEShsQL1CeyAdQ 
提取码：23ft)    
  
### Details
| Setting   | Dataset Name          | Index                          | Training Configurations  |
| :---------: | :----------------------: | :----------------------------------: | :---------------------------------------------------: |
| Setting 1 | Balloons              | 1-20 | (24,32)               |
| Setting 2 | Bookarrival           | 21-30    | Uniformly sampling 5000 images pairs                |
| Setting 3 | Undodancer            | 31-50            |             |
| Setting 4 | Gtfly                 | 51-70            |             
| Setting 5 | Outdoor               | 71-80            |             |
| Setting 6 | Poznancarpark         | 81-100            |             |
| Setting 7 | PoznanStreet          | 101-120            |             |
| Setting 8 | Shark                 | 121-140            |             |


## Test datasets
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
