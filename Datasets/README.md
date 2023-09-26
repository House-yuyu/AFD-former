# Training datasets
### Download [Baidu Netdisk link]
[[**Patch-training**](https://pan.baidu.com/s/1VIVj5alhlNEG9Kg6cfmSvA?pwd=23nt)]|[[**Full-image-training**](https://pan.baidu.com/s/1LVM8CVcvTe0fh232eo5bKA?pwd=23nt)]    
  
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

# Testing datasets
### Download [Baidu Netdisk link]
|  Types          | Datasets                            | 
| :-------------: | :---------------------------------: |
| H.264           | [newspaper](https://pan.baidu.com/s/1iDTaZpWoDDxAJfkPRiWqrw?pwd=23nt) |
| H.264           | [poznanHall2](https://pan.baidu.com/s/19B4_3sz7EGm7xmBajjtmZw?pwd=23nt) | 
| H.264           | [kendo](https://pan.baidu.com/s/10Dh1bRlqqmIii_Vooo7t-g?pwd=23nt)        | 
| H.264           | [lovebird1](https://pan.baidu.com/s/1mc89oaiyaQmQpQLgkIhSuw?pwd=23nt)        | 
| H.265           | [pantomime]()        | 
| H.265           | [newspaper](https://pan.baidu.com/s/1NJibsjEue573fxq-SJJs8w?pwd=23nt)        | 
| H.265           | [poznanHall2]()        | 
| H.265           | [kendo](https://pan.baidu.com/s/1dif23C0NuYug3Xaw9AdUCg?pwd=23nt)        | 
| H.265           | [lovebird1](https://pan.baidu.com/s/1TAj47LHAwPFLWQAf7Jbn2Q?pwd=23nt)        | 
| MCL-3D          | [MCL-3D](http://mcl.usc.edu/mcl-3d-database/)    | 
| IETR            | [IETR](https://vaader-data.insa-rennes.fr/data/stian/ieeetom/IETR_DIBR_Database.zip) | 

Your testing data directory structure should look like this: 

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
