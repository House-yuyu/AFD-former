## AFD-former: A Hybrid Transformer With Asymmetric Flow Division for Synthesized View Quality Enhancement [[IEEE TCSVT 2023](https://ieeexplore.ieee.org/abstract/document/10036109)]
Xu Zhang, Nian Cai, Huan Zhang, Yun Zhang, Jianglei Di and Weisi Lin

Guangdong University of Technology, Sun Yat-sen University, Nanyang Technological University
***
![DHAN框图 (1)](https://user-images.githubusercontent.com/93698474/219914969-265f1ae7-37f5-4acf-815c-7a91f858e407.png)

## Training datasets
### Download [Baidu Netdisk link]
[**Patch-image**](https://pan.baidu.com/s/1VIVj5alhlNEG9Kg6cfmSvA?pwd=23nt) | [**Full-image**](https://pan.baidu.com/s/1LVM8CVcvTe0fh232eo5bKA?pwd=23nt)   
**Note**:  
* Patch-image includes training and validation datasets.
  
### Details
| Setting   | Dataset Name          | Index                          |
| :-------: | :-------------------: | :----------------------------: |
| NO. 1 | Balloons              | l1~l5:1-20                     | 
| NO. 2 | Bookarrival           | l1~l5:21-30    | 
| NO. 3 | Undodancer            | l1~l5:31-50            |   
| NO. 4 | Gtfly                 | l1~l5:51-70            |             
| NO. 5 | Outdoor               | l1~l5:71-80            |           
| NO. 6 | Poznancarpark         | l1~l5:81-100            |            
| NO. 7 | PoznanStreet          | l1~l5:101-120            |            
| NO. 8 | Shark                 | l1~l5:121-140            |            

**Note**:  
* l1~l5 represent the distortion levels from 1 to 5, ordered by severity.

## Testing datasets
### Download [Baidu Netdisk link]
|  Types          | Datasets                            | 
| :-------------: | :---------------------------------: |
| H.264           | [newspaper](https://pan.baidu.com/s/1iDTaZpWoDDxAJfkPRiWqrw?pwd=23nt) |
| H.264           | [poznanHall2](https://pan.baidu.com/s/19B4_3sz7EGm7xmBajjtmZw?pwd=23nt) | 
| H.264           | [kendo](https://pan.baidu.com/s/10Dh1bRlqqmIii_Vooo7t-g?pwd=23nt)        | 
| H.264           | [lovebird1](https://pan.baidu.com/s/1mc89oaiyaQmQpQLgkIhSuw?pwd=23nt)        | 
| H.265           | [pantomime](https://pan.baidu.com/s/1fkmXtCmU6RekD64TP7lYTw?pwd=23nt)        | 
| H.265           | [newspaper](https://pan.baidu.com/s/1NJibsjEue573fxq-SJJs8w?pwd=23nt)        | 
| H.265           | [poznanHall2](https://pan.baidu.com/s/1pHr60e2ReC9j523Hg0Bung?pwd=23nt)        | 
| H.265           | [kendo](https://pan.baidu.com/s/1dif23C0NuYug3Xaw9AdUCg?pwd=23nt)        | 
| H.265           | [lovebird1](https://pan.baidu.com/s/1TAj47LHAwPFLWQAf7Jbn2Q?pwd=23nt)        | 
| MCL-3D          | [MCL-3D](http://mcl.usc.edu/mcl-3d-database/)    | 
| IETR            | [IETR](https://vaader-data.insa-rennes.fr/data/stian/ieeetom/IETR_DIBR_Database.zip) | 

## Citation
If you find this work useful for your research, please consider citing:
```
@ARTICLE{AFD-former,
  author={Zhang, Xu and Cai, Nian and Zhang, Huan and Zhang, Yun and Di, Jianglei and Lin, Weisi},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={AFD-Former: A Hybrid Transformer With Asymmetric Flow Division for Synthesized View Quality Enhancement}, 
  year={2023},
  volume={33},
  number={8},
  pages={3786-3798},
  doi={10.1109/TCSVT.2023.3241920}}
```

## Pretrained model
[Google Drive](https://drive.google.com/drive/folders/1MY0spqtkWaPDPK0Yjb2CmM1QczI1yh-Y) | [百度网盘](https://pan.baidu.com/s/10KfHP-SN2wVlEzCrdH-I8w?pwd=23nt) 

## Acknowledgement
This project is mainly based on Restormer, NAFNet, and SRMNet

## Contact
Please contact me if there is any question (Xu Zhang: zx12220802@163.com).
