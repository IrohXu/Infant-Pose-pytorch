# Infant-Pose-pytorch

### Introduction  
Apply OpenPose and Infant Key-point Dataset to Evaluate Infant Posture. This is a NYU course project for CSCI-GA 3033 Section: 091 Introduction to Deep Learning Systems (Spring 2021)  
In this project, we will use CMU's famous model Openpose to do some experiment based on a self-labeled infant posture dataset.


### Data preparation  

COCO dataset: [COCO](https://cocodataset.org/#home)  

We have a self-labeled infant-pose dataset. However, due to the clinical data has laws on privacy, we can not put it on github. If you want to use these data, please email me: xc2057@nyu.edu.  

### Training and Testing

Model: [BaiduYun](https://pan.baidu.com/s/1Mx7uPwKhgw8qVbxlBu1PrQ)  including pre-trained model and finetune model, code: g7pi  



The environment for training and evaluation:  
```
python=3.6
torch>=1.2
numpy=1.7
torchvision>=0.4.0
progress
matplotlib
scipy
pycocotools
yacs
```

Training the model:  
```
python train.py
```

Test the model: (before testing, you need to put the model into corresponding folder)   
```
python test.py
```


### Reference:
Cao, Z., Hidalgo, G., Simon, T., Wei, S. E., & Sheikh, Y. OpenPose: realtime multi-person 2D pose estimation using Part Affinity Fields. IEEE transactions on pattern analysis and machine intelligence. (2019) [PDF](https://arxiv.org/pdf/1812.08008.pdf)
```
@article{cao2019openpose,
  title={OpenPose: realtime multi-person 2D pose estimation using Part Affinity Fields},
  author={Cao, Zhe and Hidalgo, Gines and Simon, Tomas and Wei, Shih-En and Sheikh, Yaser},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={43},
  number={1},
  pages={172--186},
  year={2019},
  publisher={IEEE}
}
```  

Part of the code is refer to:  
https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation  
https://github.com/donnyyou/torchcv  
https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation  
