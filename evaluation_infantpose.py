import unittest
import torch
from collections import OrderedDict
from evaluate.coco_eval import run_eval
from lib.network.rtpose_vgg import get_model, use_vgg
from lib.network.openpose import OpenPose_Model, use_vgg
from torch import load

with torch.autograd.no_grad():
    weight_name = './network/weight/best_pose.pth'
    state_dict = torch.load(weight_name)
    
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[7:]
        new_state_dict[name]=v
        
    model = get_model(trunk='vgg19')

    model.load_state_dict(new_state_dict)
    model.eval()
    model.float()
    model = model.cuda()
    
    run_eval(image_dir= 'E:/Deep_learning_Final/dataset/images/infant_val', anno_file = 'E:/Deep_learning_Final/dataset/annotations/annotation_val.json', vis_dir = 'E:/Deep_learning_Final/dataset/images/vis_data', model=model, preprocess='vgg')

