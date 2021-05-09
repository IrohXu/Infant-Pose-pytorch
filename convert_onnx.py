import torch.onnx
from torch.autograd import Variable
import onnx
from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config
from collections import OrderedDict

model = get_model('vgg19')
state_dict = torch.load('./network/weight/best_pose.pth')

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
# model.load_state_dict(torch.load('./network/weight/best_pose.pth'))
model_path = "best_pose.onnx"
        
print(model.eval())

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)
dummy_input = Variable(torch.randn(1, 3, 224, 224)).cuda()
torch.onnx.export(model, dummy_input, model_path, verbose=False)