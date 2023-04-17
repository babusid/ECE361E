import torch
import torch.nn as nn
# from  models.mobilenetv2_8 import MobileNetv2 as MobileNetv2_8
# from models.mobilenetv2_6 import MobileNetv2 as MobileNetv2_6
# from models.mobilenetv2 import MobileNetv2
# from models.mobilenetv2_12 import MobileNetv2 as MobileNetv2_12
# from models.mobilenetv2_13 import MobileNetv2 as MobileNetv2_13
# from models.mobilenetv2_14 import MobileNetv2 as MobileNetv2_14
# from models.mobilenetv2_15 import MobileNetv2 as MobileNetv2_15
# from models.mobilenetv2_16 import MobileNetv2 as MobileNetv2_16
# from models.mobilenetv2_19_5 import MobileNetv2 as MobileNetv2_19_5
# from models.mobilenetv2_19_4 import MobileNetv2 as MobileNetv2_19_4
from models.mobilenetv2_19_4_stem import MobileNetv2 as MobileNetv2_19_4_stem

model = MobileNetv2_19_4_stem()
ckpt = torch.load(f'ckpt/mobilenetv2_19_4_stem/mobilenetv2_19_4_stem_92.pt')
model.load_state_dict(ckpt)
torch.onnx.export(model.to('cpu'),
                      torch.zeros((1, 3, 32, 32)),
                      './mobilenetv2_experiments/onnxs/mobilenetv2_19_4_stem.onnx',
                      input_names=['input'], opset_version=13)
