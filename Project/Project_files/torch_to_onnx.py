import torch
import torch.nn as nn
# from models.mobilenetv2_19_4_stem import MobileNetv2 as MobileNetv2_19_4_stem
from models.mobilenetv2_19_2_stem import MobileNetv2 as MobileNetv2_19_2_stem

model = MobileNetv2_19_2_stem()
ckpt = torch.load(f'ckpt/mobilenetv2_19_2_stem/mobilenetv2_19_2_stem_98.pt')
model.load_state_dict(ckpt)
torch.onnx.export(model.to('cpu'),
                      torch.zeros((1, 3, 32, 32)),
                      './mobilenetv2_experiments/onnxs/mobilenetv2_19_2_stem.onnx',
                      input_names=['input'], opset_version=13)
