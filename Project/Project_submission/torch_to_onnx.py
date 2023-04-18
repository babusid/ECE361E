import torch
import torch.nn as nn
import os
from models.mobilenetv2_19_2_stem_5 import MobileNetv2 as MobileNetv2_19_2_stem_5

CHOSEN_EPOCH = 98

model = MobileNetv2_19_2_stem_5()
ckpt = torch.load(f'ckpt/mobilenetv2_19_2_stem_2_5_349.pt', map_location=torch.device('cpu'))
model.load_state_dict(ckpt)

os.makedirs("onnxs", exist_ok=True)
torch.onnx.export(model.to('cpu'),
                      torch.zeros((1, 3, 32, 32)),
                      './onnxs/mobilenetv2_19_2_stem.onnx',
                      input_names=['input'], opset_version=13)
