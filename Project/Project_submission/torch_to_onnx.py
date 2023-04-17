import torch
import torch.nn as nn
import os
from models.mobilenetv2_19_2_stem import MobileNetv2 as MobileNetv2_19_2_stem

CHOSEN_EPOCH = 98

model = MobileNetv2_19_2_stem()
ckpt = torch.load(f'ckpt/mobilenetv2_19_2_stem/mobilenetv2_19_2_stem_{CHOSEN_EPOCH}.pt')
model.load_state_dict(ckpt)

os.makedirs("onnxs", exist_ok=True)
torch.onnx.export(model.to('cpu'),
                      torch.zeros((1, 3, 32, 32)),
                      './onnxs/mobilenetv2_19_2_stem.onnx',
                      input_names=['input'], opset_version=13)
