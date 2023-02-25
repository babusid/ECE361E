import torch
from models.vgg11_pt import VGG11
from models.vgg16_pt import VGG16
from models.mobilenet_pt import MobileNetv1
import os

CWD = os.path.dirname(os.path.abspath(__file__))

def convert_onnx(model, input_shape, output_path, opset):
    output_path = os.path.join(CWD, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(model, dummy_input, output_path, export_params=True, opset_version = opset)

def fmt(d):
    '''Remove total_params and total_ops from dict'''
    return {k: v for k, v in d.items() if 'total_params' not in k and 'total_ops' not in k}

def main():
    dict11 = torch.load(os.path.join(CWD,'VGG11/VGG11_pt.pth'))
    dict16 = torch.load(os.path.join(CWD,'VGG16/VGG16_pt.pth'))
    dictmnet = torch.load(os.path.join(CWD,'MobileNet/MobileNet_pt.pth'))
    dict11 = fmt(dict11)
    dict16 = fmt(dict16)
    dictmnet = fmt(dictmnet)

    vgg11 = VGG11()
    vgg11.load_state_dict(dict11)
    vgg16 = VGG16()
    vgg16.load_state_dict(dict16)
    mnet = MobileNetv1()
    mnet.load_state_dict(dictmnet)

    convert_onnx(vgg11, (1, 3, 32, 32), 'VGG11/VGG11_mc1.onnx', 13)
    convert_onnx(vgg16, (1, 3, 32, 32), 'VGG16/VGG16_mc1.onnx', 13)
    convert_onnx(vgg11, (1, 3, 32, 32), 'VGG11/VGG11_rpi.onnx', 17)
    convert_onnx(vgg16, (1, 3, 32, 32), 'VGG16/VGG16_rpi.onnx', 17)
    convert_onnx(mnet, (1, 3, 32, 32), 'MobileNet/MobileNet_mc1.onnx', 13)
    convert_onnx(mnet, (1, 3, 32, 32), 'MobileNet/MobileNet_rpi.onnx', 17)


if __name__ == '__main__':
    main()