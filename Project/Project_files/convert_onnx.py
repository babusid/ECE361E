import torch
from models.mobilenet_v1_quant import MobileNetv1
import os

def main():
    # gotta go through this bullshit
    m = MobileNetv1()
    m.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
    m.fuse_model(True)
    m_qat = torch.ao.quantization.prepare_qat(m.train())

    ckpt = torch.load(f'ckpt/post_train_mobilenetv1_quant.pt', map_location=torch.device('cpu'))
    m_qat.load_state_dict(ckpt)

    m_i8 = torch.ao.quantization.convert(m_qat)

    # TODO: pretty sure onnx doesn't support this
    # TODO: pytorch jit for deployment??

    torch.onnx.export(m_i8,
            torch.zeros((1, 3, 32, 32)),
            'mobilenetv1_qat.onnx',
            input_names=['input'], opset_version=13)


if __name__ == '__main__':
    main()