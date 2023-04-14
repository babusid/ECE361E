import torch
from models.mobilenet_v1_quant import MobileNetv1
from models.shufflenet_v2 import ShuffleNetV2
import os

def load_quantized(model_class, pt):
    dummy = torch.rand((1, 3, 32, 32))

    # gotta go through this bullshit
    m = model_class()
    m.eval()
    m.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
    m.fuse_model(True)
    torch.ao.quantization.prepare_qat(m.train(), inplace=True)
    m(dummy)
    m_i8 = torch.ao.quantization.convert(m.eval())
    
    m_i8.load_state_dict(torch.load(pt, map_location=torch.device('cpu')))
    m_i8(dummy)
    m_i8.eval()

    return m_i8

def load_normal(model_class, pt):
    m = model_class()
    m.load_state_dict(torch.load(pt, map_location=torch.device('cpu')))
    return m

def main():
    torch.backends.quantized.engine = 'qnnpack'

    # m = load_quantized(MobileNetv1, 'ckpt/mobilenetv1_quant_i8.pt')
    m = load_normal(ShuffleNetV2, 'ckpt/shufflenetv2.pt')
    out = 'shufflenetv2.onnx'

    print(m)

    inputs = torch.zeros((1, 3, 32, 32))

    with torch.no_grad():
        torch.onnx.export(m,
                inputs,
                out,
                input_names=['input'],
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX)


if __name__ == '__main__':
    main()
