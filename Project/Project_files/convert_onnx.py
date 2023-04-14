import torch
from models.mobilenet_v1_quant import MobileNetv1
import os

def load_quantized(pt):
    dummy = torch.rand((1, 3, 32, 32))

    # gotta go through this bullshit
    m = MobileNetv1()
    m.eval()
    m.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
    m.fuse_model(True)
    torch.ao.quantization.prepare_qat(m.train(), inplace=True)
    m(dummy)
    m_i8 = torch.ao.quantization.convert(m.eval())
    
    m_i8.load_state_dict(torch.load(pt))
    m_i8(dummy)
    m_i8.eval()

    return m_i8

def main():
    torch.backends.quantized.engine = 'qnnpack'

    m_i8 = load_quantized(f'ckpt/mobilenetv1_quant_i8.pt')

    print(m_i8)

    inputs = torch.zeros((1, 3, 32, 32))

    with torch.no_grad():
        torch.onnx.export(m_i8,
                inputs,
                'mobilenetv1_i8.onnx',
                input_names=['input'],
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX)


if __name__ == '__main__':
    main()
