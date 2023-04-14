import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import os
from PIL import Image
import argparse
import time

from models.mobilenet_v1_quant import MobileNetv1

#path to this script
CWD = os.path.dirname(os.path.abspath(__file__)) 

# create argument parser object
parser = argparse.ArgumentParser(description='ECE361E ckpt1 - ONNX Deployment of Pruned')

#  Add one argument for selecting VGG or MobileNet-v1 models
parser.add_argument('--model', type=str, default='ckpt/mobilenetv1_quant_i8.pt', help='Path to ONNX file to deploy')

args = parser.parse_args()

# Modify the rest of the code to use those arguments correspondingly
torch_path = os.path.join(CWD, args.model)
torch.backends.quantized.engine = 'qnnpack'

# Load in model
m = MobileNetv1()
m.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
m.fuse_model(True)
m_qat = torch.ao.quantization.prepare_qat(m.train())
m_i8 = torch.ao.quantization.convert(m_qat)

m_i8.load_state_dict(torch.load(torch_path))

# Mean and standard deviation 
mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
correct = 0
total = 0
test_time = 0

with torch.no_grad():
    for filename in tqdm(os.listdir(os.path.join(CWD, "test_deployment"))):
        # Take each image, one by one, and make inference
        with Image.open(os.path.join(CWD,"test_deployment", filename)).resize((32, 32)) as img:
            # print("Image shape:", np.float32(img).shape)

            # normalize image
            input_image = (np.float32(img) / 255. - mean) / std
            
            # Add the Batch axis in the data Tensor (C, H, W)
            input_image = np.expand_dims(np.float32(input_image), axis=0)

            # change the order from (B, H, W, C) to (B, C, H, W)
            input_image = input_image.transpose([0, 3, 1, 2])
            
            # print("Input Image shape:", input_image.shape)

            input_image = torch.from_numpy(input_image)

            # Run inference and get the prediction for the input image
            start = time.time()
            outputs = m_i8(input_image)
            test_time += time.time()-start

            # Find the prediction with the highest probability
            _, top_prediction = torch.max(outputs.data, 1)

            # Get the label of the predicted class
            pred_class = label_names[top_prediction]

            # TODO: compute test accuracy of the model 
            true_label = os.path.splitext(filename)[0].split('_')[1]

            if pred_class == true_label: 
                correct += 1 
            total += 1

print('Test Accuracy', correct/total)
print('Test time:', test_time)


