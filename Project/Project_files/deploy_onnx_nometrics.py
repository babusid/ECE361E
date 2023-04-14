import numpy as np
import onnxruntime
from tqdm import tqdm
import os
from PIL import Image
import argparse
import time

#path to this script
CWD = os.path.dirname(os.path.abspath(__file__)) 

# create argument parser object
parser = argparse.ArgumentParser(description='ECE361E ckpt1 - ONNX Deployment of Pruned')

#  Add one argument for selecting VGG or MobileNet-v1 models
parser.add_argument('-m', '--model', type=str, default='mobilenetv1_i8.onnx', help='Path to ONNX file to deploy')

args = parser.parse_args()

# Modify the rest of the code to use those arguments correspondingly
onnx_path = os.path.join(CWD, args.model)

# Create Inference session using ONNX runtime
sess = onnxruntime.InferenceSession(onnx_path)

# Get the input name for the ONNX model
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)

# Get the shape of the input
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)

# Mean and standard deviation 
mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
correct = 0
total = 0
test_time = 0

for filename in tqdm(os.listdir(os.path.join(CWD, "test_deployment"))):
    # Take each image, one by one, and make inference
    with Image.open(os.path.join(CWD,"test_deployment", filename)).resize((32, 32)) as img:

        # normalize image
        input_image = (np.float32(img) / 255. - mean) / std
        
        # Add the Batch axis in the data Tensor (C, H, W)
        input_image = np.expand_dims(np.float32(input_image), axis=0)

        # change the order from (B, H, W, C) to (B, C, H, W)
        input_image = input_image.transpose([0, 3, 1, 2])

        # Run inference and get the prediction for the input image
        start = time.time()
        pred_onnx = sess.run(None, {input_name: input_image})[0]
        test_time += time.time()-start

        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_onnx[0]) % 10

        # Get the label of the predicted class
        pred_class = label_names[top_prediction]
 
        true_label = os.path.splitext(filename)[0].split('_')[1]

        if pred_class == true_label: 
            correct += 1 
        total += 1

print('Test Accuracy', correct/total)
print('Test time:', test_time)

