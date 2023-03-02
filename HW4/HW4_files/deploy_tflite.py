from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import tflite_runtime.interpreter as tflite
import time
import argparse

# TODO: add argument parser

# TODO: add one argument loading the path to the tflite file

# TODO: add one argument loading the path to the test dataset (~/HW4_files/test_deployment)

# TODO: Modify the rest of the code to use the arguments correspondingly
tflite_model_name = ''  # Path to your tflite model
test_file_dir = ''  # Path to test dataset

# Get the interpreter for TensorFlow Lite model
interpreter = tflite.Interpreter(model_path=tflite_model_name)

# Very important: allocate tensor memory
interpreter.allocate_tensors()

# Get the position for inserting the input Tensor
input_details = interpreter.get_input_details()
# Get the position for collecting the output prediction
output_details = interpreter.get_output_details()

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

cifar_mean = np.array((0.4914, 0.4822, 0.4465), dtype=np.float32)
cifar_std = np.array((0.2023, 0.1994, 0.2010), dtype=np.float32)

for filename in tqdm(os.listdir(test_file_dir)):
    with Image.open(os.path.join(test_file_dir, filename)).resize((32, 32)) as img:
        input_image = np.expand_dims(np.float32(img), axis=0)

        # TODO: Change the scale of the image from 0~255 to 0~1 and then normalize it
        norm_image = None

        # Set the input tensor as the image
        interpreter.set_tensor(input_details[0]['index'], norm_image)

        # Run the actual inference
        # TODO: Measure the inference time
        interpreter.invoke()

        # Get the output tensor
        pred_tflite = interpreter.get_tensor(output_details[0]['index'])

        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_tflite[0])

        # Get the label of the predicted class
        pred_class = label_names[top_prediction]

        # TODO: Compare the prediction and ground truth; Update the accuracy
