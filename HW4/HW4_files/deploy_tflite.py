from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import tflite_runtime.interpreter as tflite
import time
import argparse
import subprocess
import multiprocessing as mp
from multiprocessing import Process
import get_power_temp_mc1 as mc1
import get_power_temp_rpi as rpi

parser = argparse.ArgumentParser(description='ECE361E HW4 - TFLite Deployment')
parser.add_argument('--model', type=str, default='VGG11', help='VGG11 or VGG16 or MobileNet')
parser.add_argument('--target', type=str, default='rpi', help='rpi or mc1')
args = parser.parse_args()

CWD = os.path.dirname(os.path.abspath(__file__))

# for measurement
def mcheck():
    logfile = os.path.join(CWD, f'test_RAM_{args.model}.txt')
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    while True:
        with open(logfile, 'a') as f:
            try:
                output = subprocess.check_output('free -m'.split(), stderr=subprocess.STDOUT).decode('utf-8')
            except:
                f.write('poop fart')
                break
            f.write(output)
        time.sleep(0.1)

def start_measuring(filename, device, msg_queue):
    (mc1.Computer___Show_Me_The_Power_And_The_Temperature if device == 'mc1' else 
     rpi.What_Temperature_Do_I_Preheat_My_Oven_To)(filename, msg_queue).run()

# Modify the rest of the code to use the arguments correspondingly
# Path to your tflite model
tflite_model_name = os.path.join(CWD, args.model, f'{args.model}_saved_model')
# Path to test dataset
test_file_dir = '/home/student/HW4_files/test_deployment'  

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

correct = 0
total = 0
test_time = 0

stop_measurement = mp.Queue()
filename = args.model + '_power_temperature.csv'
memcheck_process = Process(target = mcheck)
measurement_process = Process(target=start_measuring, args=(filename, args.target, stop_measurement))
memcheck_process.start()
measurement_process.start()
for filename in tqdm(os.listdir(test_file_dir)):
    with Image.open(os.path.join(test_file_dir, filename)).resize((32, 32)) as img:
        input_image = np.expand_dims(np.float32(img), axis=0)

        # Change the scale of the image from 0~255 to 0~1 and then normalize it
        norm_image = ((input_image / 255.0) - cifar_mean) / cifar_std

        # Set the input tensor as the image
        interpreter.set_tensor(input_details[0]['index'], norm_image)

        # Run the actual inference
        # Measure the inference time
        start = time.time()
        interpreter.invoke()
        test_time += time.time()-start

        # Get the output tensor
        pred_tflite = interpreter.get_tensor(output_details[0]['index'])

        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_tflite[0])

        # Get the label of the predicted class
        pred_class = label_names[top_prediction]

        # Compare the prediction and ground truth; Update the accuracy
        true_label = os.path.splitext(filename)[0].split('_')[1]
        if pred_class == true_label: 
            correct += 1 
        total += 1

stop_measurement.put('please stop running ♥‿♥')
measurement_process.join()
memcheck_process.terminate()
memcheck_process.join()

with open(f'{args.model}_testmetrics.txt', 'w') as f:
    f.write(f"Test Accuracy: {correct/total}")
    f.write(f"Test Time: {test_time} seconds")
