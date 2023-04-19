import numpy as np
import onnxruntime
from tqdm import tqdm
import os
from PIL import Image
import argparse
import time
import multiprocessing as mp
from multiprocessing import Process, Value
import subprocess
import get_power_temp_mc1 as mc1
import get_power_temp_rpi as rpi
import numpy as np

#path to this script
CWD = os.path.dirname(os.path.abspath(__file__)) 

# create argument parser object
parser = argparse.ArgumentParser(description='ECE361E ckpt1 - ONNX Deployment of Pruned')

#  Add one argument for selecting VGG or MobileNet-v1 models
parser.add_argument('--model', type=str, help='Path to ONNX file to deploy')
parser.add_argument('--target', type=str, default='rpi', help='rpi or mc1')

args = parser.parse_args()

# Modify the rest of the code to use those arguments correspondingly
onnx_path = os.path.join(CWD, args.model)

# Create Inference session using ONNX runtime
# sess_options = onnxruntime.SessionOptions()
# sess_options.intra_op_num_threads = 3
# sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
# sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
# sess_options.enable_mem_pattern=False

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

avg_runtime_mem = Value('d')
max_runtime_mem = Value('d')
num_runtime_measurements = Value('d')

def mcheck():
    # logfile = os.path.join(CWD, f'test_RAM_{args.model}.txt')
    # os.makedirs(os.path.dirname(logfile), exist_ok=True)
    while True:
        output = subprocess.check_output('free -m'.split(), stderr=subprocess.STDOUT).decode('utf-8')
        output = output.split('\n')
        mem = output[1]
        memu = " ".join(mem.split()).split(" ")[2]
        memu = int(memu)
        avg_runtime_mem.value += memu
        num_runtime_measurements.value += 1
        if(memu > max_runtime_mem.value):
            max_runtime_mem.value = memu
        time.sleep(0.1)

def start_measuring(filename, device, msg_queue):
    (mc1.Computer___Show_Me_The_Power_And_The_Temperature if device == 'mc1' else 
     rpi.What_Temperature_Do_I_Preheat_My_Oven_To)(filename, msg_queue).run()

# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
correct = 0
total = 0
test_time = 0

# run for 10 seconds
# calculating memory baseline here
avgBaselineMem = 0
numMeasurements = 0
start = time.time()
end = time.time() + 10
while(time.time() < end):
    output = subprocess.check_output('free -m'.split(), stderr=subprocess.STDOUT).decode('utf-8')
    numMeasurements+=1
    output = output.split('\n')
    mem = output[1]
    memu = " ".join(mem.split()).split(" ")[2]
    memu = int(memu)
    avgBaselineMem += memu
avgBaselineMem /= numMeasurements

stop_measurement = mp.Queue()
filename = args.model + '_power_temperature.csv'
memcheck_process = Process(target = mcheck)
measurement_process = mp.Process(target=start_measuring, args=(filename, args.target, stop_measurement))
memcheck_process.start()
measurement_process.start()

# time.sleep(10)

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

        # Run inference and get the prediction for the input image
        start = time.time()
        pred_onnx = sess.run(None, {input_name: input_image})[0]
        test_time += time.time()-start

        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_onnx[0])

        # Get the label of the predicted class
        pred_class = label_names[top_prediction]

        # TODO: compute test accuracy of the model 
        true_label = os.path.splitext(filename)[0].split('_')[1]

        if pred_class == true_label: 
            correct += 1 
        total += 1

stop_measurement.put('please stop running ♥‿♥')

retdict = ""
# while(type(retdict) != type({})):
measurement_process.join()
retdict = stop_measurement.get()
retdict = stop_measurement.get()
measurement_process.terminate()
memcheck_process.terminate()
memcheck_process.join()

with open(f'{args.model}_testmetrics.txt', 'w') as f:
    # test metrics
    f.write(f"Test Accuracy: {correct/total} \n")
    f.write(f"Test Time: {test_time} seconds\n")
    f.write(f"Avg Latency: {(test_time/total)*1000} ms\n")
    # memory metrics
    f.write(f"Avg Baseline Memory: {avgBaselineMem} mebi\n")
    f.write(f"Avg runtime memory: {avg_runtime_mem.value/num_runtime_measurements.value} mebi\n")
    f.write(f"Max runtime memory: {max_runtime_mem.value} mebi\n")
    f.write(f"Max runtime memory (subtracting baseline): {max_runtime_mem.value - avgBaselineMem}\n")
    # power measurements
    power_measurements = retdict['power_measurements']
    time_deltas = np.diff(power_measurements[:,0])
    time_deltas = np.insert(time_deltas,0,0.0,axis=0)

    measurements = power_measurements[:,1]
    measurements = measurements - np.min(measurements)

    max_power = np.max(measurements,axis=0)
    energy = measurements*time_deltas
    netenergy = np.sum(energy,axis=0)
    f.write(f'Max Power (W): {max_power}\n')
    f.write(f'Total Energy (J): {netenergy} \n')
    f.write(f'Average Energy per image(mJ): {(netenergy/total)*1000}\n')

    #merit factor
    avg_latency_s = (test_time/total)
    avg_energy_J = (netenergy/total)
    testacc = correct/total 
    fom = testacc / (avg_latency_s*avg_energy_J) 
    
    f.write(f'FOM: {fom}')
    f.write('\n')

    


