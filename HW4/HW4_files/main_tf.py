import numpy as np
import time
import random
import tensorflow as tf
import argparse
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.utils.np_utils import to_categorical

# TODO: Import your model

# Argument parser
parser = argparse.ArgumentParser(
    description='EE379K HW4 - Starter TensorFlow code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epoch to train')
# Select model to be trained
parser.add_argument('--model_type', type=str,
                    default='VGG11', help='Model type')
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
model_type = args.model_type

random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# TODO: Insert your model here
model = None

model.summary()

# Load the training and testing datasets
(trainX, trainy), (testX, testy) = cifar10.load_data()

# TODO: Convert the image from uint8 with data range 0~255 to float32 with data range 0~1
# Hint: cast the array to float and then divide by 255

cifar_mean = np.array((0.4914, 0.4822, 0.4465), dtype=np.float32)
cifar_std = np.array((0.2023, 0.1994, 0.2010), dtype=np.float32)
# TODO: Normalize the datasets (make mean to 0 and std to 1)
train_norm = None
test_norm = None

# TODO: Encode the labels into one-hot format (Hint: with to_categorical)
train_label_onehot = None
test_label_onehot = None

# Learning rate for different models
if args.model_type == 'VGG11' or args.model_type == 'VGG16':
    lr = 1e-4
else:
    lr = 1e-3

# TODO: Configures the model for training using compile method
model.compile("some args")

# TODO: Train the model using fit method
model.fit("some args")

# TODO: Save the weights of the model in .ckpt format
