import numpy as np
import time
import random
import tensorflow as tf
import argparse
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.utils.np_utils import to_categorical
import functools
import os
import pickle

# TODO: Import your model
from models.vgg_tf import VGG
from models.mobilenet_tf import MobileNetv1

MODELS = {
    'VGG11':functools.partial(VGG, vgg_name='VGG11'),
    'VGG16':functools.partial(VGG, vgg_name='VGG16'),
    'MobileNet':MobileNetv1
}

# Argument parser
parser = argparse.ArgumentParser(
    description='ECE361E HW4 - Starter TensorFlow code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epoch to train')
# Select model to be trained
parser.add_argument('--model', type=str, default='VGG11', 
                    choices=MODELS.keys(),
                    help='Which model to use')
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
model_type = args.model

random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# Insert your model here
model = MODELS[args.model]()
DIRECTORY_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model)
full_fname = lambda f_suffix: os.path.join(DIRECTORY_NAME, args.model + f_suffix)

model.summary()

# Load the training and testing dataset
# Convert the image from uint8 with data range 0~255 to float32 with data range 0~1
# Hint: cast the array to float and then divide by 255
(trainX, trainy), (testX, testy) = cifar10.load_data()
trainX, testX = trainX / 255.0, testX / 255.0

cifar_mean = np.array((0.4914, 0.4822, 0.4465), dtype=np.float32)
cifar_std = np.array((0.2023, 0.1994, 0.2010), dtype=np.float32)
# Normalize the datasets (make mean to 0 and std to 1)
normalize = lambda x: ((x - cifar_mean) / cifar_std)
train_norm = normalize(trainX)
test_norm = normalize(testX)

# Encode the labels into one-hot format (Hint: with to_categorical)
categorical = lambda y: to_categorical(y, 10)
train_label_onehot = categorical(trainy)
test_label_onehot = categorical(testy)

# Learning rate for different models
if args.model == 'VGG11' or args.model == 'VGG16':
    lr = 1e-4
else:
    lr = 1e-3

# Configures the model for training using compile method
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model using fit method
start_time = time.time()
history = model.fit(train_norm, train_label_onehot, 
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    validation_data=(test_norm, test_label_onehot),
                    verbose=2)
print(f'INFO: Training Time: {(time.time()-start_time):.3f} seconds')
                    
test_loss, test_acc = model.evaluate(test_norm, test_label_onehot,
                      batch_size=args.batch_size,
                      verbose=2)
print(f'INFO: Test Loss: {test_loss}')
print(f'INFO: Test Accuracy: {test_acc}')

# Save the weights of the model in .ckpt format
pickle.dump(history.history, open(full_fname('_history.p'), 'wb'))
model.save(full_fname('_saved_model'))
# model.save_weights(full_fname('.ckpt'))

