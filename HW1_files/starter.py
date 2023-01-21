import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
from multiprocessing import Process
import subprocess

# Argument parser
parser = argparse.ArgumentParser(description='ECE361E HW1 - Starter code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=25, help='Number of epoch to train')
# Define the learning rate of your optimizer
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
args = parser.parse_args()

# The size of input features
input_size = 28 * 28
# The number of target classes, you have 10 digits to classify
num_classes = 10

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

# Each experiment you will do will have slightly different results due to the randomness
# of the initialization value for the weights of the model. In order to have reproducible results,
# we have fixed a random seed to a specific value such that we "control" the randomness.
random_seed = 1
torch.manual_seed(random_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='data', train=False, transform=transforms.ToTensor())

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define your model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    # Your model only contains a single linear layer
    def forward(self, x):
        out = self.linear(x)
        return out


model = LogisticRegression(input_size, num_classes)
model = model.to(device)

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Capture memory usage with process, Problem 1.3
def mcheck():
    SMI_QUERY = 'nvidia-smi --query-gpu=uuid,timestamp,utilization.gpu,memory.used --format=csv'
    log = open('starter/memcheck.txt', 'a')
    while True:
        try:
            output = subprocess.check_output(SMI_QUERY.split(), stderr=subprocess.STDOUT).decode('utf-8')
        except subprocess.SubprocessError:
            print('Error: nvidia-smi not found')
            break
        with open('starter/memcheck.txt', 'a') as f:
            f.write(output)
        time.sleep(0.1)
memcheck_process = Process(target = mcheck)
memcheck_process.start()

# Training loop
epochs = []
trainloss = []
testloss = []
trainacc = []
testacc = []
training_time = 0
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    start = time.time()
    for train_batch_idx, (images, labels) in enumerate(train_loader):
        # Here we vectorize the 28*28 images as several 784-dimensional inputs
        images = images.to(device)
        labels = labels.to(device)
        images = images.view(-1, input_size)
        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        # Print every 100 steps the following information
        if (train_batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, train_batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (train_batch_idx + 1),
                                                                             100. * train_correct / train_total))
    end = time.time()
    training_time += (end-start)

    # Testing phase
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for test_batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Here we vectorize the 28*28 images as several 784-dimensional inputs
            images = images.view(-1, input_size)
            # Perform the actual inference
            start = time.time()
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    print('Epoch: %.0f'%(epoch+1))
    print('Train accuracy: %.2f %% Train loss: %.4f' % (100. * train_correct / train_total, train_loss / (train_batch_idx+1)))
    print('Test accuracy: %.2f %% Test loss: %.4f' % (100. * test_correct / test_total, test_loss / (test_batch_idx + 1)))
    epochs.append(epoch+1)
    trainloss.append(train_loss / (train_batch_idx+1))
    testloss.append(test_loss / (test_batch_idx + 1))
    trainacc.append(100. * train_correct / train_total)
    testacc.append(100. * test_correct / test_total)

#kill memory profiling process
memcheck_process.terminate()
memcheck_process.join()

#inference time profiling, Problem 1.3 (Total inference time, Average Inference time)
inference_time = 0
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
with torch.no_grad():
    for infbatch, (images,labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Here we vectorize the 28*28 images as several 784-dimensional inputs
        images = images.view(-1, input_size)
        # Perform the actual inference
        start = time.time()
        outputs = model(images)
        inference_time += (time.time()-start)


#Print results, Problem 1.3
print('Train time: %.2f s' % (training_time))
print('Inference time: %.2f s' % (inference_time))
print('Average Inference time: %.4f ms'%((1000.0*inference_time)/test_total))
# print('Average Inference time: %.4f ms' % 1000.*(inference_time/len(test_dataset)))

with open('starter/lossacc.csv', 'w') as f:

    def array_write_named_row(fd, rowname, array):
        fd.write(rowname + ',' + ','.join(map(str, array)) + '\n')
    
    for e in [
            ('Epoch', epochs),
            ('TrainLoss', trainloss),
            ('TestLoss', testloss),
            ('TrainAcc', trainacc),
            ('TestAcc', testacc)]:
        array_write_named_row(f, e[0], e[1])

#create scatterplots, Problem 1.2
# plt.scatter(epochs,trainloss, label="Training Loss")
# plt.scatter(epochs,testloss, label = "Test Loss")
# plt.xticks(np.asarray(np.arange(1,num_epochs+1)))
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend()
# plt.savefig('starter/lossplot_1_2.png')
# plt.clf()

# plt.scatter(epochs,trainacc, label="Training Accuracy")
# plt.scatter(epochs,testacc, label="Testing Accuracy")
# plt.xticks(np.asarray(np.arange(1,num_epochs+1)))
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend()
# plt.savefig('starter/accplot_1_2.png')
# plt.clf()

print('Finished')
