import time
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
from thop import profile
import os
from multiprocessing import Process
import subprocess
from torchinfo import summary
from models.vgg11_pt import VGG11
from models.vgg16_pt import VGG16

# Argument parser
parser = argparse.ArgumentParser(description='ECE361E HW3 - Starter PyTorch code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')

#which model to use
parser.add_argument('--model', type=str, default='VGG11', help='Which model to use (VGG11 or VGG16)')
args = parser.parse_args()

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size
DIRECTORY_NAME = args.model + "_dir"
DIRECTORY_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIRECTORY_NAME)


# Each experiment you will do will have slightly different results due to the randomness
# of the initialization value for the weights of the model. In order to have reproducible results,
# we have fixed a random seed to a specific value such that we "control" the randomness.
random_seed = 1
torch.manual_seed(random_seed)

# CIFAR10 Dataset (Images and Labels)
train_dataset = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]), download=True)

test_dataset = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]))

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Get chosen model
model = VGG11() if (args.model == 'VGG11') else VGG16()

# Put the model on the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters())

#metrics save file
metricsfile = os.path.join(DIRECTORY_NAME,'metrics.csv')
os.makedirs(os.path.dirname(metricsfile), exist_ok=True)
with open(metricsfile, 'w') as f:
    f.write('epoch,train_loss,train_acc,test_loss,test_acc\n')
    f.close()

# Capture memory usage with process
def mcheck():
    SMI_QUERY = 'nvidia-smi --query-gpu=uuid,timestamp,utilization.gpu,memory.used --format=csv'
    logfile = os.path.join(DIRECTORY_NAME, 'memcheck.txt')
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    with open(logfile, 'w') as f:
        f.write(SMI_QUERY)
        f.write('\n')
    while True:
        with open(logfile, 'a') as f:
            try:
                output = subprocess.check_output(SMI_QUERY.split(), stderr=subprocess.STDOUT).decode('utf-8')
            except:
                f.write('Error: nvidia-smi not found')
                break
            f.write(output)
        time.sleep(0.1)

# Training loop
train_time = 0
memcheck_process = Process(target = mcheck)
memcheck_process.start()
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    start = time.time()
    #start mem profile
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Put the images and labels on the GPU
        images = images.to(device)
        labels = labels.to(device)

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
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))
    train_time += (time.time() - start)
    

    # Testing phase
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Put the images and labels on the GPU
            images = images.to(device)
            labels = labels.to(device)

            # Perform the actual inference
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    print('Test loss: %.4f Test accuracy: %.2f %%' % (test_loss / (batch_idx + 1),100. * test_correct / test_total))

    #update metrics file
    with open(metricsfile, 'a') as f:
        f.write(    f'{epoch},'+  #epoch
                    f'{train_loss / (batch_idx + 1):.4f},'+ #train loss
                    f'{100. * train_correct / train_total:.2f},'+ #train acc
                    f'{test_loss / (batch_idx + 1):.4f},'+ #test loss
                    f'{100. * test_correct / test_total:.2f}\n' #test acc
                )
        f.close()
    torch.save(
        model.state_dict(), 
        os.path.join(DIRECTORY_NAME,f'VGG11_{epoch}.pt') 
        if (type(model) == type(VGG11())) else 
        os.path.join(DIRECTORY_NAME,f'VGG16_{epoch}.pt')
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(DIRECTORY_NAME,f'VGG11_{epoch}_optimizer.pt')
        if (type(model) == type(VGG11())) else
        os.path.join(DIRECTORY_NAME,f'VGG16_{epoch}_optimizer.pt')
    )
    print(f'Training time: {train_time:.2f} seconds')

#kill memory profiling process
memcheck_process.terminate()
memcheck_process.join()
print(f'\nTraining time: {train_time:.2f} seconds')
print(f'Training accuracy {100. * train_correct / train_total:.2f} %')
print(f'Test accuracy {100. * test_correct / test_total:.2f} %')
input = torch.randn(1, 3, 32, 32)
input = input.to(device)

start = time.time()
macs, params = profile(model, inputs=(input, ))
start = time.time() - start
print("Total FLO: \n", macs * 2)
print("Total FLOPS: \n", (macs * 2) / start)
summary(model, (1, 3, 32, 32))



# Save the PyTorch model in .pt format
torch.save(model.state_dict(), os.path.join(DIRECTORY_NAME,'VGG11.pt') if (type(model) == type(VGG11())) else os.path.join(DIRECTORY_NAME,'VGG16.pt'))