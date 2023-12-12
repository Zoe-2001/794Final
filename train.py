import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import numpy as np

from model.cnn_model import BaseModel
from data.dataset import Dataset_CIFAR
from data.dataset import Dataset_ImageNet
from config import cfg
import model.loss

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('### Starting Training Pipeline ###')
print('Device:', device)

# collect hyperparameters
batch_size = cfg['batch_size']
learning_rate = cfg['learning_rate']
epochs = cfg['epochs']
dataset_size = cfg['dataset_size']

print('Epochs:', epochs)

# initialize the dataloader
cifar_train = Dataset_CIFAR(train = True)
cifar_test = Dataset_CIFAR(train = False)

# make the training dataset our desired size
#imagenet_train = Dataset_ImageNet(train = True)
indices_tr = torch.arange(dataset_size)
tr = data_utils.Subset(cifar_train, indices_tr)

te = cifar_test

trainloader = torch.utils.data.DataLoader(tr, batch_size = batch_size, shuffle = True)
testloader = torch.utils.data.DataLoader(te, batch_size = batch_size, shuffle = True)

# initialize the CNN model
model = BaseModel().to(device)

# load the previously trained model
#model = torch.load("./model/trainedmodel_small2.pt").to(device)

# define loss function and optimizer
criterion = nn.MSELoss()
cnn_optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

train_losses = []
test_losses = []

# training loop
def train(model, optimizer, epochs):

    for epoch in range(epochs):

        print(f'### Epoch {epoch+1} / {epochs} ###')

        running_loss = 0.0

        iteration = 0
        iterations = len(trainloader)
        
        for inputs, labels in trainloader:

            inputs, labels = inputs.to(device), labels.to(device)

            iteration += 1
            if iteration % 100 == 0: 
                print(f'Iteration {iteration} / {iterations}')

             # call model and get ouptuts
            prediction = model(inputs)

            # calculate the loss with function: prediction and labels as input
            mse_loss = criterion(prediction, labels)

            hsv_loss_value = model.loss.hsv_loss(prediction, labels)
            # Combine HSV loss with MSE loss
            loss = hsv_loss_value + mse_loss

            # call backward on loss. this computes gradient of loss wrt inputs
            loss.backward()
            # now we perform single optimization step, and zero out grad
            optimizer.step()
            optimizer.zero_grad()
            # add loss of current input into total running loss of epoch
            running_loss += loss.item()

        # compute total training loss for this epoch
        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss}")

        test_loss = evaluate_loss(model, testloader)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{epochs} - Test Loss: {test_loss}")

    return train_losses, test_losses

# Define a function to evaluate loss without updating gradients
def evaluate_loss(model, dataloader):

    total_loss = 0.0
    with torch.no_grad():

        for inputs, labels in dataloader:

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion.forward(outputs, labels)
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    return average_loss


train_losses, test_losses = train(model, cnn_optimizer, epochs)
print("Done!")

# save model and losses

log = open("./logs/log_small3.txt", "w")

string = 'Train Losses:' + '\n' + str(train_losses) + '\n' + \
'Test Losses:' + '\n' + str(test_losses)

log.write(string)
log.close()

torch.save(model, './model/trainedmodel_small3.pt')

if __name__ == "__main__":
    train()