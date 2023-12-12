from data.util import Util
import torchvision.transforms as transforms
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os

import cv2

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class Dataset_ImageNet(torch.utils.data.Dataset):
    
    def __init__(self, train):

        if train: split = 'train'
        else: split = 'val'
        
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.RandomHorizontalFlip()])
        self.dataset = torchvision.datasets.ImageNet(root='./data', split = split, transform = self.transforms)
    
    def __getwithoriginal__(self, index):

        image, _ = self.dataset[index]
        
        image = image.numpy().transpose(1,2,0)

        # resize image to 256, 256
        if image.shape[0] != 256 or image.shape[1] != 256:
            image = cv2.resize(image, (256, 256))

        # change to CIE Lab Space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        return image, lab

    def __len__(self):

        return len(self.dataset)
    
    def __getitem__(self, index):

        image, _ = self.dataset[index]
        transposed = image.numpy().transpose(1, 2, 0)
        image = cv2.resize(transposed, (256, 256))

        # convert to LAB
        lchannel, ab = Util.to_lab(image)

        return lchannel, ab

def test_lchannel_extraction():

    cifar_data = Dataset_CIFAR(train = True)
    random_inds = np.random.choice(len(cifar_data), 3)

    for i, image_idx in enumerate(random_inds):

        img, lab = cifar_data.__getwithoriginal__(image_idx)

        f, ax = plt.subplots(1,2)
        
        ax[0].imshow(img)

        L,A,B = cv2.split(lab)

        ax[1].imshow(L, cmap='gray')

        plt.show()

def test_getitem():

    cifar_data = Dataset_CIFAR(train = True)
    random_inds = np.random.choice(len(cifar_data), 3)

    for i, image_idx in enumerate(random_inds):

        lchannel, ab = cifar_data[image_idx]

        f, ax = plt.subplots(1)
        ax.imshow(lchannel.reshape((256,256)), cmap='gray')

        plt.show()


class Dataset_CIFAR(torch.utils.data.Dataset):
    
    def __init__(self, train):

        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.RandomHorizontalFlip()])
        self.dataset = torchvision.datasets.CIFAR100(root='./imgdata', train = train, download = True, transform = self.transforms)
    
    def __getwithoriginal__(self, index):

        image, _ = self.dataset[index]

        image = image.numpy().transpose(1,2,0)

        # resize image to 256, 256
        if image.shape[0] != 256 or image.shape[1] != 256:
            image = cv2.resize(image, (256, 256))

        # change to CIE Lab Space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        return image, lab

    def __len__(self):

        return len(self.dataset)
    
    def __getitem__(self, index):

        image, _ = self.dataset[index]
        transposed = image.numpy().transpose(1, 2, 0)
        image = cv2.resize(transposed, (256, 256))

        # convert to LAB
        lchannel, ab = Util.to_lab(image)

        return lchannel, ab

def test_lchannel_extraction():

    cifar_data = Dataset_CIFAR(train = True)
    random_inds = np.random.choice(len(cifar_data), 3)

    for i, image_idx in enumerate(random_inds):

        img, lab = cifar_data.__getwithoriginal__(image_idx)

        f, ax = plt.subplots(1,2)
        
        ax[0].imshow(img)

        L,A,B = cv2.split(lab)

        ax[1].imshow(L, cmap='gray')

        plt.show()

def test_getitem():

    cifar_data = Dataset_CIFAR(train = True)
    random_inds = np.random.choice(len(cifar_data), 3)

    for i, image_idx in enumerate(random_inds):

        lchannel, ab = cifar_data[image_idx]

        f, ax = plt.subplots(1)
        ax.imshow(lchannel.reshape((256,256)), cmap='gray')

        plt.show()

if __name__ == '__main__':

    # for testing purposes
    test_getitem()
