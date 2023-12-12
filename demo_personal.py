import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from data.dataset import Dataset_CIFAR
from data.dataset import Dataset_ImageNet
from data.util import Util

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load trained model and set to evaluation mode
    model = torch.load("./model/trainedmodel_small3.pt")
    model = model.to(device)
    model.eval()

    # load test data
    image = cv2.imread('test7.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image.transpose(1, 2, 0)
    image = cv2.resize(image, (256, 256))
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # convert to LAB
    lchannel, ab = Util.to_lab(image)
        
    f, ax = plt.subplots(1,3)
    f.set_figheight(8)
    f.set_figwidth(15)
    f.tight_layout(pad = 1.75)

    lchannel, ab = lchannel.to(device), ab.to(device)

    # get the expected original image
    original = Util.to_rgb(lchannel, ab)

    # pass the grayscale lchannel into our trained network
    with torch.no_grad():
        pred_ab = model(lchannel.unsqueeze(0)) # output: (2,256,256)

    pred_ab = pred_ab.squeeze(0)
    # now convert with our model's prediction of the ab channel
    predicted = Util.to_rgb(lchannel, pred_ab)

    ax[0].imshow(lchannel.cpu().reshape((256,256)), cmap = 'gray')
    ax[0].title.set_text('Input: L-Channel')

    ax[1].imshow(original)
    ax[1].title.set_text('Expected')

    ax[2].imshow(predicted)
    ax[2].title.set_text('Predicted')

    plt.savefig('result8.png')
