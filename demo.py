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

    num_images = 2

    # load test data
    imagenet_data = Dataset_ImageNet(train = False)
    random_inds = np.random.choice(len(imagenet_data), num_images)

    f, ax = plt.subplots(num_images,3)
    f.set_figheight(8)
    f.set_figwidth(15)
    f.tight_layout(pad = 1.75)

    for i, image_idx in enumerate(random_inds):

        lchannel, ab = imagenet_data[image_idx]
        
        lchannel, ab = lchannel.to(device), ab.to(device)
        
        # get the expected original image
        original = Util.to_rgb(lchannel, ab)

        # pass the grayscale lchannel into our trained network
        with torch.no_grad():
            pred_ab = model(lchannel.unsqueeze(0)) # output: (2,256,256)

        pred_ab = pred_ab.squeeze(0)
        # now convert with our model's prediction of the ab channel
        predicted = Util.to_rgb(lchannel, pred_ab)

        ax[i,0].imshow(lchannel.cpu().reshape((256,256)), cmap = 'gray')
        ax[i,0].title.set_text('Input: L-Channel')

        ax[i,1].imshow(original)
        ax[i,1].title.set_text('Expected')

        ax[i,2].imshow(predicted)
        ax[i,2].title.set_text('Predicted')

    plt.savefig('result13.png')
