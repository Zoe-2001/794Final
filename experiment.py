import numpy as np
from data.dataset import Dataset_CIFAR
from data.util import Util
import matplotlib.pyplot as plt
import cv2
import torch
from skimage import color

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from data.dataset import Dataset_CIFAR
from data.dataset import Dataset_ImageNet
from data.util import Util


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load trained model and set to evaluation mode
model = torch.load("./model/trainedmodel_small3.pt")
model = model.to(device)
model.eval()

# load test data
image = cv2.imread('test1.JPEG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image.transpose(1, 2, 0)
image = cv2.resize(image, (256, 256))
image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
# convert to LAB
lchannel, ab = Util.to_lab(image)

f, ax = plt.subplots(1,2)
f.set_figheight(5)
f.set_figwidth(8)

ax[0].imshow(image)
ax[0].title.set_text('original')

ax[1].imshow(image)
ax[1].title.set_text('recovered')

plt.savefig('result6.png')