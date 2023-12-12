import cv2
import torch

class Util(object):

    def to_lab(image):
        
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        L, A, B = cv2.split(lab)

        AB = cv2.merge([A,B])
        AB_reshaped = AB.transpose(2, 0, 1) # now in channels * N * M
        L_reshaped = L.reshape((1, 256, 256))

        lchannel, ab = torch.from_numpy(L_reshaped), torch.from_numpy(AB_reshaped)

        return lchannel, ab

    def to_rgb(lchannel, ab):

        AB = ab.cpu().numpy().transpose(1,2,0)
        L = lchannel.cpu().numpy().squeeze(0)

        A, B = cv2.split(AB)
 
        merged = cv2.merge([L, A, B])
        image = cv2.cvtColor(merged, cv2.COLOR_Lab2RGB)

        return image