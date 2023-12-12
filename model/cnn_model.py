import torch.nn as nn
import torch.optim as optim

class BaseModel(nn.Module):

    def __init__(self):

        super(BaseModel, self).__init__()
		
        self.l_cent = 50
        self.l_norm = 100
        self.ab_norm = 110
        
        self.conv1 = nn.Sequential(
            # 1*256*256 -> 64*256*256
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 64*256*256 -> 64*128*128
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.conv2 = nn.Sequential(
            # 64*128*128 -> 128*128*128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # 128*128*128 -> 128*64*64
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.conv3 = nn.Sequential(
            # 128*64*64 -> 256*64*64
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # 256*64*64 -> 256*64*64
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # 256*64*64 -> 256*32*32
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        self.conv4 = nn.Sequential(
            # 256*32*32 -> 512*32*32
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # 512*32*32 -> 512*32*32
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # 512*32*32 -> 512*32*32
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )

        ## DILATED ##
        self.conv5 = nn.Sequential(
            # 512*32*32 -> 512*32*32
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(),
            # 512*32*32 -> 512*32*32
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(),
            # 512*32*32 -> 512*32*32
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )

        ## DILATED ##
        self.conv6 = nn.Sequential(
            # 512*32*32 -> 512*32*32
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(),
            # 512*32*32 -> 512*32*32
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(),
            # 512*32*32 -> 512*32*32
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )

        self.conv7 = nn.Sequential(
            # 512*32*32 -> 512*32*32
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # 512*32*32 -> 512*32*32
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # 512*32*32 -> 512*32*32
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )

        self.conv8 = nn.Sequential(
            # 512*32*32 -> 256*64*64
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            # 256*64*64 -> 256*64*64
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # 256*64*64 -> 256*64*64
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # 256*64*64 -> 313*64*64
            nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),
        )

        # to prob distribution
        self.softmax = nn.Softmax(dim = 1)
        # out channels (a,b)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        # upsample to 256*256
        self.upsample = nn.Upsample(scale_factor = 4, mode='bilinear')

    def normalize_l(self, in_l):

        return (in_l - self.l_cent) / self.l_norm
    
    def unnormalize_l(self, in_l):

        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):

        return in_ab / self.ab_norm
    
    def unnormalize_ab(self, in_ab):

        return in_ab * self.ab_norm
        
    def forward(self, x):
        
        conv1_out = self.conv1(self.normalize_l(x))
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)
        conv7_out = self.conv7(conv6_out)
        conv8_out = self.conv8(conv7_out)
        mod_out = self.model_out(self.softmax(conv8_out))
        output = self.unnormalize_ab(self.upsample(mod_out))

        return output
