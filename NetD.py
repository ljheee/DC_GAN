import torch
import torch.nn as nn


class D_Net(nn.Module):
    def __init__(self):
        super(D_Net, self).__init__()
        # 输入 3*96*96，输出 64*32*32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 输出 128*16*16
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 输出 256*8*8
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 输出 512*4*4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 输出 512*4*4
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        out = self.conv5(y4)
        return out
