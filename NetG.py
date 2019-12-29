import torch
import torch.nn as nn


class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        # 输入的是一个随机噪点，[batch,128,1,1],输出[batch,512,4,4]
        self.convtrans1 = nn.Sequential(
            nn.ConvTranspose2d(128, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.convtrans2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.convtrans3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.convtrans4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.convtrans5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        y1 = self.convtrans1(x)
        y2 = self.convtrans2(y1)
        y3 = self.convtrans3(y2)
        y4 = self.convtrans4(y3)
        out = self.convtrans5(y4)
        return out
