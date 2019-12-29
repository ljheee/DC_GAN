from Samling import sampling_data
from NetD import D_Net
from NetG import G_Net
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import PIL.Image as pimg
import numpy as np
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim

batch_size = 100
num_epoch = 1000
random_num = 128

if __name__ == '__main__':
    if not os.path.exists("./cartoon_img"):
        os.mkdir("./cartoon_img")
    if not os.path.exists("./params"):
        os.mkdir("./params")


    def to_img(x):
        out = (x + 1) * 0.5
        out = out.clamp(0, 1)
        return out


    dataloader = DataLoader(sampling_data, batch_size, shuffle=True, num_workers=8, drop_last=True)

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")

    d_net = D_Net().cuda()
    g_net = G_Net().cuda()

    d_net.load_state_dict(torch.load("./params/d_net.pth"))
    g_net.load_state_dict(torch.load("./params/g_net.pth"))

    loss_fn = nn.BCELoss()
    d_optimizer = torch.optim.Adam(d_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(g_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # d_optimizer = torch.optim.SGD(d_net.parameters(),lr = 0.0002,momentum=0.005)
    # g_optimizer = torch.optim.SGD(g_net.parameters(),lr = 0.0002,momentum=0.005)

    for epoch in range(num_epoch):
        for i, img in enumerate(dataloader):
            real_img = img.cuda()
            real_label = torch.ones(batch_size).view(-1, 1, 1, 1).cuda()
            fake_label = torch.zeros(batch_size).view(-1, 1, 1, 1).cuda()

            real_out = d_net(real_img)
            d_loss_real = loss_fn(real_out, real_label)
            real_scores = real_out

            z = torch.randn(batch_size, random_num, 1, 1).cuda()
            fake_img = g_net(z)
            fake_out = d_net(fake_img)
            d_loss_fake = loss_fn(fake_out, fake_label)
            fake_scores = fake_out

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            z = torch.randn(batch_size, random_num, 1, 1).cuda()
            fake_image = g_net(z)

            output = d_net(fake_image)
            g_loss = loss_fn(output, real_label)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if i % 100 == 0:
                print("Epoch [{}/{}],d_loss:{:.3f},g_loss:{:.3f}"
                      "d_real:{:.3f},d_fake:{:.3f}"
                      .format(epoch, num_epoch, d_loss, g_loss, real_scores.data.mean(), fake_scores.data.mean()))
        images = to_img(fake_image.cpu().data)
        show_img = images.permute([0, 2, 3, 1])
        # show_img = torch.transpose(images,1,3)
        # plt.imshow(show_img[0])
        # plt.pause(1)

        fake_images = to_img(fake_image.cpu().data)
        save_image(fake_images, "./cartoon_img/{}-fake_imgs.png".format(epoch + 1), nrow=10, normalize=True,
                   scale_each=True)

        real_images = to_img(real_img.cpu().data)
        save_image(real_images, "./cartoon_img/{}-real_imgs.png".format(epoch + 1), nrow=10, normalize=True,
                   scale_each=True)

        torch.save(d_net.state_dict(), "./params/d_net.pth")
        torch.save(g_net.state_dict(), "./params/g_net.pth")
