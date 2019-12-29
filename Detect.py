from NetD import D_Net
from NetG import G_Net
from torchvision.utils import save_image

import os

import torch.optim

batch_size = 100
num_epoch = 100
random_num = 128

if __name__ == '__main__':
    if not os.path.exists("./detect_img"):
        os.mkdir("./detect_img")


    def to_img(x):
        out = (x + 1) * 0.5
        out = out.clamp(0, 1)
        return out


    # d_net = D_Net().cuda()
    # g_net = G_Net().cuda()
    d_net = D_Net().cpu()
    g_net = G_Net().cpu()

    # gpu
    # d_net.load_state_dict(torch.load("./params/d_net.pth"))
    # g_net.load_state_dict(torch.load("./params/g_net.pth"))
    d_net.load_state_dict(torch.load("./params/d_net.pth", map_location='cpu'))
    g_net.load_state_dict(torch.load("./params/g_net.pth", map_location='cpu'))

    for i in range(num_epoch):
        # z = torch.randn(batch_size, random_num, 1, 1).cuda()
        z = torch.randn(batch_size, random_num, 1, 1).cpu()
        fake_img = g_net(z)
        fake_out = d_net(fake_img)
        fake_scores = fake_out

        # z = torch.randn(batch_size, random_num, 1, 1).cuda()
        z = torch.randn(batch_size, random_num, 1, 1).cpu()
        fake_image = g_net(z)

        print(fake_scores.data.mean())

        images = to_img(fake_image.cpu().data)
        show_img = images.permute([0, 2, 3, 1])
        # show_img = torch.transpose(images,1,3)
        # plt.imshow(show_img[0])
        # plt.pause(1)

        fake_images = to_img(fake_image.cpu().data)
        # save_image(fake_images, "./detect_img/{}-fake_imgs.png".format(i + 1), nrow=10, normalize=True, scale_each=True)
        save_image(fake_images, "./detect_img/{}-fake_imgs.png".format(i + 1), nrow=10) # torchvision版本0.1.6 不是高版本
