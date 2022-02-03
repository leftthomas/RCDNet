import scipy.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F


class RCDNet(nn.Module):
    def __init__(self, num_map, num_channel, num_block, num_stage):
        super(RCDNet, self).__init__()
        # not include the initialization process
        self.iter = num_stage - 1

        # step size
        self.etaM_S = nn.Parameter(torch.ones(self.iter, 1))
        self.etaB_S = nn.Parameter(torch.full(size=(num_stage, 1), fill_value=5.0))

        # rain kernel C [3*32*9*9]
        rain_kernel = torch.FloatTensor(io.loadmat('kernel.mat')['C9'])

        # rain kernel
        self.c0 = nn.Parameter(rain_kernel)
        # self.C (rain kernel) is inter-stage sharing
        self.c = nn.Parameter(rain_kernel)
        # filter for initializing B and Z
        self.cz = nn.Parameter(torch.full(size=(num_channel, 3, 3, 3), fill_value=1.0 / 9))
        # for sparse rain layer
        self.tau = nn.Parameter(torch.ones(1))

        # proxNet
        self.proxNet_B_0 = BNet(num_channel, num_block)
        self.proxNet_B_S = nn.Sequential(*[BNet(num_channel, num_block) for _ in range(num_stage)])
        self.proxNet_M_S = nn.Sequential(*[MNet(num_map, num_block) for _ in range(num_stage)])
        self.proxNet_B_L = BNet(num_channel, num_block)

    def forward(self, x):
        list_b, list_r = [], []
        # initialize B0 and Z0 (M0=0)
        bz_ini = self.proxNet_B_0(torch.cat((x, F.conv2d(x, self.cz, stride=1, padding=1)), dim=1))
        b0, z0 = bz_ini[:, :3, :, :], bz_ini[:, 3:, :, :]

        # 1st iteration：Updating B0-->M1
        # for sparse rain layer
        r_hat = torch.relu(x - b0 - self.tau)
        # /10 for controlling the updating speed
        m = self.proxNet_M_S[0](F.conv_transpose2d(r_hat, self.c0.div(10), stride=1, padding=4))
        # /10 for controlling the updating speed
        r = F.conv2d(m, self.c.div(10), stride=1, padding=4)

        # 1st iteration: Updating M1-->B1
        b_hat = (1 - self.etaB_S[0] / 10) * b0 + self.etaB_S[0] / 10 * (x - r)
        bz = self.proxNet_B_S[0](torch.cat((b_hat, z0), dim=1))
        b, z = bz[:, :3, :, :], bz[:, 3:, :, :]
        list_r.append(r)
        list_b.append(b)
        for i in range(self.iter):
            # M-net
            epsilon = self.etaM_S[i, :] / 10 * F.conv_transpose2d((r - (x - b)), self.c.div(10), stride=1, padding=4)
            m = self.proxNet_M_S[i + 1](m - epsilon)

            # B-net
            r = F.conv2d(m, self.c.div(10), stride=1, padding=4)
            b_hat = (1 - self.etaB_S[i + 1, :] / 10) * b + self.etaB_S[i + 1, :] / 10 * (x - r)
            bz = self.proxNet_B_S[i + 1](torch.cat((b_hat, z), dim=1))
            b, z = bz[:, :3, :, :], bz[:, 3:, :, :]
            list_r.append(r)
            list_b.append(b)
        b = self.proxNet_B_L(bz)[:, :3, :, :]
        list_b.append(b)
        return b0, list_b, list_r


def make_block(num_block, num_channel):
    layers = []
    for i in range(num_block):
        layers.append(nn.Sequential(
            nn.Conv2d(num_channel, num_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channel), nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channel)))
    return nn.Sequential(*layers)


# proxNet_M
class MNet(nn.Module):
    def __init__(self, num_map, num_block):
        super(MNet, self).__init__()
        self.channels = num_map
        self.num_block = num_block
        self.layer = make_block(self.num_block, self.channels)
        # for sparse rain map
        self.tau = nn.Parameter(torch.full(size=(1, self.channels, 1, 1), fill_value=0.5))

    def forward(self, x):
        for i in range(self.num_block):
            x = torch.relu(x + self.layer[i](x))
        x = torch.relu(x - self.tau)
        return x


# proxNet_B
class BNet(nn.Module):
    def __init__(self, num_channel, num_block):
        super(BNet, self).__init__()
        # 3 means R,G,B channels for color image
        self.channels = num_channel + 3
        self.num_block = num_block
        self.layer = make_block(self.num_block, self.channels)

    def forward(self, x):
        for i in range(self.num_block):
            x = torch.relu(x + self.layer[i](x))
        return x
