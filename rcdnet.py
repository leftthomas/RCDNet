import scipy.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F

# rain kernel C [3*32*9*9]
rain_kernel = torch.FloatTensor(io.loadmat('kernel.mat')['C9'])
# filtering on rainy image for initializing B^(0) and Z^(0)
rain_filter = torch.ones(1, 1, 3, 3).div(9)


class RCDNet(nn.Module):
    def __init__(self, num_map, num_channel, num_block, num_stage):
        super(RCDNet, self).__init__()
        self.num_stage = num_stage
        # not include the initialization process
        self.iter = self.num_stage - 1
        self.num_map = num_map
        self.num_channel = num_channel

        # step size
        self.etaM_S = nn.Parameter(torch.ones(self.iter, 1))
        self.etaB_S = nn.Parameter(torch.ones(self.num_stage, 1).mul(5))

        # rain kernel
        self.C0 = nn.Parameter(data=rain_kernel)
        # self.C (rain kernel) is inter-stage sharing
        self.C = nn.Parameter(data=rain_kernel)

        # filter for initializing B and Z
        self.C_z_const = rain_filter.expand(self.num_channel, 3, -1, -1)  # size: self.num_Z*3*3*3
        self.C_z = nn.Parameter(self.C_z_const)

        # proxNet
        self.proxNet_B_0 = BNet(num_channel, num_block)  # used in initialization process
        self.proxNet_B_S = self.make_BNet(self.num_stage, num_channel, num_block)
        self.proxNet_M_S = self.make_MNet(self.num_stage, num_map, num_block)
        # fine-tune at the last
        self.proxNet_B_last_layer = BNet(num_channel, num_block)

        # for sparse rain layer
        self.tau = nn.Parameter(torch.ones(1))

    def make_BNet(self, num_iter, num_channel, num_block):
        layers = []
        for i in range(num_iter):
            layers.append(BNet(num_channel, num_block))
        return nn.Sequential(*layers)

    def make_MNet(self, num_iter, num_map, num_block):
        layers = []
        for i in range(num_iter):
            layers.append(MNet(num_map, num_block))
        return nn.Sequential(*layers)

    def forward(self, x):
        # save mid-updating results
        list_b, list_r = [], []
        # initialize B0 and Z0 (M0=0)
        z00 = F.conv2d(x, self.C_z, stride=1, padding=1)  # dual variable z
        input_ini = torch.cat((x, z00), dim=1)
        bz_ini = self.proxNet_B_0(input_ini)
        b0 = bz_ini[:, :3, :, :]
        z0 = bz_ini[:, 3:, :, :]

        # 1st iteration：Updating B0-->M1
        r_hat = x - b0
        # for sparse rain layer
        r_hat_cut = F.relu(r_hat - self.tau)
        # /10 for controlling the updating speed
        epsilon = F.conv_transpose2d(r_hat_cut, self.C0.div(10), stride=1, padding=4)
        m1 = self.proxNet_M_S[0](epsilon)
        # /10 for controlling the updating speed
        r = F.conv2d(m1, self.C.div(10), stride=1, padding=4)

        # 1st iteration: Updating M1-->B1
        b_hat = x - r
        b_mid = (1 - self.etaB_S[0] / 10) * b0 + self.etaB_S[0] / 10 * b_hat
        input_concat = torch.cat((b_mid, z0), dim=1)
        bz = self.proxNet_B_S[0](input_concat)
        b1 = bz[:, :3, :, :]
        z1 = bz[:, 3:, :, :]
        list_b.append(b1)
        list_r.append(r)
        b = b1
        z = z1
        m = m1
        for i in range(self.iter):
            # M-net
            r_hat = x - b
            epsilon = self.etaM_S[i, :] / 10 * F.conv_transpose2d((r - r_hat), self.C.div(10), stride=1, padding=4)
            m = self.proxNet_M_S[i + 1](m - epsilon)

            # B-net
            r = F.conv2d(m, self.C.div(10), stride=1, padding=4)
            list_r.append(r)
            b_hat = x - r
            b_mid = (1 - self.etaB_S[i + 1, :] / 10) * b + self.etaB_S[i + 1, :] / 10 * b_hat
            input_concat = torch.cat((b_mid, z), dim=1)
            bz = self.proxNet_B_S[i + 1](input_concat)
            b = bz[:, :3, :, :]
            z = bz[:, 3:, :, :]
            list_b.append(b)
        bz_adjust = self.proxNet_B_last_layer(bz)
        b = bz_adjust[:, :3, :, :]
        list_b.append(b)
        return b0, list_b, list_r


def make_block(num_block, num_channel):
    layers = []
    for i in range(num_block):
        layers.append(nn.Sequential(
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(num_channel),
        ))
    return nn.Sequential(*layers)


# proxNet_M
class MNet(nn.Module):
    def __init__(self, num_map, num_block):
        super(MNet, self).__init__()
        self.channels = num_map
        self.num_block = num_block
        self.layer = make_block(self.num_block, self.channels)
        self.tau0 = torch.Tensor([0.5])
        self.tau_const = self.tau0.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).expand(-1, self.channels, -1, -1)
        # for sparse rain map
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)

    def forward(self, x):
        m = x
        for i in range(self.num_block):
            m = F.relu(m + self.layer[i](m))
        m = F.relu(m - self.tau)
        return m

# proxNet_B
class BNet(nn.Module):
    def __init__(self, num_channel, num_block):
        super(BNet, self).__init__()
        # 3 means R,G,B channels for color image
        self.channels = num_channel + 3
        self.num_block = num_block
        self.layer = make_block(self.num_block, self.channels)

    def forward(self, x):
        b = x
        for i in range(self.num_block):
            b = F.relu(b + self.layer[i](b))
        return b
