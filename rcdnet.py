import scipy.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F

# rain kernel C initialized by the Matlab code "init_rain_kernel.m"
rain_kernel = io.loadmat('init_kernel.mat')['C9']  # 3*32*9*9
kernel = torch.FloatTensor(rain_kernel)

# filtering on rainy image for initializing B^(0) and Z^(0), refer to supplementary material(SM)
filter = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) / 9
filter = filter.unsqueeze(dim=0).unsqueeze(dim=0)


class RCDNet(nn.Module):
    def __init__(self, num_map, num_channel, num_block, num_stage):
        super(RCDNet, self).__init__()
        self.num_stage = num_stage
        # not include the initialization process
        self.iter = self.num_stage - 1
        self.num_map = num_map
        self.num_channel = num_channel

        # Stepsize
        self.etaM = torch.Tensor([1])  # initialization
        self.etaB = torch.Tensor([5])  # initialization
        self.etaM_S = self.make_eta(self.iter, self.etaM)
        self.etaB_S = self.make_eta(self.num_stage, self.etaB)

        # Rain kernel
        self.C0 = nn.Parameter(data=kernel, requires_grad=True)  # used in initialization process
        self.C = nn.Parameter(data=kernel, requires_grad=True)  # self.C (rain kernel) is inter-stage sharing

        # filter for initializing B and Z
        self.C_z_const = filter.expand(self.num_channel, 3, -1, -1)  # size: self.num_Z*3*3*3
        self.C_z = nn.Parameter(self.C_z_const, requires_grad=True)

        # proxNet
        self.proxNet_B_0 = Bnet(num_channel, num_block)  # used in initialization process
        self.proxNet_B_S = self.make_Bnet(self.num_stage, num_channel, num_block)
        self.proxNet_M_S = self.make_Mnet(self.num_stage, num_map, num_block)
        # fine-tune at the last
        self.proxNet_B_last_layer = Bnet(num_channel, num_block)

        # for sparse rain layer
        self.tau_const = torch.Tensor([1])
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)

    def make_Bnet(self, iters, num_channel, num_block):
        layers = []
        for i in range(iters):
            layers.append(Bnet(num_channel, num_block))
        return nn.Sequential(*layers)

    def make_Mnet(self, iters, num_map, num_block):
        layers = []
        for i in range(iters):
            layers.append(Mnet(num_map, num_block))
        return nn.Sequential(*layers)

    def make_eta(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1)
        eta = nn.Parameter(data=const_f, requires_grad=True)
        return eta

    def forward(self, input):
        # save mid-updating results
        ListB = []
        ListR = []

        # initialize B0 and Z0 (M0 =0)
        Z00 = F.conv2d(input, self.C_z, stride=1, padding=1)  # dual variable z
        input_ini = torch.cat((input, Z00), dim=1)
        BZ_ini = self.proxNet_B_0(input_ini)
        B0 = BZ_ini[:, :3, :, :]
        Z0 = BZ_ini[:, 3:, :, :]

        # 1st iteration：Updating B0-->M1
        R_hat = input - B0
        R_hat_cut = F.relu(R_hat - self.tau)  # for sparse rain layer
        Epsilon = F.conv_transpose2d(R_hat_cut, self.C0 / 10, stride=1,
                                     padding=4)  # /10 for controlling the updating speed
        M1 = self.proxNet_M_S[0](Epsilon)
        R = F.conv2d(M1, self.C / 10, stride=1, padding=4)  # /10 for controlling the updating speed

        # 1st iteration: Updating M1-->B1
        B_hat = input - R
        B_mid = (1 - self.etaB_S[0] / 10) * B0 + self.etaB_S[0] / 10 * B_hat
        input_concat = torch.cat((B_mid, Z0), dim=1)
        BZ = self.proxNet_B_S[0](input_concat)
        B1 = BZ[:, :3, :, :]
        Z1 = BZ[:, 3:, :, :]
        ListB.append(B1)
        ListR.append(R)
        B = B1
        Z = Z1
        M = M1
        for i in range(self.iter):
            # M-net
            R_hat = input - B
            Epsilon = self.etaM_S[i, :] / 10 * F.conv_transpose2d((R - R_hat), self.C / 10, stride=1, padding=4)
            M = self.proxNet_M_S[i + 1](M - Epsilon)

            # B-net
            R = F.conv2d(M, self.C / 10, stride=1, padding=4)
            ListR.append(R)
            B_hat = input - R
            B_mid = (1 - self.etaB_S[i + 1, :] / 10) * B + self.etaB_S[i + 1, :] / 10 * B_hat
            input_concat = torch.cat((B_mid, Z), dim=1)
            BZ = self.proxNet_B_S[i + 1](input_concat)
            B = BZ[:, :3, :, :]
            Z = BZ[:, 3:, :, :]
            ListB.append(B)
        BZ_adjust = self.proxNet_B_last_layer(BZ)
        B = BZ_adjust[:, :3, :, :]
        ListB.append(B)
        return B0, ListB, ListR


def make_resblock(num_block, num_channel):
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
class Mnet(nn.Module):
    def __init__(self, num_map, num_block):
        super(Mnet, self).__init__()
        self.channels = num_map
        self.num_block = num_block
        self.layer = make_resblock(self.num_block, self.channels)
        self.tau0 = torch.Tensor([0.5])
        self.tau_const = self.tau0.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).expand(-1, self.channels, -1, -1)
        # for sparse rain map
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)

    def forward(self, input):
        M = input
        for i in range(self.num_block):
            M = F.relu(M + self.layer[i](M))
        M = F.relu(M - self.tau)
        return M

# proxNet_B
class Bnet(nn.Module):
    def __init__(self, num_channel, num_block):
        super(Bnet, self).__init__()
        # 3 means R,G,B channels for color image
        self.channels = num_channel + 3
        self.num_block = num_block
        self.layer = make_resblock(self.num_block, self.channels)

    def forward(self, input):
        B = input
        for i in range(self.num_block):
            B = F.relu(B + self.layer[i](B))
        return B
