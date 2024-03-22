# Copyright (c) 2024 Pavlo Melnyk and Andreas Robinson, <name>.<surname>@liu.se
# SPDX-License-Identifier: MIT

# Parts are from https://github.com/FlyingGiraffe/vnn-pc, model_equi.py
# Copyright (c) 2021 Congyue Deng, congyue@stanford.edu, MIT license

from .vnn import *
from .utils import *
from .spheres import embed, TetraTransform


class TetraSphere_cls(nn.Module):

    def __init__(self, num_spheres: int, k: int = 20, init_mode=None, pooling: str = 'mean', C_prime=3,
                 normalized_spheres=False, fix_tetrasphere: bool = False, output_channels: int = 40,
                 sphere_pooling: str = 'equi_max_norm', no_mean: bool = False):
        super().__init__()

        self.k = k

        print(f"Instantiating a model with {num_spheres=}")
        self.num_spheres = num_spheres
        self.normalized_spheres = normalized_spheres
        self.embed = embed

        if sphere_pooling == 'equi_max_norm':
            self.sphere_pooling = self.equivariant_sphere_pooling
        else:
            raise ValueError("\n\nSUCH POOLING DOES NOT EXIST")

        edims = 2
        self.steerable_layer = TetraTransform(3 + edims, self.num_spheres, 1, init_mode=init_mode,
                                                        normalized_spheres=normalized_spheres)

        if fix_tetrasphere:
            for p in self.steerable_layer.parameters():
                p.requires_grad = False

        self.conv1 = VNLinearLeakyReLU(2, 64 // 3, dim=5)
        self.conv2 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3, dim=5)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 128 // 3, dim=5)
        self.conv4 = VNLinearLeakyReLU(128 // 3 * 2, 256 // 3, dim=5)

        self.conv5 = VNLinearLeakyReLU(256 // 3 + 128 // 3 + 64 // 3 * 2, 1024 // C_prime, dim=4,
                                       share_nonlinearity=True)

        self.no_mean = no_mean
        m = 1 if no_mean else 2

        self.std_feature = VNStdFeature(1024 // C_prime * m, dim=4, normalize_frame=False, C_prime=C_prime)
        self.linear1 = nn.Linear((1024 // C_prime) * C_prime * 2 * m, 512)

        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(128 // 3)
            self.pool4 = VNMaxPool(256 // 3)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

    def equivariant_sphere_pooling(self, x, dim=1):
        # assuming x.shape = (B, K4, N)
        B, K4, N = x.shape
        x = x.transpose(-1, -2)                 # B x N x K4
        x = x.reshape(B, N, K4 // 4, 4)         # B x N x K x 4
        norms = x.norm(dim=-1)                  # B x N x K
        max_norms, ks = norms.max(dim=-1)       # B x N

        k, _ = torch.mode(ks, dim=1)            # B
        
        # advanced indexing:
        B_idcs = torch.arange(B).unsqueeze(-1).expand(B, N).unsqueeze(-1)  # B x N x 1
        N_idcs = torch.arange(N).unsqueeze(0).expand(B, N).unsqueeze(-1)   # B x N x 1

        k = k.unsqueeze(-1).expand(B, N).unsqueeze(-1)                     # B x N x 1
        x = x[B_idcs, N_idcs, k, :]                                        # B x N x 1 x 4
       
        x = x.squeeze(2)            # B x N x 4
        return x.transpose(-1, -2)  # B x 4 x N

    def forward(self, x, for_plotting=False):
        batch_size = x.size(0)  # B x 3 x N

        x, _ = self.steerable_layer(self.embed(x))  # B x K*4 x N
        _, K4, N = x.shape

        if self.num_spheres > 1:
            x = self.sphere_pooling(x, dim=1)

        x = x.unsqueeze(1)  # B x 1 x 4 x N

        x = nd_get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = self.pool1(x)

        x = nd_get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = self.pool2(x)

        x = nd_get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = self.pool3(x)

        x = nd_get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = self.pool4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        num_points = x.size(-1)
        if not self.no_mean:
            x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
            x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(batch_size, -1, num_points)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x


class TetraSphere_partseg(nn.Module):
    def __init__(self, num_spheres: int = 1, k: int = 40, init_mode=None, pooling: str = 'mean',
                 normalized_spheres=False, fix_tetrasphere: bool = False, seg_num_all: int = 50,
                 sphere_pooling: str = 'max'):
        super().__init__()
        self.seg_num_all = seg_num_all
        self.k = k

        self.num_spheres = num_spheres
        self.normalized_spheres = normalized_spheres
        self.embed = embed

        if sphere_pooling == 'equi_max_norm':
            self.sphere_pooling = self.equivariant_sphere_pooling
        else:
            raise ValueError("\n\nSUCH POOLING DOES NOT EXIST")

        edims = 2
        self.steerable_layer = TetraTransform(3 + edims, self.num_spheres, 1, init_mode=init_mode,
                                              normalized_spheres=normalized_spheres)

        if fix_tetrasphere:
            for p in self.steerable_layer.parameters():
                p.requires_grad = False

        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, 1024 // 3, dim=4, share_nonlinearity=True)
        self.std_feature = VNStdFeature(1024 // 3 * 2, dim=4, normalize_frame=False)
        self.conv8 = nn.Sequential(nn.Conv1d(2299, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)


    def equivariant_sphere_pooling(self, x, dim=1):
        # assuming x.shape = (B, K4, N)
        B, K4, N = x.shape
        x = x.transpose(-1, -2)                 # B x N x K4
        x = x.reshape(B, N, K4 // 4, 4)         # B x N x K x 4
        norms = x.norm(dim=-1)                  # B x N x K
        max_norms, ks = norms.max(dim=-1)       # B x N

        k, _ = torch.mode(ks, dim=1)            # B
        
        # advanced indexing:
        B_idcs = torch.arange(B).unsqueeze(-1).expand(B, N).unsqueeze(-1)  # B x N x 1
        N_idcs = torch.arange(N).unsqueeze(0).expand(B, N).unsqueeze(-1)   # B x N x 1

        k = k.unsqueeze(-1).expand(B, N).unsqueeze(-1)                     # B x N x 1
        x = x[B_idcs, N_idcs, k, :]                                        # B x N x 1 x 4
       
        x = x.squeeze(2)            # B x N x 4
        return x.transpose(-1, -2)  # B x 4 x N


    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x, _ = self.steerable_layer(self.embed(x))  # B x K*4 x N
        _, K4, N = x.shape

        if self.num_spheres > 1:
            x = self.sphere_pooling(x)  # B x 4 x N

        x = x.unsqueeze(1)  # B x 1 x 4 x N

        x = nd_get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.pool1(x)

        x = nd_get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = nd_get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x3 = self.pool3(x)

        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, z0 = self.std_feature(x)
        x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(batch_size, -1, num_points)
        x = x.view(batch_size, -1, num_points)
        x = x.max(dim=-1, keepdim=True)[0]

        l = l.view(batch_size, -1, 1)
        l = self.conv7(l)

        x = torch.cat((x, l), dim=1)
        x = x.repeat(1, 1, num_points)

        x = torch.cat((x, x123), dim=1)

        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        x = self.dp2(x)
        x = self.conv10(x)
        x = self.conv11(x)

        return x
