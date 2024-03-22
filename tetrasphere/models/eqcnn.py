# Code from https://github.com/FlyingGiraffe/vnn, vn-dgcnn/model_equi.py
# Copyright (c) 2021 Congyue Deng, congyue@stanford.edu
# SPDX-License-Identifier: MIT


from .vnn import *
from .utils import *


class EQCNN_cls(nn.Module):
    def __init__(self, k=20, pooling='mean', output_channels=40, add_x_norm=None, no_mean=False):

        super().__init__()
        self.k = k

        self.add_x_norm = add_x_norm
        assert add_x_norm in [None, 'norm', 'squared']

        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 128 // 3)
        self.conv4 = VNLinearLeakyReLU(128 // 3 * 2, 256 // 3)

        self.conv5 = VNLinearLeakyReLU(256 // 3 + 128 // 3 + 64 // 3 * 2, 1024 // 3, dim=4, share_nonlinearity=True)

        self.no_mean = no_mean
        m = 2 if not no_mean else 1

        self.std_feature = VNStdFeature(1024 // 3 * m, dim=4, normalize_frame=False)
        self.linear1 = nn.Linear((1024 // 3) * 6 * m, 512)

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

    def forward(self, x):
        batch_size = x.size(0)
        if self.add_x_norm is not None:
            x_norm = x.norm(dim=1, keepdim=True)  # B x 1 x N
            if self.add_x_norm == 'squared':
                x_norm = x_norm * x_norm
            x = torch.cat((x, x_norm), dim=1)
            # print(x.shape)

        x = x.unsqueeze(1)
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


class EQCNN_partseg(nn.Module):
    def __init__(self, k=40, pooling='mean', seg_num_all=50):

        super().__init__()
        self.seg_num_all = seg_num_all
        self.k = k

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

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.k)
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
