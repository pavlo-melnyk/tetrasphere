# Copyright (c) 2024 Pavlo Melnyk and Andreas Robinson, <name>.<surname>@liu.se
# SPDX-License-Identifier: MIT


import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def append_ones(weight) -> torch.Tensor:
    # since we learn normalized spheres, the last parameter is always 1
    # therefore, we append a constant vector of ones to the weights:
    B = weight.shape[0]

    if len(weight.shape) == 3:
        # conv1d layer
        ones_vector = torch.ones(B, 1, 1, device=weight.device)

    elif len(weight.shape) == 2:
        # linear layer
        ones_vector = torch.ones(B, 1, device=weight.device)

    else:
        raise NotImplementedError

    weight = torch.cat((weight, ones_vector), dim=1)

    return weight


def embed_spheres(centers, radii, gamma):
    # Inverse of centers_radii_gamma()
    S = torch.zeros((centers.shape[0], 5))
    S[:, -1] = gamma
    S[:, :3] = centers * gamma.unsqueeze(-1)
    S[:, -2] = - 0.5 * ((radii ** 2) - (centers.norm(dim=1) ** 2)) * gamma
    return S


def embed(x):
    if len(x.size()) == 4:
        # conv layer input
        B, M, _, N = x.size()  # batch_size x num_channels x num_points
        embed_term_1 = -torch.ones(B, M, 1, N, device=x.device)  # B x M x 1 x N
        embed_term_2 = -torch.sum(x ** 2, dim=2, keepdim=True) / 2  # along the channels D -> B x M x 1 x N

        x = torch.cat((x, embed_term_1, embed_term_2), dim=2)
        return x

    elif len(x.size()) == 3:
        # conv layer input
        B, _, N = x.size()  # batch_size x num_channels x num_points
        embed_term_1 = -torch.ones(B, 1, N, device=x.device)  # B x 1 x N
        embed_term_2 = -torch.sum(x ** 2, dim=1, keepdim=True) / 2  # along the channels D -> B x 1 x N

    elif len(x.size()) == 2:
        # linear layer input
        B, _ = x.size()  # batch_size x num_channels
        embed_term_1 = -torch.ones(B, 1, device=x.device)  # B x 1
        embed_term_2 = -torch.sum(x ** 2, dim=1, keepdim=True) / 2  # along the channels D

    else:
        raise NotImplementedError

    x = torch.cat((x, embed_term_1, embed_term_2), dim=1)
    return x


def transform_points(points, transformation):
    '''
    Applies the isomorphism transformation to embedded points.

    Args:
        points:              points embedded in R^{5}, an array of shape (num_points, 5);
        transformation:      an array of shape (5, 5).
    Returns:
        transformed points:  a tensor of the same shape as the input points.
    '''

    # reshape to (1, 5, 5) to perform matmul properly:
    T = torch.reshape(transformation, (-1, 5, 5))

    # expand dims to make a tensor of shape(num_points, 5, 1) to perform matmul properly:
    X = torch.unsqueeze(points, -1)

    # transform each point:
    transformed_points = torch.matmul(T, X)

    # reshape to the input points size -- squeeze the last dimension:
    transformed_points = torch.squeeze(transformed_points, -1)

    return transformed_points


def unembed_points(embedded_points):
    '''
    Performs a mapping that is inverse to conformal embedding.

    Args:
        embedded_points: points embedded in R^{5}, an array of shape (num_points, 5).
    Returns:
        points:          3D points, an array of shape (num_points, 3).

    '''

    # p-normalize points, i.e., divide by the last element:
    # normalized_points = embedded_points / np.expand_dims(embedded_points[:,-1], axis=-1)
    normalized_points = embedded_points / torch.unsqueeze(embedded_points[:, -1], axis=-1)

    # the first three elements are now Euclidean R^{3} coordinates:
    points = normalized_points[:, :3]

    return points


class TetraTransform(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, padding: int = 0, init_mode=None, normalized_spheres: bool = False):
        super().__init__(in_channels, out_channels, kernel_size)

        self.padding = [padding]
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.append_ones = append_ones
        self.kernel_size = kernel_size
        self.init_mode = init_mode
        self.normalized_spheres = normalized_spheres

        self.construct_filter_banks = FilterBankConstructor()

        self.init_weights(init_mode)

        # assuming the embedded input has 2 embedding dims:
        self._init_rotations = torch.eye(in_channels - 2).repeat(out_channels, 1, 1).to(
            self.weight.device)  # idenitties

        # self.reset_parameters()

    def init_weights(self, mode=None):
        # initialize the weights with some random values from a uniform distribution with k = 1 / torch.sqrt(in_channels)
        k = 1 / np.sqrt(self.in_channels)

        if self.normalized_spheres:
            weight = torch.FloatTensor(self.out_channels, self.in_channels - 1, 1).uniform_(-k, k)
        else:
            weight = torch.FloatTensor(self.out_channels, self.in_channels, 1).uniform_(-k, k)

        self.weight = torch.nn.Parameter(
            weight,
            requires_grad=True
        )

        self.bias = None

        # print('\ninitial spheres centers, radii, and gammas:')
        # print(self.centers_radii_gamma)
        # print()

    @property
    def centers_radii_gamma(self):
        S = self.weight.clone().detach()

        if self.normalized_spheres:
            gamma = torch.ones_like(S[:, -1])
            C = S[:, :-1] / gamma.unsqueeze(-1)
            r = torch.sqrt(C.norm(dim=1) ** 2 - 2 * S[:, -1] / gamma)
        else:
            gamma = S[:, -1]
            C = S[:, :-2] / gamma.unsqueeze(-1)
            r = torch.sqrt(C.norm(dim=1) ** 2 - 2 * S[:, -2] / gamma)
        return C.squeeze(-1), r.flatten(), gamma.flatten()

    def build_filter_bank(self) -> torch.Tensor:
        # using the tensor of learnable parameters (spheres), build spherical filter banks

        _init_rotations, _filter_bank = self.construct_filter_banks(self._weight, self._init_rotations)
        return _filter_bank, _init_rotations

    def forward_s0(self, x: torch.Tensor) -> torch.Tensor:
        self.device = x.device

        if self.normalized_spheres:
            self._weight = self.append_ones(self.weight)
        else:
            self._weight = self.weight

        out = torch.conv1d(x, self._weight,
                           stride=self.stride,
                           padding=self.padding,
                           bias=self.bias)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.device = x.device

        if self.normalized_spheres:
            self._weight = self.append_ones(self.weight)
        else:
            self._weight = self.weight

        _filter_bank, _init_rotations = self.build_filter_bank()
        # print(x.shape)

        assert _filter_bank.shape == (4 * self.out_channels, self.in_channels, self.kernel_size)
        assert len(x.shape) == 3 and x.shape[1] == self.in_channels

        out = torch.conv1d(x, _filter_bank,
                           stride=self.stride,
                           padding=self.padding,
                           bias=self.bias)

        self._init_rotations = _init_rotations.detach()
        # `out` has now shape `batch_size x out_channels*4 x N`
        return out, _init_rotations


class FilterBankConstructor(nn.Module):

    def __init__(self, use_prev_init_rotations=False):
        super().__init__()
        self.register_buffer('ones_vec', math.sqrt(1 / 3) * torch.ones(3))
        self.use_prev_init_rotations = use_prev_init_rotations
        # Compute the tetrahedron rotations R_{Ti}, i.e. the rotations transforming
        # (1, 1, 1) into the other three vertices

        vertices = torch.tensor([(1.0, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)])

        tetra_rotations = [torch.eye(3).unsqueeze(0)]
        tetra_rotations += [self.compute_rotation_from_two_points(vertices[0:1], v.unsqueeze(0)) for v in vertices[1:]]
        tetra_rotations = torch.stack(tetra_rotations)

        self.register_buffer('tetra_rotations', tetra_rotations.squeeze(0))

    @classmethod
    def compute_rotation_from_two_points(cls, p, q):
        '''
        A reflection method (thanks to https://math.stackexchange.com/a/2161631):
            assuming ||p|| == ||q||
            f(A, u) = A - 2 u (u^T S)/||u||^2
            S = f(I, p+q)
            R = f(S, q)

        Args:
            p, q - torch.Tensor - two nD points, necessarily with ||p|| == ||q||

        Return:
            R - DxD rotation matrix such that R p = q
        '''
        # Check for NaN and infinite values in p and q
        nan_indices_p = torch.nonzero(torch.isnan(p), as_tuple=True)
        nan_indices_q = torch.nonzero(torch.isnan(q), as_tuple=True)
        inf_indices_p = torch.nonzero(torch.isinf(p), as_tuple=True)
        inf_indices_q = torch.nonzero(torch.isinf(q), as_tuple=True)

        if nan_indices_p[0].numel() > 0 or nan_indices_q[0].numel() > 0:
            print(f"Input tensors p or q contain NaN values. "
                  f"NaN indices in p: {nan_indices_p}, "
                  f"NaN indices in q: {nan_indices_q}")

        if inf_indices_p[0].numel() > 0 or inf_indices_q[0].numel() > 0:
            print(f"Input tensors p or q contain infinite values. "
                  f"Inf indices in p: {inf_indices_p}, "
                  f"Inf indices in q: {inf_indices_q}")

        assert len(p.shape) == 2 and p.shape == q.shape
        a = torch.abs(p.norm(dim=1, keepdim=True).pow(2) - q.norm(dim=1, keepdim=True).pow(2)).max()

        # Check for NaN and infinite values in the computation of a
        if torch.isnan(a).any() or torch.isinf(a).any():
            print("The computation of 'a' results in NaN or infinite values.")

        assert a < 1e-5, 'Such rotation doesn\'t exist: ||p|| must be equal to ||q||, ' + str(a)

        B, D = p.shape

        def reflection(S, u):
            # reflection of S on hyperplane u:
            # S can be a matrix; S and u must have the same number of rows.
            assert len(S) == len(u) and S.shape[-1] == u.shape[-1]

            v = torch.matmul(u.unsqueeze(1), S)  # (Bx1xD)
            v = v.squeeze(1) / u.norm(dim=1, keepdim=True) ** 2  # (BxD) / (Bx1) = (BxD)
            M = S - 2 * torch.matmul(u.unsqueeze(-1),
                                     v.unsqueeze(1))  # the matmul performs the outer product of u and v
            return M

        S = reflection(torch.eye(D).repeat(B, 1, 1).to(p.device), p + q)  # S @ p = -q, S @ q = -p
        R = reflection(S, q)  # R @ p = q, R.T @ q = p

        return R

    @classmethod
    def construct_rotation_isoms(cls, rotations):
        rots = rotations.reshape(-1, 3, 3)
        isoms = torch.eye(5, device=rots.device).tile(rots.shape[0], 1, 1)
        isoms[:, :3, :3] = rots
        return isoms

    @staticmethod
    def cross_operator(v: torch.Tensor):
        """ Batched cross operator, v.shape = (N,3) -> y = [v]_x.shape = (N, 3, 3) """
        B = v.shape[0]
        y = v.new_zeros(B, 3, 3)
        y[:, 0, 1] = -v[:, 2]
        y[:, 0, 2] = v[:, 1]
        y[:, 1, 0] = v[:, 2]
        y[:, 1, 2] = -v[:, 0]
        y[:, 2, 0] = -v[:, 1]
        y[:, 2, 1] = v[:, 0]

        return y

    @classmethod
    def rotation_matrix(cls, axis_angle):
        """ Batched axis-angle to rotation matrix """
        return torch.matrix_exp(cls.cross_operator(axis_angle))

    @classmethod
    def unembed_points(cls, embedded_points):
        """
        Performs a mapping that is inverse to conformal embedding.

        Args:
            embedded_points: points embedded in R^{5}, an array of shape (num_points, 5).
        Returns:
            points:          3D points, an array of shape (num_points, 3).
        """
        # p-normalize points, i.e., divide by the last element.
        # The first three elements are now Euclidean R^{3} coordinates:
        epsilon = 1e-12
        points = embedded_points[:, :3] / (embedded_points[:, -1:] + epsilon)
        return points

    @classmethod
    def transform_points(cls, points, transformation):
        """
        Apply one or a batch of isomorphism transformations to embedded points.

        Args:
            points:              points embedded in R^{5}, an array of shape (num_points, 5);
            transformation:      an array of shape (num_points, 5, 5) or (1, 5, 5) or (5, 5)
        Returns:
            transformed points:  a tensor of the same shape as the input points.
        """
        T = transformation.reshape(-1, 5, 5)  # for performing matmul properly
        X = points.unsqueeze(-1)  # for performing matmul properly
        Y = torch.matmul(T, X)  # transform points
        Y = Y.squeeze(-1)  # to input shape

        return Y

    def forward(self, spheres, prev_rotation_0=None, verbose=False):
        N = spheres.shape[0]
        spheres = spheres.reshape(-1, 5)

        if self.use_prev_init_rotations and prev_rotation_0 is not None:
            prev_rotation_0_isom = self.construct_rotation_isoms(prev_rotation_0)
            spheres = self.transform_points(spheres, prev_rotation_0_isom.to(spheres.device))

        # Step 1) compute the rotations R_O^k, i.e., from the original sphere centers to (1,1,1):
        centers = self.unembed_points(spheres)  # (n_spheres x 3)
        centers_n = F.normalize(centers, dim=1)  # for computing the cross_product appropriately

        ones = F.normalize(self.ones_vec.unsqueeze(0).expand(N, -1), dim=1)

        # compute initial rotations (from sphere centers to (1,1,1))
        rotations_0 = self.compute_rotation_from_two_points(centers_n, ones)

        rotations_0_isom = self.construct_rotation_isoms(rotations_0)

        # rotate the original spheres into (1,1,1) (in R^5):
        rotated_spheres = self.transform_points(spheres, rotations_0_isom)

        # Step 2) get the tetrahedron rotations R_{Ti},
        # i.e. the rotations transforming (1, 1, 1) into the other three vertices
        tetra_rotations = self.tetra_rotations

        # Step 3) construct the filter banks, a.k.a, the steerable 3D spherical neurons B(S):
        # rotate *directly* in the conformal R^5 space
        # (already includes the INVERSE of the rotations from the original sphere centers to (1,1,1)):

        # Tiling so that the four tetra rotations are applied to all N points:
        # tile N times as 1,2,3,4, 1,2,3,4, ... :
        tetra_rotations4 = tetra_rotations.tile(N, 1, 1, 1).view(4 * N, 3, 3)
        # tile 4 times as 1,1,1,1, 2,2,2,2, ... to match the tetra rotations:
        rotated_spheres4 = rotated_spheres.unsqueeze(1).tile(1, 4, 1).view(4 * N, 5)
        rotations_0_4 = rotations_0.unsqueeze(1).tile(1, 4, 1, 1).view(4 * N, 3, 3)

        # actually construct the filter bank
        b = self.construct_rotation_isoms(rotations_0_4.transpose(1, 2) @ tetra_rotations4)
        filter_banks = transform_points(rotated_spheres4, b).unsqueeze(-1)

        if verbose:
            print('\noriginal centers:\n', centers)
            print('\nrotated original spheres[0]:', rotated_spheres[0])
            print('\noriginal spheres[0]:', spheres[0])
            print('\nrotations_0[0]:', rotations_0[0])

        return rotations_0, filter_banks
