from dataclasses import dataclass
import os

import torch
from torch.utils.data import Dataset
import numpy as np

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py

from tetrasphere.pytorch3d_transforms import Transform3d, random_rotations
from tetrasphere.config import Environment


@dataclass
class PointcloudRandomTransform1:
    rot: str = 'I'  # Rotation variant (z, so3 or I (or aligned), or o3 (with reflections))
    scale: bool = False  # Whether to apply random (0.66 - 1.5) scaling after rotation
    trans: bool = False  # Whether to apply random (-0.2 - 0.2) translation after scaling
    shuffle: bool = False  # Whether to shuffle the points after transforming them
    num_points: int = 1024

    # if center and normalize to a unit sphere:
    center_normalize: bool = False

    def __call__(self, pcd: torch.Tensor):

        # if self.center_normalize:

        T = Transform3d(device=pcd.device)

        if self.scale:
            T = T.scale(*np.random.uniform(low=2. / 3., high=3. / 2., size=[3]))

        if self.trans:
            T = T.translate(*np.random.uniform(low=-0.2, high=0.2, size=[3]))

        if self.rot == 'x':
            T = T.rotate_axis_angle(angle=torch.rand(1) * 360, axis="X", degrees=True)
        elif self.rot == 'y':
            T = T.rotate_axis_angle(angle=torch.rand(1) * 360, axis="Y", degrees=True)
        elif self.rot == 'z':
            T = T.rotate_axis_angle(angle=torch.rand(1) * 360, axis="Z", degrees=True)
        elif self.rot == 'so3':
            T = T.rotate(R=random_rotations(1))
        elif self.rot == 'o3':
            T = T.rotate(R=random_rotations(1))
            # select a reflection axis:
            xyz = np.array([-1., 1., 1.])
            np.random.shuffle(xyz)
            T = T.scale(*xyz)
        elif self.rot == 'I' or self.rot == 'aligned':
            pass
        else:
            raise ValueError

        pcd = T.transform_points(pcd)

        if len(pcd.shape) == 2:
            pcd = pcd[:self.num_points]
        else:
            # len(pcd.shape) == 3
            pcd = pcd[:, :self.num_points]

        if self.shuffle:
            pcd = pcd[torch.randperm(pcd.shape[0])]

        return pcd


class ScanObjectNN(Dataset):
    label_names = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table",
                   "bed", "pillow", "sink", "sofa", "toilet"]

    @staticmethod
    def pc_normalize(pc):
        centroid = torch.mean(pc, dim=0)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1)))
        pc = pc / m
        return pc

    def __init__(self, partition, split='main_split_nobg', variant='obj', transform: callable = None, shuffle=False,
                 normalize=False):
        super().__init__()

        # https://hkust-vgd.ust.hk/scanobjectnn/

        assert partition in ('train', 'test')
        assert split in ('main_split', 'main_split_nobg', 'split1', 'split1_nobg', 'split2', 'split2_nobg',
                         'split3', 'split3_nobg', 'split4', 'split4_nobg')

        self.partition = partition
        self.split = split
        self.variant = variant
        self.transform = transform
        self.shuffle = shuffle
        self.normalize = normalize

        dset_root = Environment.dset_path

        import os
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        import h5py

        partition = 'training' if partition == 'train' else partition
        variant = dict(obj='objectdataset', pb_t25='objectdataset_augmented25_norot',
                       pb_t25_r='objectdataset_augmented25_rot', pb_t50_r='objectdataset_augmentedrot',
                       pb_t50_rs='objectdataset_augmentedrot_scale75')[variant.lower()]
        split_name = f"{split}/{partition}_{variant}"
        h5_name = dset_root / "scanobjectnn" / (split_name + ".h5")

        with h5py.File(h5_name) as f:
            self.data = f['data'][:].astype('float32')
            assert len(self.data) > 0
            self.label = f['label'][:].astype('int64')

        self.data = torch.from_numpy(self.data)
        self.label = torch.from_numpy(self.label)

    @property
    def num_classes(self):
        return 15

    def __getitem__(self, item):

        pcd = self.data[item]
        label = self.label[item]

        pt_idxs = np.arange(0, pcd.shape[0])  # 2048
        if self.partition == "train" and self.shuffle:
            np.random.shuffle(pt_idxs)

        pcd = pcd[pt_idxs][:1024].clone()

        if self.normalize:
            pcd[:, 0:3] = self.pc_normalize(pcd[:, 0:3])

        if self.transform is not None:
            pcd = self.transform(pcd)

        return pcd, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40(Dataset):

    def __init__(self, partition, variant='hdf5', transform: callable = None, corruption=None, severity=None):
        super().__init__()

        self.partition = partition
        self.variant = variant
        self.transform = transform

        self.corruption = corruption
        self.severity = severity

        dset_root = Environment.dset_path

        if variant == 'hdf5':

            import os
            os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
            import h5py

            dset_dir = dset_root / "modelnet40_ply_hdf5_2048"

            all_data = []
            all_labels = []

            for h5_name in dset_dir.glob(f'*{partition}*.h5'):
                with h5py.File(h5_name) as f:
                    data = f['data'][:].astype('float32')
                    assert len(data) > 0
                    labels = f['label'][:].astype('int64')
                all_data.append(data)
                all_labels.append(labels)

            self.data = np.concatenate(all_data, axis=0)
            self.label = np.concatenate(all_labels, axis=0)

        else:
            raise ValueError(f"Unknown variant {variant}")

        self.data = torch.from_numpy(self.data)
        self.label = torch.from_numpy(self.label)

    @property
    def num_classes(self):
        return 40

    def __getitem__(self, item):

        pcd = self.data[item]
        label = self.label[item]

        if self.transform is not None:
            pcd = self.transform(pcd)

        return pcd, label.squeeze()

    def __len__(self):
        return self.data.shape[0]


class ShapeNetPart(Dataset):

    def load_data(self, partition):

        dset_root = Environment.dset_path / "shapenet_part_seg"

        all_data = []
        all_label = []
        all_seg = []
        if partition == 'trainval':
            file = list(dset_root.glob('*train*.h5')) + list(dset_root.glob('*val*.h5'))
        else:
            file = list(dset_root.glob(f'*{partition}*.h5'))

        for h5_name in file:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            seg = f['pid'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
            all_seg.append(seg)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        all_seg = np.concatenate(all_seg, axis=0)

        return all_data, all_label, all_seg

    def __init__(self, num_points, partition='train', class_choice=None, transform=None):

        self.partition = partition
        self.transform = transform

        self.data, self.label, self.seg = self.load_data(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition
        self.class_choice = class_choice
        self.transform = transform

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            # pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]

        pointcloud = torch.from_numpy(pointcloud)
        if self.transform is not None:
            pointcloud = self.transform(pointcloud)

        return pointcloud, label.squeeze(), seg

    def __len__(self):
        return self.data.shape[0]

    @property
    def num_classes(self):
        return len(self.cat2id)
