"""
Based on the implementation of
    @Author: An Tao
    @Contact: ta19@mails.tsinghua.edu.cn
    @File: main_partseg.py
"""

import os
import argparse
from pathlib import Path
import sys

project_path = [p.parent for p in Path(__file__).resolve().parents if p.name == 'tetrasphere'][0]
sys.path.append(str(project_path))


def init_logging(args):
    import torch
    from tetrasphere.config import Environment

    if args.dry_run:
        print("****************** DRY RUN ********************")

    logger = None

    # Uncomment for Weights and Biases logging:

    # if Environment.wandb_api_key.exists():
    #     import wandb
    #     logger = wandb
    #     with open(Environment.wandb_api_key) as f:
    #         logger.login(key=f.readline().strip())

    #     logger.init(
    #         # Set the project where this run will be logged
    #         project="<your-name-here>",
    #         name=args.exp_name,
    #         # Track hyperparameters and run metadata
    #         config=vars(args),
    #     )
    io = sys.stdout
    print(str(args), file=io)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        print(f'Using GPU : {torch.cuda.current_device()} out of {torch.cuda.device_count()} devices', file=io)
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU', file=io)

    return logger, io


def make_model(args, seg_num_all):
    from tetrasphere.models.eqcnn import EQCNN_partseg
    from tetrasphere.models.tetrasphere import TetraSphere_partseg

    if args.model == 'eqcnn':
        model = EQCNN_partseg(k=args.k, pooling=args.pooling, seg_num_all=seg_num_all)
    elif args.model == 'ts_partseg':
        mdl_args = dict(num_spheres=args.num_spheres, k=args.k, pooling=args.pooling, seg_num_all=seg_num_all,
                        sphere_pooling=args.sphere_pooling)
        print("Args: ", mdl_args)
        model = TetraSphere_partseg(**mdl_args)
    else:
        raise Exception("Not implemented")

    # print(str(model))
    return model


def train(args, io, logger):
    from tqdm import tqdm
    import torch
    from torch import nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR  # noqa
    from torch.utils.data import DataLoader

    import numpy as np
    import sklearn.metrics as metrics

    from tetrasphere.config import Environment
    from tetrasphere.dataset import ShapeNetPart
    from tetrasphere.dataset import PointcloudRandomTransform1
    from tetrasphere.utils import WorkerSeed, calculate_shape_IoU, cal_loss

    torch.cuda.empty_cache()

    if logger is not None:
        logger.init(
            # Set the project where this run will be logged
            project="ps-tsphere",
            # Track hyperparameters and run metadata
            config=vars(args))

    seeder = WorkerSeed(args.seed)

    train_transform = PointcloudRandomTransform1(rot=args.train_rot, num_points=args.num_points)
    train_dataset = ShapeNetPart(partition='trainval', num_points=args.num_points, class_choice=args.class_choice,
                                 transform=train_transform)
    test_transform = PointcloudRandomTransform1(rot='so3', num_points=args.num_points)
    test_dataset = ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice,
                                transform=test_transform)

    if (len(train_dataset) < 100):
        drop_last = False
    else:
        drop_last = True
    num_workers = 0
    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=args.batch_size, shuffle=True,
                              drop_last=drop_last, worker_init_fn=seeder)
    test_loader = DataLoader(test_dataset, num_workers=num_workers, batch_size=args.test_batch_size, shuffle=True,
                             drop_last=False, worker_init_fn=seeder)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    seg_num_all = train_loader.dataset.seg_num_all
    seg_start_index = train_loader.dataset.seg_start_index
    model = make_model(args, seg_num_all).to(device)

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)
    else:
        raise ValueError(args.scheduler)

    criterion = cal_loss

    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        for data, label, seg in tqdm(train_loader):

            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)

            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1, 1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]  # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()  # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))  # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))  # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            train_label_seg.append(label.reshape(-1))

            if args.dry_run:
                break

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_label_seg = np.concatenate(train_label_seg)
        train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, args.class_choice)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (
            epoch, train_loss * 1.0 / count, train_acc, avg_per_class_acc, np.mean(train_ious))

        print(outstr, file=io)
        if logger is not None:
            logger.log(
                dict(epoch=epoch, loss=train_loss * 1.0 / count, train_acc=train_acc, train_avg_acc=avg_per_class_acc,
                     train_iou=np.mean(train_ious)))

        torch.cuda.empty_cache()

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        test_label_seg = []
        for data, label, seg in test_loader:

            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
            data = data.permute(0, 2, 1)

            batch_size = data.size()[0]
            seg_pred = model(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1, 1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            test_label_seg.append(label.reshape(-1))

            if args.dry_run:
                break

        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_label_seg = np.concatenate(test_label_seg)
        test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (
            epoch, test_loss * 1.0 / count, test_acc, avg_per_class_acc, np.mean(test_ious))

        print(outstr, file=io)
        if logger is not None:
            logger.log(
                dict(epoch=epoch, loss=test_loss * 1.0 / count, test_acc=test_acc, test_avg_acc=avg_per_class_acc,
                     test_iou=np.mean(test_ious)))

        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            model_file = Environment.log_path / f'partseg/{args.exp_name}/model.t7'
            model_file.parent.mkdir(exist_ok=True, parents=True)
            sdict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

            torch.save(sdict, str(model_file))
            args.model_path = str(model_file)

        torch.cuda.empty_cache()

        if args.dry_run:
            break


def test(args, io, logger):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    import sklearn.metrics as metrics
    import numpy as np

    from tetrasphere.utils import WorkerSeed, calculate_shape_IoU
    from tetrasphere.dataset import PointcloudRandomTransform1, ShapeNetPart

    seeder = WorkerSeed(args.seed)
    test_transform = PointcloudRandomTransform1(rot=args.test_rot, num_points=2048)
    test_dataset = ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice,
                                transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, drop_last=False,
                             worker_init_fn=seeder)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    seg_num_all = test_loader.dataset.seg_num_all
    seg_start_index = test_loader.dataset.seg_start_index
    model = make_model(args, seg_num_all).to(device)

    model.load_state_dict(torch.load(args.model_path), strict=False)
    model = nn.DataParallel(model)
    model = model.eval()

    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []

    torch.cuda.empty_cache()

    for data, label, seg in test_loader:

        seg = seg - seg_start_index
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)

        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        seg_pred = model(data, label_one_hot)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        pred = seg_pred.max(dim=2)[1]
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        test_label_seg.append(label.reshape(-1))

        if args.dry_run:
            break

    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label_seg = np.concatenate(test_label_seg)
    test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
    outstr = f'Test train-{args.train_rot} :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (
        test_acc, avg_per_class_acc, np.mean(test_ious).item())

    log = dict()
    rot = args.test_rot
    log[f"test-{rot}/acc"] = test_acc
    log[f"test_avg_acc/{rot}"] = avg_per_class_acc
    log[f"test_iou/{rot}"] = np.mean(test_ious).item()
    print(outstr, file=io)
    if logger is not None:
        logger.log(log)


def initialize(args):
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='ts_shapenet_debug', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='ts_partseg', metavar='N',
                        choices=['dgcnn', 'eqcnn', 'ts_partseg'])
    parser.add_argument('--dataset', type=str, default='shapenetpart', metavar='N',
                        choices=['shapenetpart'])
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop',
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--dry_run', action=argparse.BooleanOptionalAction,
                        help='train and test one batch')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--c_prime', type=int, default=3)
    parser.add_argument('--num_spheres', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--pooling', type=str, default='mean', metavar='N',
                        choices=['mean', 'max'],
                        help='VNN only: pooling method.')
    parser.add_argument('--sphere_pooling', type=str, default='max_norm')
    parser.add_argument('--train_rot', type=str, default='z', choices=['z', 'so3'],
                        help='Training data rotation augmentation type')

    args = parser.parse_args(args)
    logger, io = init_logging(args)

    return args, logger, io


def main(args, array_id):
    args, logger, io = initialize(args)
    train(args, io, logger)
    for test_rot in ['z', 'so3', 'o3']:
        args.test_rot = test_rot
        test(args, io, logger)
    if logger is not None:
        logger.finish()


if __name__ == "__main__":

    epochs = 200
    epochs_per_hour = 5  # < 12 minutes per epoch
    bs = 16

    experiments = []
    train_rot = 'z'
    model = "ts_partseg"
    C_prime = 3
    spool = "equi_max_norm"

    for seed in [1, 2, 3]:
        for num_spheres in [16, 8, 4, 2, 1]:
            name = f"EQ1_cvpr2024_partseg_{model}_{train_rot}_K{num_spheres}_{spool}_seed{seed}"
            ex = f"--exp_name {name} --epochs {epochs} --batch_size {bs} --train_rot {train_rot} " \
                 f"--model {model} --c_prime {C_prime} --num_spheres {num_spheres} --sphere_pooling {spool}"
            experiments.append(ex)

    array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', "-1"))
    if array_id == -1:
        for args in experiments:
            args = args  # + " --dry_run --batch_size 8 --test_batch_size 8"
            main(args.split(), array_id)
    else:
        args = experiments[array_id]
        main(args.split(), array_id)
