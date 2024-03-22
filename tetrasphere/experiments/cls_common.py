# Copyright (c) 2024 Pavlo Melnyk and Andreas Robinson, <name>.<surname>@liu.se
# SPDX-License-Identifier: MIT

from pathlib import Path
from datetime import datetime


def train_one_variant(file, full_name,
                      num_epochs, seed, model_type, dset_name, rot,
                      translate_points=True,  # Translate points in data augmentation.
                      dset_split=("main_split", "obj"),
                      model_kwargs={},
                      loss_type="smooth_labels",
                      batch_size=32,
                      dry_run=False,
                      run_test=False, test_ckpt=None):
    import torch
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch import optim

    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.loggers import WandbLogger, CSVLogger, TensorBoardLogger
    from pytorch_lightning.loggers.logger import DummyLogger

    import wandb

    from tetrasphere.config import Environment
    from tetrasphere.utils import cal_loss
    from tetrasphere.runner import ModelRunner
    from tetrasphere.dataset import PointcloudRandomTransform1, ModelNet40, ScanObjectNN
    from tetrasphere.models.tetrasphere import TetraSphere_cls
    from tetrasphere.models.eqcnn import EQCNN_cls

    # Note: All evaluation variations are set after trainer.fit(), below

    logger_args = dict(name=f'{Path(file).stem}_{datetime.now().strftime("%Y-%m-%d_%H%M")}',
                       project="tetrasphere", entity="<your-name-here>", save_dir=str(Environment.log_path),
                       log_model=True)
    run_args = dict(seed=seed, epochs=num_epochs, base_lr=0.001, batch_size=batch_size,
                    num_workers=4 if not dry_run else 0, deterministic=True)

    if dset_name == 'sobjnn':
        split, vari = dset_split
        dset_args = dict(
            train=dict(cls=ScanObjectNN, partition='train', split=split, variant=vari, shuffle=True, normalize=True),
            val=dict(cls=ScanObjectNN, partition='test', split=split, variant=vari, shuffle=False, normalize=True),
            test=dict(cls=ScanObjectNN, partition='test', split=split, variant=vari, shuffle=False, normalize=True))
        num_classes = 15
    elif dset_name == 'mn40':
        dset_args = dict(train=dict(cls=ModelNet40, partition='train', variant='hdf5'),
                         val=dict(cls=ModelNet40, partition='test', variant='hdf5'),
                         test=dict(cls=ModelNet40, partition='test', variant='hdf5'))
        num_classes = 40
    elif dset_name == 'mn40c':
        corruption, severity = dset_split.split(":")
        dset_args = dict(val=dict(cls=ModelNet40, partition='test', variant='40c', corruption=corruption, severity=severity),
                         test=dict(cls=ModelNet40, partition='test', variant='40c', corruption=corruption, severity=severity))
        num_classes = 40
    else:
        raise ValueError

    if model_type == 'eqcnn':
        model_args = dict(cls=EQCNN_cls, output_channels=num_classes, k=20, pooling='mean',
                          add_x_norm=model_kwargs["add_x_norm"], no_mean=model_kwargs["no_mean"])
    elif model_type == 'tetrasphere':
        model_args = dict(cls=TetraSphere_cls,
                          k=20, init_mode=None, output_channels=num_classes,
                          fix_tetrasphere=model_kwargs.get("fix_tetrasphere", False),
                          normalized_spheres=model_kwargs.get("normalized_spheres", False),
                          num_spheres=model_kwargs["num_spheres"],
                          sphere_pooling=model_kwargs.get("sphere_pooling", "equi_max_norm"))
    else:
        raise ValueError(f"Undefined model type {model_type}")

    tform_args = dict(
        train=dict(cls=PointcloudRandomTransform1, rot=rot, scale=True, trans=translate_points, shuffle=False),
        val=dict(cls=PointcloudRandomTransform1, rot='so3', scale=False, trans=False, shuffle=False),
        test=dict(cls=PointcloudRandomTransform1, rot='so3', scale=False, trans=False, shuffle=False))

    lr = run_args['base_lr']
    optim_args = dict(cls=optim.SGD, params_structure="basic", lr=lr * 100, momentum=0.9, weight_decay=1e-4)
    sched_args = dict(cls=CosineAnnealingLR, T_max=run_args['epochs'], eta_min=lr)

    if loss_type == "smooth_labels":
        loss_args = dict(callable=cal_loss, smoothing=True)
        pred_args = dict(callable=torch.argmax, dim=1)

    elif loss_type == "cross_entropy":
        loss_args = dict(callable=F.cross_entropy, reduction='mean')
        pred_args = dict(callable=torch.argmax, dim=1)

    else:
        raise ValueError

    log_path = Environment.log_path
    log_path.mkdir(exist_ok=True)

    torch.backends.cudnn.deterministic = run_args['deterministic']
    torch.backends.cudnn.benchmark = not run_args['deterministic']

    loggers = []

    if not run_test:
        # Uncomment for Weights and Biases logging. Also, edit logger_args["entity"] for your project.
        # if Environment.wandb_api_key.exists():
        #     with open(Environment.wandb_api_key) as f:
        #         wandb.login(key=f.readline().strip())
        #     loggers += [WandbLogger(**logger_args)]
        pass
    else:
        loggers += [DummyLogger()]

    save_dir = Path(logger_args['save_dir']) / "tetrasphere"
    csv_logger_args = dict(name=full_name, save_dir=str(save_dir / "csv"))
    loggers += [CSVLogger(**csv_logger_args)]
    tb_logger_args = dict(name=full_name, save_dir=str(save_dir / "tensorboard"))
    # loggers += [TensorBoardLogger(**tb_logger_args)]

    seed_everything(run_args['seed'], workers=True)
    if not run_test:
        init_rotations_log_path = Path(
            save_dir) / f'{full_name}_init_rotations_{datetime.now().strftime("%Y-%m-%d_%H%M")}'
    else:
        init_rotations_log_path = None
    runner = ModelRunner(run_args, model_args, dset_args, tform_args, optim_args, sched_args,
                         loss=loss_args, pred=pred_args,
                         dry_run=dry_run, test_name='so3', init_rotations_log_path=init_rotations_log_path)

    if not run_test:

        checkpoint_cb = ModelCheckpoint(every_n_epochs=125, filename="{epoch:03d}", save_top_k=-1)
        trainer = Trainer(max_epochs=run_args['epochs'], accelerator="gpu", devices=-1, logger=loggers,
                          strategy=DDPStrategy(find_unused_parameters=False), sync_batchnorm=True,
                          callbacks=[TQDMProgressBar(refresh_rate=10),
                                     LearningRateMonitor(logging_interval='step'),
                                     checkpoint_cb])

        trainer.fit(runner, datamodule=runner.datamodule)
        trainer.test(runner, datamodule=runner.datamodule)

    else:

        trainer = Trainer(max_epochs=run_args['epochs'], accelerator="gpu", devices=1, logger=loggers,
                          strategy=DDPStrategy(find_unused_parameters=False), sync_batchnorm=True,
                          callbacks=[TQDMProgressBar(refresh_rate=10)])

    # Tests

    ckpt_epoch = num_epochs

    if run_test and test_ckpt is not None:
        ckpt_path = Path(test_ckpt)
    else:
        ckpt_path = Path(checkpoint_cb.dirpath) / f"epoch={ckpt_epoch - 1:03d}.ckpt"

    for rot in ['z', 'so3', 'o3']:
        tform_args['test'] = dict(cls=PointcloudRandomTransform1, rot=rot, scale=False, trans=False, shuffle=False)
        runner = ModelRunner(run_args, model_args, dset_args, tform_args, optim_args, sched_args,
                             loss=loss_args, pred=pred_args, test_name=rot)
        state = torch.load(ckpt_path)['state_dict']
        state = {key[6:]: val for key, val in state.items()}
        runner.model.load_state_dict(state, strict=False)
        trainer.test(runner, datamodule=runner.datamodule)
