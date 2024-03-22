# Copyright (c) 2024 Pavlo Melnyk and Andreas Robinson, <name>.<surname>@liu.se
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import random_split
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule, LightningDataModule


def make_object(**kwargs):
    cls = kwargs['cls']
    kwargs = {k: v for k, v in kwargs.items() if k != 'cls'}
    obj = cls(**kwargs)
    return obj


def make_function(**kwargs):
    function = kwargs['callable']
    kwargs = {k: v for k, v in kwargs.items() if k != 'callable'}

    def fn(*args):
        return function(*args, **kwargs)

    return fn


def build_datasets(train_dset, train_xform,
                   val_dset, val_xform,
                   test_dset, test_xform,
                   batch_size=1, num_workers=0):
    if train_dset is not None:
        train_dset = make_object(**train_dset)
        if train_xform is not None:
            train_dset.transform = make_object(**train_xform)

    if val_dset is not None:
        val_dset = make_object(**val_dset)
        if val_xform is not None:
            val_dset.transform = make_object(**val_xform)

    if test_dset is not None:
        test_dset = make_object(**test_dset)
        if test_xform is not None:
            test_dset.transform = make_object(**test_xform)

    data = LightningDataModule.from_datasets(train_dataset=train_dset, val_dataset=val_dset, test_dataset=test_dset,
                                             batch_size=batch_size, num_workers=num_workers)
    return data


class ModelRunner(LightningModule):

    def __init__(self, run, model, dsets, tforms, optim, sched, loss=None, pred=None, dry_run=False, test_name=None,
                 init_rotations_log_path=None):
        super().__init__()
        self.save_hyperparameters(ignore=['test_name'])

        self.run_args = run
        self.optim_args = optim
        self.sched_args = sched
        self.dry_run = dry_run
        self.init_rotations_log_path = init_rotations_log_path

        if loss is None:
            loss = dict(callable=F.cross_entropy, reduction="mean")
        if pred is None:
            pred = dict(callable=torch.argmax, dim=1)

        self.loss_fn = make_function(**loss)
        self.pred_fn = make_function(**pred)

        self.model = make_object(**model)
        dm = build_datasets(dsets.get('train', None), tforms.get('train', None),
                            dsets.get('val', None), tforms.get('val', None),
                            dsets.get('test', None), tforms.get('test', None),
                            batch_size=run['batch_size'], num_workers=run['num_workers'])
        self.datamodule = dm

        if 'train' in dsets:
            self.train_accuracy = Accuracy("multiclass", num_classes=dm.train_dataloader().dataset.num_classes)
        if 'val' in dsets:
            self.val_accuracy = Accuracy("multiclass", num_classes=dm.val_dataloader().dataset.num_classes)
        if 'test' in dsets:
            self.test_accuracy = Accuracy("multiclass", num_classes=dm.test_dataloader().dataset.num_classes)
            self.test_name = test_name

        self.spheres_log = dict(centers=[], radii=[], gamma=[])

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):

        if self.dry_run:
            self.log_dict({"train/acc": 0.0}, prog_bar=True, sync_dist=True, on_epoch=True)
            return None

        x, y = batch
        x = x.permute(0, 2, 1)  # B x 3 x N

        r = self(x)
        loss = self.loss_fn(r, y)
        preds = self.pred_fn(r)
        self.train_accuracy.update(preds, y)

        self.log_dict({"train/loss": loss, "train/acc": self.train_accuracy}, prog_bar=True, sync_dist=True,
                      on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):

        if self.dry_run:
            self.log_dict({"val/acc": 0.0}, prog_bar=True, sync_dist=True, on_epoch=True)
            return None

        x, y = batch
        x = x.permute(0, 2, 1)  # B x 3 x N

        r = self(x)
        loss = self.loss_fn(r, y)
        preds = self.pred_fn(r)
        self.val_accuracy.update(preds, y)

        self.log_dict({"val/loss": loss, "val/acc": self.val_accuracy}, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):

        name = f"test-{self.test_name}" if self.test_name is not None else "test"

        if self.dry_run:
            self.log_dict({f"{name}/acc": 0.0}, prog_bar=True, sync_dist=True, on_epoch=True)
            return None

        x, y = batch
        x = x.permute(0, 2, 1)  # B x 3 x N

        r = self(x)
        loss = self.loss_fn(r, y)
        preds = self.pred_fn(r)
        self.test_accuracy.update(preds, y)

        self.log_dict({f"{name}/loss": loss, f"{name}/acc": self.test_accuracy}, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):

        if self.optim_args['params_structure'] == "basic":
            params = self.model.parameters()
        else:
            raise ValueError

        optim_args = {k: v for k, v in self.optim_args.items() if k != 'params_structure'}
        optimizer = make_object(params=params, **optim_args)
        scheduler = make_object(optimizer=optimizer, **self.sched_args)
        return [optimizer], [scheduler]

    def on_train_start(self) -> None:
        n = sum(p.numel() for p in self.model.parameters())
        print(f"\n{n} model parameters")
        self.log("model/parameters", float(n), rank_zero_only=True, sync_dist=True)
