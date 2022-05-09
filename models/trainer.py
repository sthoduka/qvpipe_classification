import math
import os
import copy
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import torchvision
import pytorch_lightning as pl
import torchnet.meter as meter

from datasets.qvpipe_dataset import QVPipeDataset

import pdb

import torch
from torch import nn
class TModel7(nn.Module):
    def __init__(self, num_images=5):
        super().__init__()
        transformer_dim = 512
        model = models.resnet18(pretrained=True)
        self.cls_model = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Sequential(nn.LayerNorm(transformer_dim),
                                        nn.Linear(transformer_dim, 17))
        num_heads = 4
        num_layers = 8
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                transformer_dim, num_heads, dim_feedforward=1024
            ),
            num_layers,
        )
        self.class_token = nn.Parameter(torch.randn(1, transformer_dim))
        self.num_images = num_images

    def forward(self, img):
        B, T, C, H, W = img.size()
        img = img.view(B*T, C, H, W)
        out = self.cls_model(img)
        out = out.squeeze(2).squeeze(2)
        out = out.view(B, T, -1)
        class_token = self.class_token.repeat(B, 1).unsqueeze(1)
        enc_inp = torch.cat((class_token, out), axis=1).permute(1, 0, 2) # T X B X 512
        enc_out = self.encoder(enc_inp)
        out = enc_out[0].squeeze(0) # get the output corresponding to the class_token
        out = self.classifier(out)
        return out

class TModel2(nn.Module):
    def __init__(self, num_images=5):
        super().__init__()
        output_dim = 512
        model = models.resnet18(pretrained=True)
        self.cls_model = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Sequential(nn.Linear(output_dim*num_images, 1024, bias=True),
                                        nn.Hardswish(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(1024, 17, bias=True))
        self.num_images = num_images

    def forward(self, img):
        B, T, C, H, W = img.size()
        img = img.view(B*T, C, H, W)
        out = self.cls_model(img)
        out = out.squeeze(2).squeeze(2)
        out = out.view(B, -1)
        out = self.classifier(out)
        return out

class QVPipeTrainer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = TModel7(num_images=5)
        self.meter = meter.mAPMeter() # keep track of mean average precision
        self.cls_meter = meter.APMeter() # keep track of class-wise average precision

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--crop_size', type=int, default=450, help='Size to crop to')
        parser.add_argument('--resize_size', type=int, default=224, help='Size to resize to')
        parser.add_argument('--group', type=int, default=1, help='1, 2, 3, 4, 5')
        parser.add_argument('--num_key_frames', type=int, default=5, help='num key frames')
        return parser

    def forward(self, img):
        out = self.model(img)
        return out

    def loss_function(self, out, label):
        return torchvision.ops.sigmoid_focal_loss(out, label, reduction='mean')

    def training_step(self, batch, batch_idx):
        img, label, _ = batch
        out = self(img)
        loss = self.loss_function(out, label)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        img, label, _ = batch
        out = self(img)
        self.meter.add(out, label)
        self.cls_meter.add(out, label)
        loss = self.loss_function(out, label)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        img, label, _ = batch
        out = self(img)
        loss = self.loss_function(out, label)
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        mAP = self.meter.value()
        AP = self.cls_meter.value()
        self.log("val_loss", avg_loss)
        self.log("val_mAP", mAP)
        for idx, cls_ap in enumerate(AP):
            self.log("val_AP_%02d" % idx, cls_ap)
        self.meter.reset()
        self.cls_meter.reset()

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("train_loss", avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9, weight_decay=1e-3)
        if self.hparams.group == 4:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate, steps_per_epoch=len(self.train_dataloader()), epochs=self.hparams.max_epochs)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(300),
                transforms.RandomCrop(self.hparams.crop_size),
                transforms.Resize(self.hparams.resize_size),
                transforms.RandomAdjustSharpness(1.5),
                transforms.RandomAutocontrast(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomErasing(),
                transforms.GaussianBlur(kernel_size=3),
                transforms.RandomRotation(30),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.75, 1.25)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        train_dataset = QVPipeDataset(self.hparams.video_root, dataset_type='train', group=self.hparams.group, num_key_frames=self.hparams.num_key_frames, transform=transform)
        if self.training:
            return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                               shuffle=True, num_workers=self.hparams.n_threads, pin_memory=False)
        else:
            return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                               shuffle=False, num_workers=self.hparams.n_threads, pin_memory=False)

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(300),
                transforms.CenterCrop(self.hparams.crop_size),
                transforms.Resize(self.hparams.resize_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        val_dataset = QVPipeDataset(self.hparams.video_root, dataset_type='val', group=self.hparams.group, transform=transform)
        return torch.utils.data.DataLoader(val_dataset, batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.n_threads, pin_memory=False)
    def test_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(300),
                transforms.CenterCrop(self.hparams.crop_size),
                transforms.Resize(self.hparams.resize_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        test_dataset = QVPipeDataset(self.hparams.video_root, dataset_type='test', transform=transform)
        return torch.utils.data.DataLoader(test_dataset, batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.n_threads, pin_memory=False)

