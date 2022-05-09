import pytorch_lightning as pl
from argparse import ArgumentParser
from models import trainer
import pdb
import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
import json
from torchvision import transforms
import cv2
import json
import matplotlib.pyplot as plt
from sklearn import metrics
import json
import torchnet.meter as meter


def main():
    parser = ArgumentParser()

    parser.add_argument('--video_root', default='', type=str, help='Root path of input videos')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint file')

    parser = trainer.QVPipeTrainer.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    device = 'cuda:0'

    model = trainer.QVPipeTrainer.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.hparams.batch_size = args.batch_size
    model.hparams.n_threads = args.n_threads


    model.eval()

    test_loader = model.test_dataloader()
    total = 0

    output = {}

    mapmeter = meter.mAPMeter()
    mapmeter_norm = meter.mAPMeter()

    with torch.no_grad():
        for batch in test_loader:
            imgs, label, names = batch
            outs = []
            for img in imgs:
                B, T, C, H, W = img.size()
                img = img.to(device)
                out = model(img)
                outs.append(out)
            out = torch.stack(outs, axis=1)
            out = out.mean(axis=1)
            mapmeter.add(out, label)
            out = F.sigmoid(out)
            mapmeter_norm.add(out, label)
            for idx, nm in enumerate(names):
                res = out[idx].detach().cpu().numpy().tolist()
                res = [round(dd, 5) for dd in res]
                output[nm] = res
            total += 1
            print("%d / %d" % (total * args.batch_size, len(test_loader.dataset)))
    print('mAP: ', mapmeter.value())
    print('mAP_norm: ', mapmeter_norm.value())
    with open('result_%s' % os.path.basename(args.checkpoint), 'w') as fp:
        json.dump(output, fp)

if __name__ == '__main__':
    main()
