# -*- coding: utf-8 -*-
"""
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_
Deepak Sharma
"""
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchnet.logger import VisdomPlotLogger
import shutil
import time
import os
import copy
from torchnet.meter import ConfusionMeter
import torchnet as tnt
from src.models.densenet import densenet121
from src.tools.finetune import Train


class TrainDensnet(Train):
    def __init__(self, cfg, datasplit=None, **kwargs):
        self.momentum = cfg.DENSNET.MOMENTUM
        self.learning_rate = cfg.DENSNET.LEARNING_RATE
        self.gamma = cfg.DENSNET.GAMMA
        self.step_size = cfg.DENSNET.STEP_SIZE
        self.batch_size = cfg.DATASET.BATCH_SIZE
        self.num_epochs = cfg.DENSNET.NUM_EPOCHS
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.arch = cfg.DENSNET.ARCH
        self.port = cfg.DENSNET.PORT
        super().__init__(datasplit, **kwargs)

    def load_or_set_model(self, checkpoint_path=None):
        self.model = densenet121(pretrained=True)

        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, self.num_classes)
        #num_ftrs = self.model.fc.in_features
        #self.model.fc = nn.Linear(num_ftrs, self.datasplit.num_classes)
        if self.use_gpu:
            self.model = self.model.cuda()
        if checkpoint_path and os.path.isfile(checkpoint_path):
            print("Loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            self.start_epoch = checkpoint['epoch']
            self.best_val_acc = checkpoint['best_prec1']
            self.model.load_state_dict(checkpoint['state_dict'])
            if self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
        elif checkpoint_path and not os.path.isfile(checkpoint_path):
            print("Checkpoint not found at '{}'".format(checkpoint_path))
