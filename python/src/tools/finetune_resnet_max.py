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
import os
from src.models.resnet import resnet18
from src.tools.finetune import Train


class TrainResentMax(Train):
    def __init__(self, cfg, datasplit=None, **kwargs):
        self.data_dir = cfg.DATASET.DATA_DIR
        self.momentum = cfg.RESNET_MAX.MOMENTUM
        self.learning_rate = cfg.RESNET_MAX.LEARNING_RATE
        self.gamma = cfg.RESNET_MAX.GAMMA
        self.step_size = cfg.RESNET_MAX.STEP_SIZE
        self.batch_size = cfg.DATASET.BATCH_SIZE
        self.num_epochs = cfg.RESNET_MAX.NUM_EPOCHS
        self.arch  = cfg.RESNET_MAX.ARCH
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.port = cfg.RESNET_MAX.PORT
        super().__init__(datasplit, **kwargs)
        '''
        self.start_epoch = 0
        self.datasplit = datasplit
        self.optimizer = None
        self.use_gpu = torch.cuda.is_available()
        self.best_model_path = None
        self.__dict__.update(**kwargs)
        '''


    def load_or_set_model(self, checkpoint_path=None):
        self.model = resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.datasplit.num_classes)
        self.model.avgpool = nn.AdaptiveMaxPool2d((1,1))
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

