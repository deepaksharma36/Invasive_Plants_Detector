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


class Train():
    def __init__(self, datasplit=None, **kwargs):
        self.start_epoch = 0
        self.datasplit = datasplit
        #self.data_dir = datasplit.data_dir
        self.optimizer = None
        self.use_gpu = torch.cuda.is_available()
        self.best_model_path = None
        self.__dict__.update(**kwargs)

    def set_loss_fun_optimizer_and_scheduler(self):
        assert self.model, "Model is not intialized, optimizer can't be set"
        self.optimizer =\
            optim.SGD(self.model.parameters(), lr=self.learning_rate,
                      momentum=self.momentum)
        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                             step_size=self.step_size,
                                             gamma=self.gamma)
        self.criterion =\
            nn.CrossEntropyLoss(weight=\
                                torch.FloatTensor(self.datasplit.class_weights).cuda())

    def load_or_set_model(self, checkpoint_path=None):
        raise NotImplementedError

    def __init_meters__(self):
        self.train_loss_logger = VisdomPlotLogger('line', port=self.port,
                                            opts={'title': 'Train Loss'})
        self.train_err_logger = VisdomPlotLogger('line', port=self.port,
                                            opts={'title': 'Train Class Accuracy'})
        self.val_loss_logger = VisdomPlotLogger('line', port=self.port,
                                            opts={'title': 'Val Loss'})
        self.val_err_logger = VisdomPlotLogger('line', port=self.port,
                                            opts={'title': 'Val Class Accuracy'})
        self.meter_loss = tnt.meter.AverageValueMeter()
        self.meter_acc = tnt.meter.ClassErrorMeter(accuracy=True)
        self.best_val_acc = 0
        self.epoch_loss = 0
        self.epoch_acc = 0

    def save_checkpoint(self, epoch):
        model_checkpoint_dict = {'epoch': epoch + 1,
                                 'arch': self.arch,
                                 'state_dict': self.model.state_dict(),
                                 'best_prec1': self.best_val_acc,
                                 'optimizer': self.optimizer.state_dict()}

        checkpoint_dir = os.path.join(*[self.datasplit.data_dir, 'trained_models', self.network_type])
        if not  os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(*[checkpoint_dir,
                                         self.arch+'_checkpoint.pth.tar'])
        torch.save(model_checkpoint_dict, checkpoint_path)
        print("Saving Checkpoint '{}'".format(checkpoint_path))
        if self.phase == 'val' and self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_model_path =\
                os.path.join(*[checkpoint_dir, self.arch+'_model_best.pth.tar'])
            shutil.copyfile(checkpoint_path, self.best_model_path )

    def __train_model__(self):
        self.__init_meters__()
        since = time.time()
        for epoch in range(self.start_epoch+1, self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)
            # Each epoch has a training and validation self.phase
            for self.phase in ['train', 'val']:
                if self.phase == 'train':
                    self.scheduler.step()
                    self.model.train(True)  # Set self.model to training mode
                else:
                    self.model.train(False)  # Set self.model to evaluate mode
                for data in self.datasplit.dataloaders[self.phase]:
                    inputs, labels = data
                    if self.use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)
                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    self.meter_acc.add(outputs.data, labels.data)
                    self.meter_loss.add(loss.data[0])
                    if self.phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                self.update_and_reset_meters(epoch)
                self.save_checkpoint(epoch)
        time_elapsed = time.time() - since
        self.print_training_summery(time_elapsed)
        return self.best_model_path

    def print_training_summery(self, time_elapsed):
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(self.best_val_acc))
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            self.phase, self.epoch_loss, self.epoch_acc))

    def update_and_reset_meters(self, epoch):
        self.epoch_loss = self.meter_loss.value()[0]
        self.epoch_acc = self.meter_acc.value()[0]
        print(self.phase, "Loss:", self.epoch_loss)
        print(self.phase, "Acc:", self.epoch_acc)
        if self.phase == 'train':
            self.train_loss_logger.log(epoch, self.epoch_loss)
            self.train_err_logger.log(epoch, self.epoch_acc)
        elif self.phase == 'val':
            self.val_loss_logger.log(epoch, self.epoch_loss)
            self.val_err_logger.log(epoch, self.epoch_acc)
        self.meter_acc.reset()
        self.meter_loss.reset()

    def test_model(self, best_model_path=None, datasplit=None, split='test'):
        assert self.best_model_path or best_model_path, "No trained model available for testing"
        if not best_model_path and self.best_model_path:
            best_model_path = self.best_model_path
        if not datasplit:
            assert self.datasplit, "No datasplit provided"
            datasplit = self.datasplit

        self.load_or_set_model(checkpoint_path=best_model_path)
        self.model.train(False)  # Set model to evaluate mode
        confusion_matrix = ConfusionMeter(self.num_classes, normalized=True)
        consolidated_score = None
        consolidated_label = None
        for data in datasplit.dataloaders[split]:
            inputs, labels = data
            if self.use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, 1)
            if consolidated_score is None and consolidated_label is None:
                consolidated_score = copy.deepcopy(outputs.data)
                consolidated_label = copy.deepcopy(labels.data)
            else:
                consolidated_score =\
                    torch.cat((consolidated_score, outputs.data), 0)
                consolidated_label =\
                    torch.cat((consolidated_label, labels.data), 0)

            confusion_matrix.add(outputs.data, labels.data)

        print(confusion_matrix.conf)
        return consolidated_label, consolidated_score, confusion_matrix

    def finetune_model_fun(self, checkpoint_path=None):
        assert self.datasplit, "No datasplit assigned"
        print(self.datasplit.class_counts)
        print(self.datasplit.class_weights)
        self.load_or_set_model(checkpoint_path)
        self.set_loss_fun_optimizer_and_scheduler()
        return self.__train_model__()
