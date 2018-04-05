from __future__ import print_function, division
import torch
import sys
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
from src.models.resnet import resnet18
from src.tools.data_split import DataSplit

from src.models.resnet_surgery import resnet18, resnet9
from src.utils.cfg import yfile_to_cfg
class FVE_Resnet():
    def __init__(self, cfg, datasplit, **kwargs):
        self.data_dir = cfg.DATASET.DATA_DIR
        self.momentum = cfg.RESNET.MOMENTUM
        self.learning_rate = cfg.RESNET.LEARNING_RATE
        self.gamma = cfg.RESNET.GAMMA
        self.step_size = cfg.RESNET.STEP_SIZE
        self.batch_size = cfg.DATASET.BATCH_SIZE
        self.num_epochs = cfg.RESNET.NUM_EPOCHS
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.port = cfg.RESNET.PORT
        self.start_epoch = 0
        self.datasplit = datasplit
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
            nn.CrossEntropyLoss(weight=torch.FloatTensor(self.datasplit.class_weights).cuda())



    def load_or_set_model(self, checkpoint_path=None):
        self.model = resnet9(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.datasplit.num_classes)
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
                                 'arch': 'resnet18',
                                 'state_dict': self.model.state_dict(),
                                 'best_prec1': self.best_val_acc,
                                 'optimizer': self.optimizer.state_dict()}

        file_path = self.data_dir+'checkpoint.pth.tar'
        torch.save(model_checkpoint_dict, file_path)
        if self.phase == 'val' and self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_model_path = self.data_dir+'model_best.pth.tar'
            shutil.copyfile(file_path, self.data_dir+'model_best.pth.tar')

    def train_model(self):
        assert self.optimizer, "No optimizer defined"
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

    def test_model(self, best_model_path=None, datasplit=None):
        assert self.best_model_path or best_model_path, "No trained model available for testing"
        if not best_model_path and self.best_model_path:
            best_model_path = self.best_model_path
        if not datasplit:
            datasplit = self.datasplit

def main():

    assert len(sys.argv) > 1, "cfg file path missing"
    cfg = yfile_to_cfg(sys.argv[1])
    data_dir = './hymenoptera_data'
    datasplit = DataSplit(cfg, data_dir=data_dir)
    fve_resnet = FVE_Resnet(cfg, datasplit)
    fve_resnet.load_or_set_model()
    fve_resnet.set_loss_fun_optimizer_and_scheduler()
    fve_resnet.train_model()

if __name__ == '__main__':
    main()

