import os
from torchvision import datasets, models, transforms
from collections import Counter
from torch.utils.data.dataset import random_split
import torch

class DataSplit():

    def __init__(self, cfg, **kwargs):
        self.data_dir = cfg.DATASET.DATA_DIR
        self.batch_size = cfg.DATASET.BATCH_SIZE
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.num_worker = cfg.DATASET.NUM_WORKER
        self.train_shuffle = cfg.DATASET.TRAIN_SHUFFLE
        self.train_val_ratio = cfg.DATASET.TRAIN_VAL_RATIO
        self.dataloaders = None
        self.dataset_sizes = None
        self.classes_name = None
        self.class_counts = None
        self.image_datasets = None
        self.img_size = cfg.DATASET.IMG_SIZE
        self.__dict__.update(**kwargs)
        self.__define_trasformations__(self.img_size)
        self.load_data()

    def __define_trasformations__(self, size=(224,224)):
        self.data_transforms = {
            'train': transforms.Compose([
                #transforms.Resize((13311//4, 2800//4)),
                transforms.Resize(size),
                #13311 x2800
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(size),
                #transforms.Resize((13311//4, 2800//4)),
                #transforms.Resize((13312//4, 6656//4)),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(size),
                #transforms.Resize((13311//4, 2800//4)),
                #transforms.Resize((13312//4, 6656//4)),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def __load_image_dataset__(self):
        self.image_datasets = {}
        self.image_datasets['train_val'] =\
            datasets.ImageFolder(os.path.join(self.data_dir, 'train'), self.data_transforms['train'])
        self.image_datasets['test'] =\
            datasets.ImageFolder(os.path.join(self.data_dir, 'test'), self.data_transforms['test'])


        self.classes_name = self.image_datasets['test'].classes

        self.class_counts =\
            {split: dict(Counter(sample_tup[1]\
                                 for sample_tup in\
                                 self.image_datasets[split].imgs)) for \
             split in ['train_val', 'test']}

    def __devide_train_data__(self):
        dataset_size = self.image_datasets['train_val'].__len__()
        train_split_size = int(dataset_size*self.train_val_ratio)
        val_split_size = dataset_size-int(dataset_size*self.train_val_ratio)
        self.image_datasets['train'], self.image_datasets['val'] =\
            random_split(self.image_datasets['train_val'],[train_split_size,
                                                       val_split_size])

    def __set_data_loader__(self):
        self.dataloaders = {}
        self.dataloaders['train_val'] =\
            torch.utils.data.DataLoader(self.image_datasets['train_val'],
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=self.num_worker)
        self.__devide_train_data__()
        for x in ['train', 'val' ]:
            self.dataloaders[x] =\
                torch.utils.data.DataLoader(self.image_datasets[x],
                                            batch_size=self.batch_size,
                                            shuffle=self.train_shuffle,
                                            num_workers=self.num_worker)
        self.dataloaders['test'] =\
            torch.utils.data.DataLoader(self.image_datasets['test'],
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=self.num_worker)

    def load_data(self):
        self.__load_image_dataset__()
        self.__set_data_loader__()
        self.dataset_sizes =\
            {x: len(self.image_datasets[x]) for x in ['train', 'val', 'test']}
        self.class_weights =\
            [1-(float(self.class_counts['train_val'][class_id])/(self.dataset_sizes['train']+self.dataset_sizes['val']))
             for class_id in range(self.num_classes)]
