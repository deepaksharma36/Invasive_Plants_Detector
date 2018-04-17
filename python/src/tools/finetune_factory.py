from src.tools.finetune_resnet_max import TrainResnetMax
from src.tools.finetune_resnet import TrainResnet
from src.tools.finetune_densnet import TrainDensnet
from src.tools.finetune_resnet_4_block_max import TrainResnet4BlockMax
from src.tools.finetune_resnet_3_block_max import TrainResnet3BlockMax

class TrainFactory:
    networks = ['resnet', 'densnet', 'resnet4BlockMax', 'resnetMax', 'resnet3BlockMax']

    @staticmethod
    def get_trainer(network_type, cfg, **kwargs):
        assert network_type in  TrainFactory.networks, "Bad network name"
        if network_type == 'resnet':
            return TrainResnet(cfg, **kwargs)
        elif network_type == 'densnet':
            return TrainDensnet(cfg, **kwargs)
        elif network_type == 'resnet4BlockMax':
            return TrainResnetMax(cfg, **kwargs)
        elif network_type == 'resnet3BlockMax':
            return TrainResnetMax(cfg, **kwargs)
        elif network_type == 'resnetMax':
            return TrainResnetMax(cfg, **kwargs)
