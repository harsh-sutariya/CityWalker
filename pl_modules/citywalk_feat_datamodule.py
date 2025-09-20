# data/datamodule.py

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.citywalk_dataset import CityWalkSampler
from data.citywalk_feat_dataset import CityWalkFeatDataset
from torch.utils.data.distributed import DistributedSampler

class CityWalkFeatDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.data.num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CityWalkFeatDataset(self.cfg, mode='train')
            self.val_dataset = CityWalkFeatDataset(self.cfg, mode='val')

        if stage == 'test' or stage is None:
            self.test_dataset = CityWalkFeatDataset(self.cfg, mode='test')

    def train_dataloader(self):
        # Use DistributedSampler for multi-GPU training, fallback to custom sampler for single GPU
        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
        else:
            sampler = CityWalkSampler(self.train_dataset)
        
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, sampler=sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
