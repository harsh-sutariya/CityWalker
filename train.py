# main.py

import pytorch_lightning as pl
import argparse
import yaml
import os
from pl_modules.citywalk_datamodule import CityWalkDataModule
from pl_modules.urban_nav_module import UrbanNavModule
import torch
torch.set_float32_matmul_precision('medium')


# Remove the WandbLogger import from the top
# from pytorch_lightning.loggers import WandbLogger

class DictNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, DictNamespace(**value))
            else:
                setattr(self, key, value)

def parse_args():
    parser = argparse.ArgumentParser(description='Train UrbanNav model')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = DictNamespace(**cfg_dict)
    return cfg

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Create result directory
    result_dir = os.path.join(cfg.project.result_dir, cfg.project.run_name)
    os.makedirs(result_dir, exist_ok=True)
    cfg.project.result_dir = result_dir  # Update result_dir in cfg

    # Save config file in result directory
    with open(os.path.join(result_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg.__dict__, f)

    # Initialize the DataModule
    datamodule = CityWalkDataModule(cfg)

    # Initialize the model
    model = UrbanNavModule(cfg)

    # Initialize logger
    logger = None  # Default to no logger

    # Check if logging with Wandb is enabled in config
    use_wandb = cfg.logging.enable_wandb

    if use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger  # Import here to handle ImportError
            wandb_logger = WandbLogger(
                project=cfg.project.name,
                name=cfg.project.run_name,
                save_dir=result_dir
            )
            logger = wandb_logger
            print("WandbLogger initialized.")
        except ImportError:
            print("Wandb is not installed. Skipping Wandb logging.")

    # Optionally, you can add a default logger like TensorBoardLogger
    # If you want to have some logging even when Wandb is disabled, uncomment below:
    #
    # if logger is None:
    #     from pytorch_lightning.loggers import TensorBoardLogger
    #     tb_logger = TensorBoardLogger(save_dir=result_dir, name='tb_logs')
    #     logger = tb_logger
    #     print("TensorBoardLogger initialized.")
    
    # Set up Trainer
    trainer = pl.Trainer(
        default_root_dir=result_dir,
        max_epochs=cfg.training.max_epochs,
        logger=logger,  # Pass the logger (WandbLogger or None)
        devices=cfg.training.gpus,
        precision='16-mixed' if cfg.training.amp else 32,
        accelerator='ddp' if cfg.training.gpus > 1 else 'gpu',
        callbacks=[pl.callbacks.ModelCheckpoint(dirpath=os.path.join(result_dir, 'checkpoints'))],
        log_every_n_steps=1,
        num_sanity_val_steps=0
        # Other trainer arguments
    )

    # Start training
    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    main()
