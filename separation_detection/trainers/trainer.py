# /project_name/trainers/trainer.py

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

class AudioModelTrainer:
    def __init__(self, model, data_module, config):
        self.model = model
        self.data_module = data_module
        self.config = config

        # Create callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=config['checkpoint_dir'],
            filename='{epoch}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        )

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min'
        )

        # Create logger
        logger = TensorBoardLogger(config['log_dir'], name=config['experiment_name'])

        # Initialize the Trainer
        self.trainer = pl.Trainer(
            max_epochs=config['max_epochs'],
            gpus=config['gpus'],
            accelerator=config['accelerator'],
            logger=logger,
            callbacks=[checkpoint_callback, early_stop_callback],
            precision=config['precision'],
            gradient_clip_val=config['gradient_clip_val'],
            accumulate_grad_batches=config['accumulate_grad_batches'],
            val_check_interval=config['val_check_interval'],
            log_every_n_steps=config['log_every_n_steps'],
            # Add any other Trainer arguments you need
        )

    def train(self):
        self.trainer.fit(self.model, self.data_module)

    def test(self):
        self.trainer.test(self.model, self.data_module)

    def predict(self, dataloader):
        return self.trainer.predict(self.model, dataloaders=dataloader)