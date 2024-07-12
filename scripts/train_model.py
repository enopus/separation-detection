from separation_detection.data import AudioDataModule
from separation_detection.models import ResNet50AudioClassifier, EfficientNetV2Classifier, DenseNet169Classifier, SwinTransformerV2
from separation_detection.trainers import AudioModelTrainer


def main():
    # Configuration dictionary
    config = {
        'data_dir': '/path/to/data',
        'batch_size': 32,
        'num_workers': 4,
        'max_epochs': 10,
        'gpus': 1,
        'accelerator': 'gpu',  # Use 'ddp' for multi-GPU training
        'precision': 16,  # Mixed precision training
        'gradient_clip_val': 0.5,
        'accumulate_grad_batches': 1,
        'val_check_interval': 1.0,
        'log_every_n_steps': 50,
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'experiment_name': 'audio_classification'
    }

    # Create the data module
    data_module = AudioDataModule(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Initialize the model
    models = [
        ResNet50AudioClassifier(num_classes=2),
        EfficientNetV2Classifier(num_classes=2),
        DenseNet169Classifier(num_classes=2),
        SwinTransformerV2(num_classes=2),
    ]

    for model in models:
        config['experiment_name'] = model.__class__.__name__
        
        # Initialize the trainer
        trainer = AudioModelTrainer(model, data_module, config)

        # Train the model
        trainer.train()

        # Test the model
        trainer.test()

if __name__ == "__main__":
    main()