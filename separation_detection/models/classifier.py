import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import torchvision.models as models
from torchvision.models import vit_b_16
import torch.nn as nn
import torch.optim as optim


class BaseAudioClassifier(pl.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = self.setup_model()  # Placeholder, will be defined in derived classes
        self.num_classes = num_classes


    def setup_model(self):
        raise NotImplementedError("This method should be implemented in derived classes")

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class ResNet18AudioClassifier(BaseAudioClassifier):
    def setup_model(self):
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

class ResNet34AudioClassifier(BaseAudioClassifier):
    def setup_model(self):
        self.model = models.resnet34(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

class EfficientNetClassifier(BaseAudioClassifier):
    def setup_model(self):
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[1] = nn.Linear(1280, self.num_classes)

class MobileNetV3Classifier(BaseAudioClassifier):
    def setup_model(self):
        self.model = models.mobilenet_v3_small(pretrained=True)
        self.model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[3] = nn.Linear(1024, self.num_classes)

class DenseNetClassifier(BaseAudioClassifier):
    def setup_model(self):
        self.model = models.densenet121(pretrained=True)
        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.classifier = nn.Linear(1024, self.num_classes)

class ViTClassifier(BaseAudioClassifier):
    def setup_model(self):
        self.model = vit_b_16(pretrained=True)
        self.model.conv_proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
        self.model.heads.head = nn.Linear(768, self.num_classes)
