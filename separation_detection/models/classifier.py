import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import torchvision.models as models
from torchvision.models import vit_b_16
import torch.nn as nn
import torch.optim as optim
import torchmetrics

class BaseAudioClassifier(pl.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = self.setup_model()  # Placeholder, will be defined in derived classes
        self.num_classes = num_classes
        self.precision = torchmetrics.Precision(num_classes=num_classes)
        self.recall = torchmetrics.Recall(num_classes=num_classes)
        self.f1 = torchmetrics.F1(num_classes=num_classes)
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def setup_model(self):
        raise NotImplementedError("This method should be implemented in derived classes")

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc(y_hat, y), prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc(y_hat, y), prog_bar=True)
        self.log('val_precision', self.precision(y_hat, y), prog_bar=True)
        self.log('val_recall', self.recall(y_hat, y), prog_bar=True)
        self.log('val_f1', self.f1(y_hat, y), prog_bar=True)
        return loss
    
    def test_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc(y_hat, y), prog_bar=True)
        self.log('test_precision', self.precision(y_hat, y), prog_bar=True)
        self.log('test_recall', self.recall(y_hat, y), prog_bar=True)
        self.log('test_f1', self.f1(y_hat, y), prog_bar=True)
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
