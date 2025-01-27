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
        self.num_classes = num_classes
        self.model = self.setup_model()  # Placeholder, will be defined in derived classes
        self.precision = torchmetrics.Precision(task="BINARY", num_classes=num_classes)
        self.recall = torchmetrics.Recall(task="BINARY", num_classes=num_classes)
        self.f1 = torchmetrics.F1Score(task="BINARY", num_classes=num_classes)
        self.train_acc = torchmetrics.Accuracy(task="BINARY", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="BINARY", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="BINARY", num_classes=num_classes)

    def setup_model(self):
        raise NotImplementedError("This method should be implemented in derived classes")

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc(preds, y), prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc(preds, y), prog_bar=True)
        self.log('val_precision', self.precision(preds, y), prog_bar=True)
        self.log('val_recall', self.recall(preds, y), prog_bar=True)
        self.log('val_f1', self.f1(preds, y), prog_bar=True)
        self.log('test number', 0, prog_bar=True)
        self.log('test number2', 0, prog_bar=False)
        return loss
    
    def test_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc(preds, y), prog_bar=True)
        self.log('test_precision', self.precision(preds, y), prog_bar=True)
        self.log('test_recall', self.recall(preds, y), prog_bar=True)
        self.log('test_f1', self.f1(preds, y), prog_bar=True)
        self.log('value test', 4, prog_bar=True)
        self.log('value test2', 5, prog_bar=False)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class ResNet50AudioClassifier(BaseAudioClassifier):
    def setup_model(self):
        model = models.resnet50(weights="IMAGENET1K_V2")
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        return model

class EfficientNetV2Classifier(BaseAudioClassifier):
    def setup_model(self):
        model = models.efficientnet_v2_m(weights="IMAGENET1K_V1")
        model.features[0][0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(1280, self.num_classes)
        return model

class DenseNet169Classifier(BaseAudioClassifier):
    def setup_model(self):
        model = models.densenet169(weights="IMAGENET1K_V1")
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.classifier.in_features  # This will be 1664 for DenseNet169
        model.classifier = nn.Linear(num_ftrs, self.num_classes)
        return model

class SwinTransformerV2(BaseAudioClassifier):
    def setup_model(self):
        model = models.swin_v2_b(weights="IMAGENET1K_V1")
        model.features[0][0] = nn.Conv2d(1, 768, kernel_size=16, stride=16, bias=False)
        model.head = nn.Linear(768, self.num_classes)
        return model