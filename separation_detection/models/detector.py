import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchaudio
import numpy as np

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DeepFakeDetector(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.ae_normal = AutoEncoder()
        self.ae_separated = AutoEncoder()
        self.classifier = nn.Linear(2, 1)

    def forward(self, x):
        recon_normal = self.ae_normal(x)
        recon_separated = self.ae_separated(x)
        
        error_normal = nn.functional.mse_loss(recon_normal, x, reduction='none').mean(dim=(1,2,3))
        error_separated = nn.functional.mse_loss(recon_separated, x, reduction='none').mean(dim=(1,2,3))
        
        features = torch.stack([error_normal, error_separated], dim=1)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y.float())
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



# 학습 실행
def train_model():
    # 데이터셋 준비 (예시)
    train_dataset = VocalDataset(['file1.wav', 'file2.wav'], [0, 1])
    val_dataset = VocalDataset(['file3.wav', 'file4.wav'], [1, 0])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    model = DeepFakeDetector()
    trainer = pl.Trainer(max_epochs=10, gpus=1 if torch.cuda.is_available() else 0)
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    train_model()