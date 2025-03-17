
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torch.optim as optim


class Emodetector(pl.LightningModule):
    def __init__(self, input_shape, num_classes):
        super(Emodetector, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        B, C, H, W = input_shape

        # Define layers
        self.conv1 = nn.Conv2d(C, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(self._get_fc_size(H, W, 3), 512)  # 3 pooling layers
        self.fc2 = nn.Linear(512, self.num_classes)
        self.dropout = nn.Dropout(0.25)

        # Accuracy metric (use `task="multiclass"` for classification)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  
        return x  

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat.argmax(dim=1), y)  # Convert logits to predicted classes
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat.argmax(dim=1), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss  # Ensure it's returned for proper logging

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat.argmax(dim=1), y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss  # Ensure it's returned

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

    def _get_fc_size(self, H, W, n_pooling_layers):
        """Calculates the flattened size after conv layers."""
        for _ in range(n_pooling_layers):
            H, W = H // 2, W // 2
        return H * W * 128  # 128 channels from last conv layer