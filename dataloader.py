import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import numpy as np

import pytorch_lightning as pl
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Create a mapping: class name -> index
        self.class_to_idx = {class_name: label for label, class_name in enumerate(sorted(os.listdir(root_dir)))}
        
        # Load image paths and labels
        for class_name, label in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):  # Ensure it's a valid directory
                for file in os.listdir(class_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):  
                        self.data.append((os.path.join(class_dir, file), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")  # Load image
        
        if self.transform:
            image = self.transform(image)  # Apply transformations
        
        return image, torch.tensor(label, dtype=torch.long)

    def get_class_mapping(self):
        """Returns the class-to-index mapping."""
        return self.class_to_idx

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4, val_split=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split  # Percentage of data for validation
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize images
            transforms.ToTensor(),  # Convert to PyTorch tensors
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
        ])
    def size(self):
        # Get a sample image to determine the input shape
        class_mapping = self.train_dataset.dataset.get_class_mapping()
        sample_class = next(iter(class_mapping))
        sample_image_path = os.path.join(self.data_dir, "train", sample_class, os.listdir(os.path.join(self.data_dir, "train", sample_class))[0])
        sample_image = self.transform(Image.open(sample_image_path))
        return (self.batch_size, *sample_image.shape)
    def setup(self, stage=None):
        full_dataset = ImageFolderDataset(root_dir=f"{self.data_dir}/train", transform=self.transform)
        val_size = int(self.val_split * len(full_dataset))
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        self.test_dataset = ImageFolderDataset(root_dir=f"{self.data_dir}/test", transform=self.transform)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
