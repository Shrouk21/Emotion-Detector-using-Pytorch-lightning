# Emotion Detection using FIER Dataset and PyTorch Lightning

## Overview
This project implements an **Emotion Detection** model trained on the **FIER dataset** using **PyTorch Lightning**. The model learns to classify emotions from facial images and achieves a test accuracy of approximately **52.3%**.

## Dataset
The **FIER dataset** (Facial Emotion Intensity Recognition) consists of labeled facial images categorized into different emotional states. The dataset is preprocessed and split into training, validation, and test sets for effective model training.

## Model Architecture
The model is a **Convolutional Neural Network (CNN)** implemented using **PyTorch Lightning**. It consists of:
- Three convolutional layers with ReLU activations and max pooling
- Fully connected layers for classification
- Dropout layers for regularization
- Cross-entropy loss for training

## Training Setup
- **Framework**: PyTorch Lightning
- **Optimizer**: Adam (Learning Rate: 0.001)
- **Loss Function**: Cross Entropy Loss
- **Batch Size**: 32
- **Epochs**: 20
- **Metrics**: Accuracy

## Installation
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install torch torchvision pytorch-lightning torchmetrics
```

## Training the Model
Run the following command to start training:
```bash
python train.py
```

## Evaluating the Model
To evaluate the model on the test set:
```bash
python test.py
```

## Results
- **Test Accuracy**: ~52.3%
- **Loss Trend**: Loss decreases over epochs, indicating successful learning.

## Directory Structure
```
|-- emotion_detection/
    |-- data/                # FIER dataset (images, labels)
    |-- logs/                # Training logs
    |-- models/              # Saved model checkpoints
    |-- train.py             # Training script
    |-- test.py              # Testing script
    |-- README.md            # Project documentation
```

## Future Improvements
- Fine-tuning the model with additional layers
- Experimenting with data augmentation
- Using pre-trained models like ResNet for better feature extraction

## Contributors
- Your Name

## License
This project is open-source and available under the MIT License.

