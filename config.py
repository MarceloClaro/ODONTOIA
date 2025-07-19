"""
Configuration settings for the ODONTO.IA project.
"""
import torch
from torchvision import transforms

# -------------------
# General Settings
# -------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "dataset"
TRAIN_DIR = "dataset/Training"
VALID_DIR = "dataset/Validation"
TEST_DIR = "dataset/Testing"

# -------------------
# Model Configurations
# -------------------
AVAILABLE_MODELS = ['ResNet18', 'ResNet50', 'DenseNet121']

# -------------------
# Training Hyperparameters
# -------------------
TRAINING_PARAMS = {
    'image_size': 224,
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'train_split': 0.7,
    'patience': 10,
    'l2_lambda': 0.01,
}

# -------------------
# Data Augmentation
# -------------------
# Basic transformations for training when no augmentation is selected
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Transformations with data augmentation
train_transforms_augmented = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, shear=10, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Advanced augmentations (applied in the training loop)
AUGMENTATION_TECHNIQUES = ['Nenhum', 'Padrão', 'Mixup', 'Cutmix']
AUGMENTATION_PARAMS = {
    'mixup_alpha': 0.4,
    'cutmix_alpha': 1.0,
}

# -------------------
# Optimizers and Schedulers
# -------------------
AVAILABLE_OPTIMIZERS = ['Adam', 'AdamW', 'SGD', 'Ranger', 'Lion']
AVAILABLE_SCHEDULERS = ['Nenhum', 'CosineAnnealingLR', 'OneCycleLR']

# -------------------
# Regularization and Loss
# -------------------
REGULARIZATION_PARAMS = {
    'use_weighted_loss': True,
}

# -------------------
# XAI (Explainable AI)
# -------------------
AVAILABLE_XAI_METHODS = ['SmoothGradCAMpp', 'ScoreCAM', 'LayerCAM']
