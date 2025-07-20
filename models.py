"""
Model definitions for the ODONTO.IA project.
"""
import torch
from torch import nn
from torchvision import models
from typing import Optional

def get_model(model_name: str, num_classes: int, fine_tune: bool = False) -> Optional[nn.Module]:
    """
    Loads a pretrained model and replaces the final layer for transfer learning.
    
    Args:
        model_name (str): The name of the model to load (e.g., 'ResNet18').
        num_classes (int): The number of output classes.
        fine_tune (bool): If True, all model parameters are unfrozen for fine-tuning.
    
    Returns:
        Optional[nn.Module]: The loaded model or None if the name is not supported.
    """
    model: Optional[nn.Module] = None
    weights = 'DEFAULT' # Use the latest recommended weights

    if model_name == 'ResNet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == 'ResNet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif model_name == 'DenseNet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    
    if model is None:
        return None

    # Freeze or unfreeze parameters based on the fine_tune flag
    for param in model.parameters():
        param.requires_grad = fine_tune

    # Replace the classifier layer
    if isinstance(model, models.ResNet):
        num_ftrs = model.fc.in_features
        # Unfreeze the new classifier layer
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif isinstance(model, models.DenseNet):
        num_ftrs = model.classifier.in_features
        # Unfreeze the new classifier layer
        model.classifier = nn.Linear(num_ftrs, num_classes)

    return model
