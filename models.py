import torch
import torch.nn as nn
from torchvision import models
import streamlit as st

def get_model(model_name: str, num_classes: int, fine_tune: bool = True, dropout_rate: float = 0.5):
    """
    Carrega um modelo pré-treinado e o adapta para a tarefa de classificação.
    Versão atualizada com suporte a ResNet18 e insensível a maiúsculas/minúsculas.
    """
    try:
        # Converte o nome do modelo para minúsculas para comparação insensível a maiúsculas
        model_name_lower = model_name.lower()
        
        if model_name_lower == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_ftrs, num_classes)
            )
        elif model_name_lower == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_ftrs, num_classes)
            )
        elif model_name_lower == 'densenet121':
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_ftrs, num_classes)
            )
        elif model_name_lower == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(num_ftrs, num_classes),
            )
        else:
            st.error(f"Modelo '{model_name}' não suportado.")
            return None

        # Congela ou descongela os pesos com base no parâmetro fine_tune
        for param in model.parameters():
            param.requires_grad = fine_tune

        # Garante que os parâmetros da nova camada de classificação sejam sempre treináveis
        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True

        return model

    except Exception as e:
        st.error(f"Erro ao carregar o modelo '{model_name}': {e}")
        return None
