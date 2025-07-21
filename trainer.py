# trainer.py

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from tqdm.auto import tqdm
import copy
from typing import List, Dict, Any

import config # ODONTO.IA import

# Funções de Otimizador e Agendador (Scheduler)
def get_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, l2_lambda: float) -> torch.optim.Optimizer:
    """Cria um otimizador para o modelo."""
    if optimizer_name == 'Adam':
        return Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'AdamW':
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'SGD':
        return SGD(model.parameters(), lr=learning_rate, weight_decay=l2_lambda, momentum=0.9)
    # Adicione outros otimizadores aqui, como 'Ranger' ou 'Lion', se necessário
    # Ex: from pytorch_ranger import Ranger
    #     if optimizer_name == 'Ranger':
    #         return Ranger(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    else:
        raise ValueError(f"Otimizador '{optimizer_name}' não suportado.")

def get_scheduler(optimizer: torch.optim.Optimizer, scheduler_name: str, epochs: int, steps_per_epoch: int) -> torch.optim.lr_scheduler._LRScheduler:
    """Cria um agendador de taxa de aprendizado (learning rate scheduler)."""
    if scheduler_name == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_name == 'OneCycleLR':
        return OneCycleLR(optimizer, max_lr=config.TRAINING_PARAMS['learning_rate'], steps_per_epoch=steps_per_epoch, epochs=epochs)
    elif scheduler_name == 'Nenhum':
        return None
    else:
        raise ValueError(f"Agendador '{scheduler_name}' não suportado.")

# Lógica de Treino
def train_loop(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader,
               criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any,
               epochs: int, patience: int, augmentation: str, l1_lambda: float = 0.0):
    """Loop principal de treinamento do modelo."""
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    progress_bar = tqdm(range(epochs), desc="Épocas")

    for epoch in progress_bar:
        # Treino
        model.train()
        train_loss, train_corrects = 0.0, 0
        for batch in train_loader:
            inputs, labels = batch['image'].to(config.DEVICE), batch['label'].to(config.DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Regularização L1
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_norm
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data)

        if scheduler and not isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        # Validação
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, labels = batch['image'].to(config.DEVICE), batch['label'].to(config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        # Cálculo das métricas da época
        train_loss /= len(train_loader.dataset)
        train_acc = train_corrects.double() / len(train_loader.dataset)
        val_loss /= len(valid_loader.dataset)
        val_acc = val_corrects.double() / len(valid_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())

        progress_bar.set_postfix({
            'Train Loss': f'{train_loss:.4f}', 'Train Acc': f'{train_acc:.4f}',
            'Val Loss': f'{val_loss:.4f}', 'Val Acc': f'{val_acc:.4f}'
        })

        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            st.warning(f"Early stopping at epoch {epoch+1}")
            break
            
    return {"weights": best_model_wts, "history": history}

# Métricas e Análise de Erros
def compute_metrics(model: nn.Module, dataloader: DataLoader, classes: List[str]) -> Dict[str, Any]:
    """Calcula e exibe as métricas de classificação e a matriz de confusão."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch['image'].to(config.DEVICE), batch['label'].to(config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    
    # Matriz de Confusão
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    ax.set_title('Matriz de Confusão')
    st.pyplot(fig)
    plt.close(fig)
    
    return report

def error_analysis(model: nn.Module, dataloader: DataLoader, classes: List[str]):
    """Exibe as imagens que foram classificadas incorretamente."""
    model.eval()
    error_images, true_labels, pred_labels = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch['image'].to(config.DEVICE), batch['label'].to(config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            errors = preds != labels
            if torch.any(errors):
                error_indices = errors.nonzero(as_tuple=True)[0]
                for idx in error_indices:
                    error_images.append(inputs[idx].cpu())
                    true_labels.append(classes[labels[idx].item()])
                    pred_labels.append(classes[preds[idx].item()])

    if not error_images:
        st.success("Nenhum erro de classificação encontrado no conjunto de teste!")
        return
        
    st.write(f"Total de {len(error_images)} erros de classificação:")
    
    num_cols = 5
    num_rows = (len(error_images) + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    axes = axes.flatten()

    for i, (img_tensor, true, pred) in enumerate(zip(error_images, true_labels, pred_labels)):
        ax = axes[i]
        
        # Desnormalização para visualização
        img_display = img_tensor.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)
        
        ax.imshow(img_display)
        ax.set_title(f"Real: {true}\nPred: {pred}", fontsize=10)
        ax.axis('off')

    # Ocultar eixos não utilizados
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    st.pyplot(fig)
    plt.close(fig)
