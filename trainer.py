"""
Lógica principal de treinamento e avaliação para o projeto ODONTO.IA.
Versão corrigida para incluir regularização L1 e com melhorias na interface.
"""
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler
from monai.data.dataloader import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import streamlit as st
from typing import List, Dict, Any, cast, Tuple, Sized, Optional

# Importar otimizadores avançados
from lion_pytorch import Lion
from pytorch_ranger import Ranger

import config

# --- Funções de Aumento de Dados (Mixup/Cutmix) ---

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Retorna inputs misturados, pares de alvos e lambda."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=config.DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion: nn.Module, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Calcula a perda para o Mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    """Gera uma bounding box aleatória para o Cutmix."""
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Retorna inputs com Cutmix, pares de alvos e lambda ajustado."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=config.DEVICE)
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

# --- Funções de Configuração ---

def get_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, l2_lambda: float) -> optim.Optimizer:
    """Função de fábrica para otimizadores."""
    params = model.parameters()
    if optimizer_name == 'Adam':
        return optim.Adam(params, lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(params, lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'SGD':
        return optim.SGD(params, lr=learning_rate, weight_decay=l2_lambda, momentum=0.9)
    elif optimizer_name == 'Ranger':
        return Ranger(params, lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'Lion':
        return Lion(params, lr=learning_rate, weight_decay=l2_lambda)
    else:
        st.warning(f"Otimizador '{optimizer_name}' não reconhecido. Usando Adam como padrão.")
        return optim.Adam(params, lr=learning_rate, weight_decay=l2_lambda)

def get_scheduler(optimizer: optim.Optimizer, scheduler_name: str, epochs: int, steps_per_epoch: int) -> Optional[_LRScheduler]:
    """Função de fábrica para agendadores de taxa de aprendizado."""
    if scheduler_name == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'OneCycleLR':
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.TRAINING_PARAMS['learning_rate'], steps_per_epoch=steps_per_epoch, epochs=epochs)
    elif scheduler_name == 'Nenhum':
        return None
    else:
        st.warning(f"Agendador '{scheduler_name}' não reconhecido. Nenhum agendador será usado.")
        return None

# --- Loop Principal de Treinamento ---

def train_loop(
    model: nn.Module, 
    train_loader: DataLoader, 
    valid_loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    scheduler: Optional[_LRScheduler], 
    epochs: int, 
    patience: int, 
    augmentation_technique: str,
    l1_lambda: float = 0.0  # **CORREÇÃO: Parâmetro adicionado**
) -> Dict[str, Any]:
    """O loop de treinamento principal, agora com regularização L1."""
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    train_dataset_size = len(cast(Sized, train_loader.dataset))
    valid_dataset_size = len(cast(Sized, valid_loader.dataset))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        corrects_train = 0
        
        for batch_data in train_loader:
            inputs, labels = batch_data["image"].to(config.DEVICE), batch_data["label"].to(config.DEVICE)
            optimizer.zero_grad()

            if augmentation_technique in ['Mixup', 'Cutmix'] and model.training:
                if augmentation_technique == 'Mixup':
                    aug_inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, config.AUGMENTATION_PARAMS['mixup_alpha'])
                else: # Cutmix
                    aug_inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, config.AUGMENTATION_PARAMS['cutmix_alpha'])
                outputs = model(aug_inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                with torch.no_grad():
                    _, preds = torch.max(model(inputs), 1)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            # **CORREÇÃO: Adicionar a penalidade L1 à perda**
            if l1_lambda > 0.0:
                l1_penalty = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
                loss += l1_lambda * l1_penalty

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            corrects_train += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / train_dataset_size
        epoch_acc_train = corrects_train / train_dataset_size
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc_train)
        
        # Fase de Validação
        model.eval()
        val_loss = 0.0
        corrects_val = 0
        with torch.no_grad():
            for batch_data in valid_loader:
                inputs, labels = batch_data["image"].to(config.DEVICE), batch_data["label"].to(config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects_val += torch.sum(preds == labels.data).item()
        
        val_loss /= valid_dataset_size
        val_acc = corrects_val / valid_dataset_size
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        status_text.text(f"Época {epoch+1}/{epochs} | Perda Treino: {epoch_loss:.4f} | Acc Treino: {epoch_acc_train:.4f} | Perda Val: {val_loss:.4f} | Acc Val: {val_acc:.4f}")
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            st.warning(f"Parada antecipada na época {epoch+1}.")
            break
            
        progress_bar.progress((epoch + 1) / epochs)

    return {"weights": best_model_wts, "history": history}

# --- Funções de Avaliação ---

def compute_metrics(model: nn.Module, dataloader: DataLoader, classes: List[str]) -> Dict[str, Any]:
    """Calcula e exibe as métricas de classificação."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_data in dataloader:
            inputs, labels = batch_data["image"].to(config.DEVICE), batch_data["label"].to(config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report)
    
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predito'); ax.set_ylabel('Verdadeiro')
    st.pyplot(fig)
    
    # Adiciona acurácia geral ao dicionário retornado
    report['accuracy'] = accuracy_score(all_labels, all_preds)
    return cast(Dict[str, Any], report)

def error_analysis(model: nn.Module, dataloader: DataLoader, classes: List[str]):
    """Realiza e exibe a análise de erros."""
    model.eval()
    misclassified = []
    with torch.no_grad():
        for batch_data in dataloader:
            inputs, labels = batch_data["image"].to(config.DEVICE), batch_data["label"].to(config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in range(len(preds)):
                if preds[i] != labels[i]:
                    img_tensor = inputs[i].cpu().permute(1, 2, 0).numpy()
                    img = np.clip(img_tensor, 0, 1)
                    misclassified.append({
                        "image": img,
                        "true_label": classes[labels[i]],
                        "predicted_label": classes[preds[i]]
                    })
    
    st.write(f"Total de {len(misclassified)} imagens classificadas incorretamente.")
    if misclassified:
        # **CORREÇÃO: Exibir erros em uma grade**
        num_to_show = min(len(misclassified), 10)
        num_cols = 5
        cols = st.columns(num_cols)
        for i, item in enumerate(misclassified[:num_to_show]):
            with cols[i % num_cols]:
                st.image(item['image'], width=120)
                st.caption(f"Verdadeiro: {item['true_label']}\nPredito: {item['predicted_label']}")
    else:
        st.success("Nenhuma imagem foi classificada incorretamente!")
