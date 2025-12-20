# trainer.py
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional
import streamlit as st

# Import optional optimizers
try:
    from torch_optimizer import Ranger
    RANGER_AVAILABLE = True
except ImportError:
    RANGER_AVAILABLE = False
    
try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False

import config

def apply_mixup(inputs: torch.Tensor, labels: torch.Tensor, alpha: float = 0.4):
    """Apply Mixup data augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(inputs.device)

    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    labels_a, labels_b = labels, labels[index]
    return mixed_inputs, (labels_a, labels_b, lam)

def apply_cutmix(inputs: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
    """Apply CutMix data augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(inputs.device)

    # Generate random bounding box
    _, _, h, w = inputs.shape
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    # Uniform sampling
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    # Apply cutmix
    mixed_inputs = inputs.clone()
    mixed_inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    
    labels_a, labels_b = labels, labels[index]
    return mixed_inputs, (labels_a, labels_b, lam)

def get_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, l2_lambda: float) -> torch.optim.Optimizer:
    """Cria um otimizador para o modelo."""
    if optimizer_name == 'Adam':
        return Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'AdamW':
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'SGD':
        return SGD(model.parameters(), lr=learning_rate, weight_decay=l2_lambda, momentum=0.9)
    elif optimizer_name == 'Ranger':
        if not RANGER_AVAILABLE:
            st.warning("Ranger optimizer not available. Install with: pip install torch-optimizer. Falling back to AdamW.")
            return AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
        return Ranger(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'Lion':
        if not LION_AVAILABLE:
            st.warning("Lion optimizer not available. Install with: pip install lion-pytorch. Falling back to AdamW.")
            return AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
        return Lion(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    else:
        raise ValueError(f"Otimizador '{optimizer_name}' não suportado.")

def get_scheduler(optimizer: torch.optim.Optimizer, scheduler_name: str, epochs: int, steps_per_epoch: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Cria um agendador de taxa de aprendizado (learning rate scheduler)."""
    if scheduler_name == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_name == 'OneCycleLR':
        # Get the learning rate from the optimizer
        learning_rate = optimizer.param_groups[0]['lr']
        return OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=steps_per_epoch, epochs=epochs)
    elif scheduler_name == 'Nenhum':
        return None
    else:
        raise ValueError(f"Agendador '{scheduler_name}' não suportado.")

def train_loop(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader,
               criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
               epochs: int, patience: int, augmentation: str = 'Nenhum', l1_lambda: float = 0.0):
    """Loop principal de treinamento do modelo."""
    device = config.DEVICE
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Show augmentation strategy being used
    if augmentation != 'Nenhum':
        st.info(f"Usando estratégia de aumento de dados: {augmentation}")

    for epoch in range(epochs):
        # Fase de treino
        model.train()
        train_loss, train_corrects = 0.0, 0
        train_pbar = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs} [Treino]", leave=False)
        for batch in train_pbar:
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            
            # Apply Mixup or Cutmix if selected
            if augmentation == 'Mixup':
                inputs, labels = apply_mixup(inputs, labels, alpha=config.AUGMENTATION_PARAMS['mixup_alpha'])
            elif augmentation == 'Cutmix':
                inputs, labels = apply_cutmix(inputs, labels, alpha=config.AUGMENTATION_PARAMS['cutmix_alpha'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Handle mixed labels for Mixup/Cutmix
            if augmentation in ['Mixup', 'Cutmix'] and isinstance(labels, tuple):
                labels_a, labels_b, lam = labels
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)

            # Add L1 regularization if specified
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_norm

            loss.backward()
            optimizer.step()
            if scheduler and isinstance(scheduler, OneCycleLR):
                scheduler.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            
            # Calculate accuracy correctly for mixed labels
            if augmentation in ['Mixup', 'Cutmix'] and isinstance(labels, tuple):
                labels_a, labels_b, lam = labels
                train_corrects += (lam * preds.eq(labels_a).sum().float() + 
                                 (1 - lam) * preds.eq(labels_b).sum().float())
            else:
                train_corrects += torch.sum(preds == labels.data)
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if scheduler and not isinstance(scheduler, OneCycleLR) and scheduler is not None:
            scheduler.step()

        # Fase de validação
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, labels = batch['image'].to(device), batch['label'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        # Guarda as métricas da época
        train_loss /= len(train_loader.dataset)
        train_acc = train_corrects.double() / len(train_loader.dataset)
        val_loss /= len(valid_loader.dataset)
        val_acc = val_corrects.double() / len(valid_loader.dataset)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())

        print(f"Época {epoch+1}/{epochs} | Perda Val: {val_loss:.4f} | Acerto Val: {val_acc:.2%}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Parada antecipada na época {epoch+1}")
            break

    return {"weights": best_model_wts, "history": history}

def compute_metrics(model: nn.Module, dataloader: DataLoader, classes: List[str]) -> Dict[str, Any]:
    """Calcula as métricas de classificação e retorna o relatório e a figura da matriz de confusão."""
    device = config.DEVICE
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True, zero_division=0)
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
    """Encontra imagens classificadas incorretamente e retorna uma figura para visualização."""
    device = config.DEVICE
    model.eval()
    error_images, true_labels, pred_labels = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            errors = preds != labels
            error_indices = errors.nonzero(as_tuple=True)[0]
            for idx in error_indices:
                error_images.append(inputs[idx].cpu())
                true_labels.append(classes[labels[idx].item()])
                pred_labels.append(classes[preds[idx].item()])

    if not error_images:
        st.info("Nenhum erro de classificação encontrado no conjunto de teste!")
        return

    num_cols = 5
    num_rows = min(4, (len(error_images) + num_cols - 1) // num_cols)  # Limit to 4 rows max
    num_display = min(len(error_images), num_rows * num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(num_display):
        img_tensor, true, pred = error_images[i], true_labels[i], pred_labels[i]
        ax = axes[i]
        img_display = img_tensor.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)
        ax.imshow(img_display)
        ax.set_title(f"Verdadeiro: {true}\nPredito: {pred}", fontsize=10)
        ax.axis('off')

    for j in range(num_display, len(axes)):
        axes[j].axis('off')

    st.pyplot(fig)
    plt.close(fig)
