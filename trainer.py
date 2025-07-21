"""
Main training loop and evaluation logic for the ODONTO.IA project.
"""
import torch
from torch import nn, optim
from monai.data.dataloader import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
import streamlit as st
from typing import List, Dict, Any, cast, Tuple, Sized

# Import advanced optimizers
from lion_pytorch import Lion
from pytorch_ranger import Ranger

import config

# --- Data Augmentation Functions ---

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(config.DEVICE)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion: nn.Module, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float):
    # Convert lam to a tensor on the correct device to avoid dtype errors
    lam_t = torch.tensor(lam, device=pred.device, dtype=pred.dtype)
    return lam_t * criterion(pred, y_a) + (1 - lam_t) * criterion(pred, y_b)

def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(config.DEVICE)
    
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y_a, y_b, lam

def rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, l2_lambda: float) -> optim.Optimizer:
    """Factory function for optimizers."""
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_lambda, momentum=0.9)
    elif optimizer_name == 'Ranger':
        return Ranger(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'Lion':
        return Lion(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    else:
        st.warning(f"Otimizador '{optimizer_name}' não reconhecido. Usando Adam como padrão.")
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

def get_scheduler(optimizer: optim.Optimizer, scheduler_name: str, epochs: int, steps_per_epoch: int):
    """Factory function for learning rate schedulers."""
    if scheduler_name == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'OneCycleLR':
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.TRAINING_PARAMS['learning_rate'], 
                                             steps_per_epoch=steps_per_epoch, epochs=epochs)
    elif scheduler_name == 'Nenhum':
        return None
    else:
        st.warning(f"Agendador '{scheduler_name}' não reconhecido. Nenhum agendador será usado.")
        return None

def train_loop(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, 
               criterion: nn.Module, optimizer: optim.Optimizer, scheduler, 
               epochs: int, patience: int, augmentation_technique: str,
               l1_lambda: float = 0.0) -> Dict[str, Any]:  # Adicionado parâmetro l1_lambda
    """The main training loop."""
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    # The MONAI Dataset is wrapped, so we can get its length.
    train_dataset_size = len(cast(Sized, train_loader.dataset))
    valid_dataset_size = len(cast(Sized, valid_loader.dataset))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        corrects_train = 0
        
        loop = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs}", leave=False)
        for batch_data in loop:
            inputs, labels = batch_data["image"].to(config.DEVICE), batch_data["label"].to(config.DEVICE)
            
            optimizer.zero_grad()

            # Augmentation logic remains the same, as it operates on tensors
            if augmentation_technique in ['Mixup', 'Cutmix'] and model.training:
                if augmentation_technique == 'Mixup':
                    aug_inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, config.AUGMENTATION_PARAMS['mixup_alpha'])
                else: # Cutmix
                    aug_inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, config.AUGMENTATION_PARAMS['cutmix_alpha'])
                
                outputs = model(aug_inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                with torch.no_grad():
                    plain_outputs = model(inputs)
                    _, preds = torch.max(plain_outputs, 1)
            else: # Padrão
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            # Aplicar regularização L1 (LASSO) se l1_lambda > 0
            if l1_lambda > 0:
                l1_penalty = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
                loss += l1_lambda * l1_penalty

            # Certifique-se de que a retropropagação ocorra antes de qualquer conversão
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            corrects_train += torch.sum(preds == labels.data)
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / train_dataset_size
        epoch_acc_train = float(corrects_train) / train_dataset_size
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc_train)
        
        # Validation phase
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
                corrects_val += torch.sum(preds == labels.data)
        
        val_loss /= valid_dataset_size
        val_acc = float(corrects_val) / valid_dataset_size
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        status_text.text(f"Época {epoch+1}/{epochs} | Perda Treino: {epoch_loss:.4f} | Acurácia Treino: {epoch_acc_train:.4f} | Perda Val: {val_loss:.4f} | Acurácia Val: {val_acc:.4f}")
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss) # Monitorar a perda de validação
            else:
                scheduler.step()

        # Salvar o melhor modelo com base na menor perda de validação
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            st.toast(f"🎉 Novo melhor modelo encontrado na época {epoch+1} com perda de validação: {val_loss:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            st.warning(f"Parada antecipada na época {epoch+1} por falta de melhora na perda de validação.")
            break
            
        progress_bar.progress((epoch + 1) / epochs)

    return {"weights": best_model_wts, "history": history}

def compute_metrics(model: nn.Module, dataloader: DataLoader, classes: List[str]) -> Dict[str, Any]:
    """Computes and displays classification metrics, and returns the report dictionary."""
    model.eval()
    all_preds = []
    all_labels = []
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
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    st.pyplot(fig)
    
    return cast(Dict[str, Any], report)

def error_analysis(model: nn.Module, dataloader: DataLoader, classes: List[str]):
    """Performs and displays an error analysis."""
    model.eval()
    misclassified = []
    with torch.no_grad():
        for batch_data in dataloader:
            inputs, labels = batch_data["image"].to(config.DEVICE), batch_data["label"].to(config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in range(len(preds)):
                if preds[i] != labels[i]:
                    # The input from MONAI is already scaled, so we just need to handle channel format
                    img_tensor = inputs[i].cpu()
                    # MONAI LoadImaged provides channel-first, so permute to channel-last for display
                    img = img_tensor.permute(1, 2, 0).numpy()
                    # The ScaleIntensityd normalizes to [0,1], so no need for standard denormalization
                    img = np.clip(img, 0, 1)
                    
                    misclassified.append({
                        "image": img,
                        "true_label": classes[labels[i]],
                        "predicted_label": classes[preds[i]]
                    })
    
    if misclassified:
        st.write(f"Total de {len(misclassified)} imagens classificadas incorretamente.")
        
        # Display a few examples
        num_to_show = min(len(misclassified), 10)
        cols = st.columns(num_to_show)
        for i in range(num_to_show):
            with cols[i]:
                st.image(misclassified[i]['image'], width=100)
                st.caption(f"Verdadeiro: {misclassified[i]['true_label']}\nPredito: {misclassified[i]['predicted_label']}")
    else:
        st.success("Nenhuma imagem foi classificada incorretamente!")
