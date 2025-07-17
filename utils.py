"""
Utility functions for the ODONTO.IA project.
"""
import torch
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, cast
from torchvision.datasets import ImageFolder
from matplotlib.patches import Rectangle
from typing import cast
import torchvision
from torch.utils.data import DataLoader

def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int):
    """Seed for DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def visualize_data(dataset: ImageFolder, classes: List[str]):
    """Display sample images from the dataset."""
    st.write("Visualização de algumas imagens do conjunto de dados:")
    if len(dataset) < 5:
        st.warning("Não há imagens suficientes para visualização.")
        return
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        idx = np.random.randint(len(dataset))
        image, label = dataset[idx]
        ax = axes[i]
        ax.imshow(np.array(image))
        ax.set_title(classes[label])
        ax.axis('off')
    st.pyplot(fig)
    plt.close(fig) # Liberar memória

def plot_class_distribution(dataset: Any, classes: List[str]):
    """
    Plota a distribuição de classes de um dataset, lidando de forma robusta com
    wrappers como CustomDataset e torch.utils.data.Subset.
    """
    st.write("Distribuição das classes:")
    
    labels: List[int] = []
    
    # Lida com nosso CustomDataset, que tem um atributo 'subset'
    if hasattr(dataset, 'subset'):
        dataset = dataset.subset
    
    # Lida com torch.utils.data.Subset
    if isinstance(dataset, torch.utils.data.Subset):
        subset_instance = cast(torch.utils.data.Subset, dataset)
        indices = subset_instance.indices
        base_dataset = subset_instance.dataset
        
        if hasattr(base_dataset, 'targets'):
            all_targets = cast(List[int], getattr(base_dataset, 'targets'))
            labels = [all_targets[i] for i in indices]
        else:
            st.error("O dataset base dentro do Subset não possui o atributo 'targets'.")
            return
            
    # Lida com um dataset base (como ImageFolder) diretamente
    elif hasattr(dataset, 'targets'):
        labels = cast(List[int], getattr(dataset, 'targets'))
        
    else:
        st.error("Formato de dataset não suportado ou o dataset não possui o atributo 'targets'.")
        return

    if not labels:
        st.warning("Não foi possível extrair os rótulos para plotar a distribuição.")
        return

    class_counts = {cls: labels.count(i) for i, cls in enumerate(classes)}
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), ax=ax)
    ax.set_title('Distribuição de Classes no Conjunto de Dados')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Número de Imagens')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)

def visualize_augmented_data(dataloader: DataLoader, num_images: int = 16):
    """Mostra um grid de imagens do dataloader para visualizar o aumento de dados."""
    st.subheader("Visualização do Aumento de Dados (Imagens Sintéticas)")
    st.info("Abaixo estão exemplos de imagens do primeiro lote de treinamento, após a aplicação das transformações de aumento de dados.")
    
    try:
        # Pega um lote de dados
        images, _ = next(iter(dataloader))
        
        # Garante que não estamos tentando mostrar mais imagens do que temos
        num_images = min(num_images, len(images))
        
        # Cria um grid de imagens
        img_grid = torchvision.utils.make_grid(images[:num_images])
        
        # Desnormaliza as imagens para exibição
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_grid = img_grid * std + mean
        img_grid = torch.clamp(img_grid, 0, 1)
        
        # Converte para formato de imagem do Matplotlib e exibe
        np_img = img_grid.numpy()
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(np.transpose(np_img, (1, 2, 0)))
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Não foi possível gerar a visualização do aumento de dados: {e}")


def plot_metrics(history: Dict[str, List[float]]):
    """Plota as curvas de perda e acurácia."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plotando Perda
    ax1.plot(history['train_loss'], label='Perda de Treino')
    ax1.plot(history['val_loss'], label='Perda de Validação')
    ax1.set_title('Perda por Época')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Perda')
    ax1.legend()
    
    # Plotando Acurácia
    ax2.plot(history['train_acc'], label='Acurácia de Treino')
    ax2.plot(history['val_acc'], label='Acurácia de Validação')
    ax2.set_title('Acurácia por Época')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Acurácia')
    ax2.legend()
    
    st.pyplot(fig)
    plt.close(fig)
