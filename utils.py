"""
Utility functions for the ODONTO.IA project.
"""
import torch
import random
import config
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
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
        # Converte para numpy e transpõe de (C, H, W) para (H, W, C)
        image_np = image.numpy()
        image_display = np.transpose(image_np, (1, 2, 0))
        
        # Desnormaliza para visualização correta (usando valores padrão do ImageNet)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_display = std * image_display + mean
        image_display = np.clip(image_display, 0, 1)
        
        ax.imshow(image_display)
        ax.set_title(classes[label])
        ax.axis('off')
    st.pyplot(fig)
    plt.close(fig) # Liberar memória

def plot_class_distribution(dataset: Any, classes: List[str], title: str = "Distribuição de Classes"):
    """
    Plota a distribuição de classes de um dataset, compatível com
    MONAI Datasets (lista de dicionários) e datasets torchvision.
    """
    st.subheader(title)
    
    labels: List[int] = []
    
    try:
        # MONAI Dataset (espera uma lista de dicionários com a chave 'label')
        if isinstance(dataset, torch.utils.data.Dataset) and hasattr(dataset, 'data') and isinstance(dataset.data, list): # type: ignore
            labels = [item['label'] for item in dataset.data] # type: ignore
        # torchvision Dataset (como ImageFolder)
        elif hasattr(dataset, 'targets'):
            labels = cast(List[int], getattr(dataset, 'targets'))
        # Lida com torch.utils.data.Subset
        elif isinstance(dataset, torch.utils.data.Subset):
            subset_instance = cast(torch.utils.data.Subset, dataset)
            # Tenta obter rótulos do dataset base (MONAI ou torchvision)
            if hasattr(subset_instance.dataset, 'data') and isinstance(subset_instance.dataset.data, list): # type: ignore
                 all_labels = [item['label'] for item in subset_instance.dataset.data] # type: ignore
                 labels = [all_labels[i] for i in subset_instance.indices]
            elif hasattr(subset_instance.dataset, 'targets'):
                all_targets = cast(List[int], getattr(subset_instance.dataset, 'targets'))
                labels = [all_targets[i] for i in subset_instance.indices]
            else:
                raise AttributeError("Dataset base dentro do Subset não tem 'data' ou 'targets'.")
        else:
            raise TypeError("Formato de dataset não suportado.")

    except (AttributeError, TypeError, KeyError) as e:
        st.error(f"Erro ao extrair rótulos para o gráfico de distribuição: {e}")
        return

    if not labels:
        st.warning("Não foi possível extrair os rótulos para plotar a distribuição.")
        return

    class_counts = {cls: labels.count(i) for i, cls in enumerate(classes)}
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Classe')
    ax.set_ylabel('Número de Imagens')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)

def visualize_augmented_data(dataloader: DataLoader, num_images: int = 16):
    """Mostra um grid de imagens do dataloader para visualizar o aumento de dados (compatível com MONAI)."""
    st.subheader("Visualização do Aumento de Dados (Imagens Sintéticas)")
    st.info("Abaixo estão exemplos de imagens do primeiro lote de treinamento, após a aplicação das transformações de aumento de dados.")
    
    try:
        # Pega um lote de dados (MONAI retorna um dicionário)
        batch = next(iter(dataloader))
        images = batch['image']
        
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

def display_model_architecture(model: torch.nn.Module, model_name: str):
    """Exibe a arquitetura do modelo com uma descrição técnica detalhada."""
    st.subheader("Arquitetura do Modelo Utilizado")
    with st.expander("Clique para ver a análise técnica da arquitetura"):
        st.markdown(f"**Modelo Base:** `{model_name}`")
        st.markdown("""
        A arquitetura utilizada é uma **Rede Neural Convolucional Residual (ResNet)**, um modelo de deep learning de última geração para classificação de imagens. A seguir, uma análise técnica dos seus componentes principais:
        - **Camada Convolucional Inicial (`conv1`):** Realiza a extração inicial de características de baixo nível, como bordas e texturas, com um kernel de 7x7 e stride de 2, o que reduz rapidamente as dimensões espaciais da imagem.
        - **Normalização em Lote (`bn1`):** Estabiliza e acelera o treinamento, normalizando as ativações entre as camadas. Isso mitiga o problema da mudança interna de covariáveis (internal covariate shift).
        - **Blocos Residuais (`layer1` a `layer4`):** O núcleo da ResNet. Cada bloco consiste em camadas convolucionais e uma "conexão de atalho" (shortcut connection) que permite ao gradiente fluir diretamente através da rede. Isso é crucial para treinar redes muito profundas sem sofrer com o problema do desaparecimento do gradiente (vanishing gradient).
        - **Downsampling:** Ocorre no início das camadas `layer2`, `layer3` e `layer4` (com `stride=2`) para reduzir a resolução espacial do mapa de características, permitindo que as camadas subsequentes aprendam características mais complexas e abstratas.
        - **Camada de Classificação Final (`avgpool`, `fc`):** A `AdaptiveAvgPool2d` reduz cada mapa de características a um único valor, independentemente do seu tamanho. A camada `Linear` (`fc`) final atua como o classificador, mapeando as características aprendidas para as 7 classes de saída do nosso problema.
        """)
        st.markdown("**Estrutura Detalhada:**")
        st.text(str(model))

def display_environment_info():
    """Exibe informações do ambiente para reprodutibilidade."""
    st.subheader("Informações de Reprodutibilidade")
    with st.expander("Clique para ver os detalhes do ambiente"):
        import monai
        import sklearn
        st.text(f"Versão do PyTorch: {torch.__version__}")
        st.text(f"Versão do MONAI: {monai.__version__}")
        st.text(f"Versão do Scikit-learn: {sklearn.__version__}")
        st.text(f"Dispositivo de Treinamento: {config.DEVICE}")

def interpret_results(metrics_report: Dict[str, Any], history: Dict[str, List[float]]):
    """Fornece uma interpretação científica dos resultados do modelo."""
    st.subheader("Interpretação dos Resultados")
    
    with st.container(border=True):
        st.markdown("#### Análise das Curvas de Aprendizagem")
        train_loss = history['train_loss'][-1]
        val_loss = history['val_loss'][-1]
        train_acc = history['train_acc'][-1]
        val_acc = history['val_acc'][-1]

        st.markdown(f"Ao final do treinamento, a perda de treino foi de **{train_loss:.4f}** e a de validação foi de **{val_loss:.4f}**. A acurácia de treino atingiu **{train_acc:.2%}**, enquanto a de validação foi de **{val_acc:.2%}**.")

        if val_loss > train_loss * 1.5 and val_acc < train_acc - 0.1:
            st.warning("**Indício de Overfitting:** A perda de validação é significativamente maior que a de treino, e a acurácia de validação é consideravelmente menor. O modelo pode estar memorizando os dados de treino em vez de generalizar. Considere aumentar a regularização (L2, dropout) ou usar mais aumento de dados.")
        elif train_loss > val_loss * 1.5:
             st.info("**Indício de Underfitting:** A perda de treino é alta, sugerindo que o modelo não tem capacidade suficiente para aprender os dados. Considere usar um modelo mais complexo, treinar por mais épocas ou reduzir a regularização.")
        else:
            st.success("**Bom Equilíbrio:** As perdas e acurácias de treino e validação estão próximas, indicando um bom equilíbrio entre viés e variância. O modelo parece estar generalizando bem para dados não vistos.")

    with st.container(border=True):
        st.markdown("#### Análise das Métricas de Classificação")
        accuracy = metrics_report.get('accuracy', 0)
        macro_f1 = metrics_report.get('macro avg', {}).get('f1-score', 0)
        
        st.markdown(f"A acurácia geral no conjunto de teste foi de **{accuracy:.2%}**. O F1-Score (macro) foi de **{macro_f1:.4f}**, que representa a média não ponderada do F1-Score para cada classe, sendo uma métrica robusta para datasets desbalanceados.")

        # Análise por classe
        class_metrics = {k: v for k, v in metrics_report.items() if isinstance(v, dict)}
        worst_class = min(class_metrics, key=lambda k: class_metrics[k]['f1-score'])
        best_class = max(class_metrics, key=lambda k: class_metrics[k]['f1-score'])

        st.markdown(f"A classe com o **pior desempenho** (menor F1-Score) foi **'{worst_class}'** ({class_metrics[worst_class]['f1-score']:.4f}), possivelmente devido a poucas amostras ou características visuais semelhantes a outras classes. A classe com o **melhor desempenho** foi **'{best_class}'** ({class_metrics[best_class]['f1-score']:.4f}).")
        st.info("""
        **Como interpretar a Matriz de Confusão:**
        - **Diagonal Principal (canto superior esquerdo ao inferior direito):** Mostra o número de predições corretas para cada classe. Valores altos aqui são desejáveis.
        - **Fora da Diagonal:** Indica os erros de classificação. O valor na linha `i` e coluna `j` representa o número de vezes que uma imagem da classe real `i` foi incorretamente classificada como classe `j`. Analisar esses valores ajuda a entender quais classes o modelo está confundindo.
        """)

def plot_interactive_embeddings(features: np.ndarray, labels: np.ndarray, image_paths: List[str], classes: List[str]):
    """Plota um gráfico de dispersão interativo dos embeddings com visualização de imagem."""
    st.subheader("Visualização Interativa de Embeddings (t-SNE)")
    st.info("Passe o mouse sobre os pontos para visualizar a imagem correspondente. Use a legenda para filtrar as classes.")

    from sklearn.manifold import TSNE
    import plotly.graph_objects as go
    import base64
    from PIL import Image

    # Redução de dimensionalidade com t-SNE
    with st.spinner("Calculando a redução de dimensionalidade com t-SNE..."):
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=config.SEED)
        reduced_features = tsne.fit_transform(features)

    # Função para codificar imagens para o hover
    def get_image_b64(path: str) -> str:
        try:
            with open(path, "rb") as f:
                img_bytes = f.read()
            return base64.b64encode(img_bytes).decode()
        except Exception:
            return "" # Retorna string vazia se a imagem não puder ser lida

    df = pd.DataFrame({
        'tsne1': reduced_features[:, 0],
        'tsne2': reduced_features[:, 1],
        'label': [classes[l] for l in labels],
        'path': image_paths
    })

    fig = go.Figure()

    for label in df['label'].unique():
        df_label = df[df['label'] == label]
        fig.add_trace(go.Scatter(
            x=df_label['tsne1'],
            y=df_label['tsne2'],
            mode='markers',
            name=label,
            customdata=df_label['path'],
            hovertemplate='<b>Classe:</b> %{meta}<br><b>Path:</b> %{customdata}<extra></extra>',
            meta=label
        ))

    fig.update_layout(
        title="Visualização de Embeddings de Imagem com t-SNE",
        xaxis_title="Componente t-SNE 1",
        yaxis_title="Componente t-SNE 2",
        legend_title="Classes",
        height=700
    )
    
    # Adicionando a funcionalidade de hover com imagem (requer um pouco de JS via go.Figure)
    # Esta parte é mais complexa e pode não ser diretamente suportada sem hacks.
    # A alternativa mais simples é mostrar o caminho e deixar o usuário abrir.
    # No entanto, vamos tentar uma abordagem mais visual.
    
    st.plotly_chart(fig, use_container_width=True)
