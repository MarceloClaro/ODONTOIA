import os
import zipfile
import shutil
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
import streamlit as st
import gc
from torchcam.methods import SmoothGradCAMpp, ScoreCAM, LayerCAM
from torchvision.transforms.functional import to_pil_image
import cv2
from typing import List, Tuple, Optional, Any, cast
import json
from datetime import datetime
import torchvision

# ODONTO.IA imports
import config
from utils import (set_seed, seed_worker, visualize_data, 
                   plot_class_distribution, plot_metrics, visualize_augmented_data)
from models import get_model
from trainer import train_loop, compute_metrics, error_analysis, get_optimizer, get_scheduler
from llm_modal import show_disease_modal, get_disease_key

# Set seed for reproducibility
set_seed(config.SEED)

class CustomDataset(Dataset):
    """Custom Dataset wrapper."""
    def __init__(self, subset: Subset, transform: Optional[Any] = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        original_index = self.subset.indices[index]
        x, y = self.subset.dataset[original_index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.subset)

def run_training_pipeline(app_config):
    """Main function to run the training and evaluation pipeline."""
    try:
        # Apply augmentations based on selection
        train_transform = config.train_transforms_augmented if app_config['augmentation'] != 'Nenhum' else config.train_transforms
        
        # Load datasets directly from the specified folders
        train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_transform)
        valid_dataset = datasets.ImageFolder(root=config.VALID_DIR, transform=config.test_transforms)
        test_dataset = datasets.ImageFolder(root=config.TEST_DIR, transform=config.test_transforms)

        classes = train_dataset.classes
        num_classes = len(classes)

        st.info(f"Dataset carregado: {len(train_dataset)} imagens de treino, "
                f"{len(valid_dataset)} de validação e {len(test_dataset)} de teste.")

        st.subheader("Análise Inicial do Dataset de Treino")
        visualize_data(train_dataset, classes)

    except (FileNotFoundError, IndexError) as e:
        st.error(f"Erro ao carregar dados: {e}. Verifique se os diretórios 'Training', 'Validation' e 'Testing' existem em '{config.DATASET_PATH}'.")
        return None

    g = torch.Generator().manual_seed(config.SEED)
    train_loader = DataLoader(train_dataset, batch_size=app_config['batch_size'], shuffle=True, worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=app_config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=app_config['batch_size'])

    # --- Visualizations ---
    st.subheader("Balanceamento de Classes (Conjunto de Treino)")
    plot_class_distribution(train_dataset, classes)

    if app_config['augmentation'] != 'Nenhum':
        visualize_augmented_data(train_loader)

    # --- Loss Function ---
    criterion = nn.CrossEntropyLoss()
    if app_config['use_weighted_loss']:
        # Use train_dataset.targets which is available in ImageFolder
        class_counts = np.bincount(train_dataset.targets, minlength=num_classes)
        class_weights = 1.0 / (class_counts + 1e-6)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(config.DEVICE))

    # --- Model, Optimizer, Scheduler ---
    model = get_model(app_config['model_name'], num_classes, fine_tune=app_config['fine_tune'])
    if model is None: 
        st.error("Falha ao carregar o modelo.")
        return None
    model.to(config.DEVICE)

    optimizer = get_optimizer(model, app_config['optimizer'], app_config['learning_rate'], app_config['l2_lambda'])
    scheduler = get_scheduler(optimizer, app_config['scheduler'], app_config['epochs'], len(train_loader))

    # --- Training ---
    train_results = train_loop(model, train_loader, valid_loader, criterion, optimizer, scheduler, 
                                app_config['epochs'], app_config['patience'], app_config['augmentation'])
    
    best_model_wts = train_results['weights']
    history = train_results['history']
    
    model.load_state_dict(best_model_wts)
    
    # --- Evaluation ---
    st.write("**Curvas de Aprendizagem**")
    plot_metrics(history)

    st.write("**Avaliação no Conjunto de Teste**")
    metrics_report = compute_metrics(model, test_loader, classes)
    st.write("**Análise de Erros**")
    error_analysis(model, test_loader, classes)
    
    gc.collect()
    return model, classes, history, metrics_report

def extract_features(dataset: Dataset, model: nn.Module, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(model, nn.DataParallel):
        model = model.module # Handle DataParallel wrapper
        
    feature_extractor_layers = []
    if hasattr(model, 'features'): # DenseNet, etc.
        feature_extractor_layers.append(model.features)
        feature_extractor_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    elif hasattr(model, 'fc'): # ResNet
        # Remove the final fully connected layer
        feature_extractor_layers.extend(list(model.children())[:-1])
    else:
        st.error("Arquitetura de modelo não suportada para extração de features.")
        return np.array([]), np.array([])

    feature_extractor = nn.Sequential(*feature_extractor_layers)
    feature_extractor.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    features, labels = [], []
    with torch.no_grad():
        for inputs, lbls in dataloader:
            outputs = feature_extractor(inputs.to(config.DEVICE))
            features.append(outputs.cpu().numpy().reshape(len(outputs), -1))
            labels.extend(lbls.numpy())
    return np.concatenate(features), np.array(labels)

def perform_clustering(features: np.ndarray, num_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    hierarchical = AgglomerativeClustering(n_clusters=num_clusters).fit_predict(features)
    kmeans = KMeans(n_clusters=num_clusters, random_state=config.SEED, n_init=10).fit_predict(features)
    return hierarchical, kmeans

def evaluate_clustering(true_labels: np.ndarray, cluster_labels: np.ndarray, method_name: str):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    st.write(f"**Métricas para {method_name}:**")
    st.write(f"Adjusted Rand Index: {adjusted_rand_score(true_labels, cluster_labels):.4f}")
    st.write(f"Normalized Mutual Information: {normalized_mutual_info_score(true_labels, cluster_labels):.4f}")

def visualize_clusters(features: np.ndarray, true_labels: np.ndarray, hierarchical_labels: np.ndarray, kmeans_labels: np.ndarray, classes: List[str]):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    
    df = pd.DataFrame({
        'pca1': reduced_features[:, 0],
        'pca2': reduced_features[:, 1],
        'true_labels': [classes[l] for l in true_labels],
        'hierarchical': hierarchical_labels,
        'kmeans': kmeans_labels
    })

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    import seaborn as sns
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='hierarchical', palette="deep", ax=axes[0], legend='full').set_title('Clustering Hierárquico')
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='kmeans', palette="deep", ax=axes[1], legend='full').set_title('K-Means')
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='true_labels', palette="tab10", ax=axes[2], legend='full').set_title('Classes Verdadeiras')
    st.pyplot(fig)

def evaluate_image(model: nn.Module, image: Image.Image, classes: List[str]) -> Tuple[str, float]:
    model.eval()
    image_tensor = config.test_transforms(image)
    if not isinstance(image_tensor, torch.Tensor):
        st.error("A transformação da imagem não retornou um tensor.")
        return "Erro", 0.0
        
    image_tensor = image_tensor.unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_item = cast(int, predicted_idx.item())
    return classes[predicted_item], confidence.item()

def visualize_activations(model: nn.Module, image: Image.Image, xai_method: str):
    model.eval()
    
    image_tensor = config.test_transforms(image)
    if not isinstance(image_tensor, torch.Tensor):
        st.error("A transformação da imagem não retornou um tensor.")
        return
        
    input_tensor = image_tensor.unsqueeze(0).to(config.DEVICE)
    
    target_layer: Optional[nn.Module] = None
    # Heuristics to find the last convolutional layer
    if hasattr(model, 'layer4') and isinstance(model.layer4, nn.Module): # ResNet
        target_layer = model.layer4
    elif hasattr(model, 'features') and isinstance(model.features, nn.Module): # DenseNet
        target_layer = model.features
    
    if target_layer is None:
        st.warning("Não foi possível encontrar uma camada alvo adequada para XAI. A visualização pode não ser ideal.")
        # Fallback to the last child module if it's a Sequential block
        if len(list(model.children())) > 0 and isinstance(list(model.children())[-1], nn.Sequential):
             target_layer = list(model.children())[-1][-1] # type: ignore
        if target_layer is None:
             st.error("Não foi possível determinar a camada alvo para o Grad-CAM.")
             return

    cam_extractor = None
    try:
        cam_extractor_class = {
            'SmoothGradCAMpp': SmoothGradCAMpp,
            'ScoreCAM': ScoreCAM,
            'LayerCAM': LayerCAM
        }.get(xai_method)

        if not cam_extractor_class:
            st.error(f"Método XAI '{xai_method}' não suportado.")
            return
            
        cam_extractor = cam_extractor_class(model, target_layer=target_layer)

        with torch.set_grad_enabled(True):
            out = model(input_tensor)
            pred_class = out.argmax().item()
            activation_map = cam_extractor(pred_class, out)

        result = to_pil_image(activation_map[0].squeeze(0).cpu(), mode='F')
        resized_map = result.resize(image.size, Image.Resampling.BICUBIC)
        
        map_np = np.array(resized_map)
        map_np = (map_np - np.min(map_np)) / (np.max(map_np) - np.min(map_np) + 1e-6)
        
        heatmap_np = np.uint8(255 * map_np)
        heatmap = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET) # type: ignore
        
        superimposed_img = np.uint8(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) * 0.4 + np.array(image) * 0.6)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(image); ax1.set_title('Imagem Original'); ax1.axis('off')
        ax2.imshow(superimposed_img); ax2.set_title(xai_method); ax2.axis('off')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erro ao gerar Grad-CAM: {e}")
    finally:
        if cam_extractor:
            cam_extractor.remove_hooks()

def main():
    st.set_page_config(page_title="ODONTO.IA", layout="wide")

    # --- Session State Initialization ---
    if 'training_done' not in st.session_state:
        st.session_state.training_done = False
        st.session_state.model = None
        st.session_state.classes = None
        st.session_state.data_dir = None
        st.session_state.history = None
        st.session_state.metrics = None

    if os.path.exists('capa.png'): st.image('capa.png', use_container_width=True)
    st.title("ODONTO.IA - Bancada de Experimentação")
    
    # --- Sidebar for Configuration ---
    with st.sidebar:
        if os.path.exists("logo.png"): st.image("logo.png", width=200)
        st.title("🔬 Configurações do Experimento")
        
        st.header("Modelo")
        model_name = st.selectbox("Arquitetura:", config.AVAILABLE_MODELS, key="model_name")
        fine_tune = st.checkbox("Fine-Tuning Completo", True, key="fine_tune")

        st.header("Treinamento")
        epochs = st.slider("Épocas:", 1, 500, config.TRAINING_PARAMS['epochs'], key="epochs")
        batch_size = st.select_slider("Tamanho do Lote:", [4, 8, 16, 32, 64], config.TRAINING_PARAMS['batch_size'], key="batch_size")
        
        st.header("Otimização")
        optimizer = st.selectbox("Otimizador:", config.AVAILABLE_OPTIMIZERS, key="optimizer")
        learning_rate = st.select_slider("Taxa de Aprendizagem:", [1e-2, 1e-3, 1e-4, 1e-5], config.TRAINING_PARAMS['learning_rate'], key="lr")
        scheduler = st.selectbox("Agendador de LR:", config.AVAILABLE_SCHEDULERS, key="scheduler")
        
        st.header("Regularização")
        l2_lambda = st.number_input("Regularização L2 (Weight Decay):", 0.0, 0.1, config.TRAINING_PARAMS['l2_lambda'], 0.001, key="l2")
        patience = st.number_input("Paciência (Early Stopping):", 1, 20, config.TRAINING_PARAMS['patience'], key="patience")
        use_weighted_loss = st.checkbox("Usar Perda Ponderada", config.REGULARIZATION_PARAMS['use_weighted_loss'], key="weighted_loss")
        
        st.header("Aumento de Dados")
        augmentation = st.selectbox("Técnica de Aumento:", config.AUGMENTATION_TECHNIQUES, key="augmentation")

    # --- Main App Body ---
    tab1, tab2, tab3, tab4 = st.tabs(["Treinamento", "Análise de Clustering", "Avaliação de Imagem", "Comparação de Experimentos"])

    with tab1:
        st.header("1. Fonte de Dados e Início")
        st.info("Faça o upload de um arquivo ZIP ou use o conjunto de dados local para iniciar o treinamento.")
        zip_file = st.file_uploader("Upload do arquivo ZIP com imagens", type=["zip"], key="zip_uploader")
        
        if st.button("🚀 Iniciar Treinamento"):
            app_config = {
                'model_name': model_name, 'fine_tune': fine_tune, 'epochs': epochs,
                'batch_size': batch_size, 'optimizer': optimizer, 'learning_rate': learning_rate,
                'scheduler': scheduler, 'l2_lambda': l2_lambda, 'patience': patience,
                'use_weighted_loss': use_weighted_loss, 'augmentation': augmentation,
                'train_split': config.TRAINING_PARAMS['train_split']
            }
            
            data_dir = None
            temp_dir_to_clean = None
            try:
                if zip_file:
                    temp_dir = tempfile.mkdtemp()
                    temp_dir_to_clean = temp_dir
                    with zipfile.ZipFile(zip_file, 'r') as z: z.extractall(temp_dir)
                    # Handle cases where the zip extracts to a subdirectory
                    extracted_folders = [f for f in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, f))]
                    if len(extracted_folders) == 1 and extracted_folders[0] != '__MACOSX':
                         data_dir = os.path.join(temp_dir, extracted_folders[0])
                    else:
                         data_dir = temp_dir
                else:
                    data_dir = config.DATASET_PATH

                if os.path.isdir(data_dir):
                    app_config['data_dir'] = data_dir
                    train_result = run_training_pipeline(app_config)
                    if train_result:
                        st.session_state.model, st.session_state.classes, st.session_state.history, st.session_state.metrics = train_result
                        st.session_state.training_done = True
                        st.session_state.data_dir = data_dir
                        
                        # Save results
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        results_filename = f"results/experiment_{app_config['model_name']}_{timestamp}.json"
                        results_data = {
                            "config": app_config,
                            "history": st.session_state.history,
                            "metrics": st.session_state.metrics
                        }
                        with open(results_filename, 'w') as f:
                            json.dump(results_data, f, indent=4)
                        
                        st.success(f"Experimento concluído! Resultados salvos em `{results_filename}`")
                        st.balloons()
                else:
                    st.error(f"Diretório de dados '{data_dir}' não encontrado.")
            
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")
            finally:
                if temp_dir_to_clean: shutil.rmtree(temp_dir_to_clean)

    with tab2:
        st.header("2. Análise de Clustering (Embeddings)")
        if st.button("Analisar Clusters"):
            if st.session_state.training_done and st.session_state.model and st.session_state.data_dir and st.session_state.classes:
                with st.spinner("Extraindo features e analisando clusters..."):
                    model = cast(nn.Module, st.session_state.model)
                    data_dir = cast(str, st.session_state.data_dir)
                    classes = cast(List[str], st.session_state.classes)

                    full_dataset = datasets.ImageFolder(root=data_dir, transform=config.test_transforms)
                    features, labels = extract_features(full_dataset, model, batch_size)
                    
                    if features.size > 0:
                        num_clusters = len(classes)
                        hierarchical_labels, kmeans_labels = perform_clustering(features, num_clusters)
                        evaluate_clustering(labels, hierarchical_labels, "Hierárquico")
                        evaluate_clustering(labels, kmeans_labels, "K-Means")
                        visualize_clusters(features, labels, hierarchical_labels, kmeans_labels, classes)
            else:
                st.warning("Treine um modelo primeiro. Os resultados do treinamento são necessários para esta análise.")

    with tab3:
        st.header("3. Avaliação e Interpretabilidade (XAI)")
        if st.session_state.training_done and st.session_state.model and st.session_state.classes:
            model = cast(nn.Module, st.session_state.model)
            classes = cast(List[str], st.session_state.classes)
            
            xai_method = st.selectbox("Método de Interpretabilidade:", config.AVAILABLE_XAI_METHODS)
            eval_image_file = st.file_uploader("Upload de imagem para avaliação", type=["png", "jpg", "jpeg"], key="eval_uploader")
            
            if eval_image_file:
                image = Image.open(eval_image_file).convert("RGB")
                st.image(image, caption='Imagem para avaliação', use_container_width=True)
                
                class_name, confidence = evaluate_image(model, image, classes)
                st.metric(label="Classe Predita", value=class_name, delta=f"Confiança: {confidence:.2%}")
                
                visualize_activations(model, image, xai_method)
                
                disease_key = get_disease_key(class_name)
                show_disease_modal(class_name, disease_key)
        else:
            st.warning("Treine um modelo primeiro.")

    with tab4:
        st.header("4. Comparação de Experimentos")
        
        results_files = [f for f in os.listdir('results') if f.endswith('.json')]
        
        if not results_files:
            st.info("Nenhum resultado de experimento encontrado. Execute um treinamento para começar.")
        else:
            selected_files = st.multiselect("Selecione os experimentos para comparar:", results_files)

            if st.button("🗑️ Apagar Todos os Resultados", help="Clique para remover todos os arquivos .json da pasta de resultados."):
                try:
                    for file in results_files:
                        os.remove(os.path.join('results', file))
                    st.success("Todos os resultados dos experimentos foram apagados.")
                    # Força o rerender para atualizar a lista de arquivos
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao apagar os arquivos: {e}")
            
            if selected_files:
                all_results = []
                for file in selected_files:
                    with open(os.path.join('results', file), 'r') as f:
                        all_results.append(json.load(f))
                
                # --- Tabela de Comparação ---
                st.subheader("Resumo das Configurações e Métricas")
                summary_data = []
                for res in all_results:
                    conf = res['config']
                    met = res['metrics']
                    summary_data.append({
                        "Modelo": conf.get('model_name', 'N/A'),
                        "Otimizador": conf.get('optimizer', 'N/A'),
                        "LR": conf.get('learning_rate', 'N/A'),
                        "Augmentation": conf.get('augmentation', 'N/A'),
                        "F1-Score (Macro)": met.get('macro avg', {}).get('f1-score', 0),
                        "Acurácia": met.get('accuracy', 0)
                    })
                st.dataframe(pd.DataFrame(summary_data))

                # --- Gráficos de Comparação ---
                st.subheader("Curvas de Aprendizagem Comparadas")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                for res in all_results:
                    label = res['config'].get('model_name', 'exp') + "_" + res['config'].get('optimizer', '')
                    hist = res['history']
                    epochs = range(1, len(hist['train_loss']) + 1)
                    ax1.plot(epochs, hist['val_loss'], 'o-', label=f"Val Loss ({label})")
                    ax2.plot(epochs, hist['val_acc'], 'o-', label=f"Val Acc ({label})")

                ax1.set_title('Perda de Validação')
                ax1.set_xlabel('Épocas'); ax1.set_ylabel('Perda'); ax1.grid(True); ax1.legend()
                ax2.set_title('Acurácia de Validação')
                ax2.set_xlabel('Épocas'); ax2.set_ylabel('Acurácia'); ax2.grid(True); ax2.legend()
                st.pyplot(fig)

if __name__ == "__main__":
    main()
