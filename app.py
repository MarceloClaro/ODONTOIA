# app.py
import os
import streamlit as st
st.set_page_config(page_title="ODONTO.IA", layout="wide")

import zipfile
import shutil
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
import gc
from torchcam.methods import SmoothGradCAMpp, ScoreCAM, LayerCAM
from torchvision.transforms.functional import to_pil_image
import cv2
from typing import List, Tuple, Optional, Any, cast, Dict
import json
from datetime import datetime
import copy

# MONAI imports
from monai.data.dataset import Dataset as MONAIDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized,
    RandFlipd, RandRotate90d, RandZoomd, EnsureTyped, Lambdad
)

# ODONTO.IA imports
import config
from utils import (
    set_seed, seed_worker, plot_class_distribution, plot_metrics,
    visualize_augmented_data, display_model_architecture,
    display_environment_info, interpret_results, plot_interactive_embeddings
)
from models import get_model
from trainer import train_loop, compute_metrics, error_analysis, get_optimizer, get_scheduler
from llm_modal import show_disease_modal, get_disease_key
from groq_llm import interpretar_predicao, gerar_prognostico, consulta_groq

# Set seed for reproducibility
set_seed(config.SEED)

def prepare_data_splits(data_dir: str, train_split_ratio: float, seed: int) -> Dict[str, Any]:
    """
    Descobre imagens, divide-as em treino/validação/teste e retorna os datasets MONAI.
    """
    image_files = []
    labels = []
    
    # Descobrir classes e ficheiros
    classes = sorted([d.name for d in os.scandir(data_dir) if d.is_dir()])
    if not classes:
        raise ValueError(f"Nenhum diretório de classe encontrado em '{data_dir}'. Verifique a estrutura do seu dataset.")
        
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    for target_class in classes:
        class_dir = os.path.join(data_dir, target_class)
        for fname in os.listdir(class_dir):
            # Ignorar ficheiros ocultos
            if not fname.startswith('.'):
                image_files.append(os.path.join(class_dir, fname))
                labels.append(class_to_idx[target_class])
    
    if not image_files:
        raise ValueError(f"Nenhuma imagem encontrada em '{data_dir}'.")

    # Criar lista de dicionários para MONAI
    data_dicts = [{"image": img, "label": lab} for img, lab in zip(image_files, labels)]
    
    # Dividir em treino+validação e teste (ex: 80% treino/val, 20% teste)
    train_val_indices, test_indices = train_test_split(
        list(range(len(data_dicts))),
        test_size=0.2,
        stratify=[d['label'] for d in data_dicts],
        random_state=seed
    )

    # Dividir treino e validação (ex: 87.5% de 80% = 70% total)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        train_size=train_split_ratio / 0.8, # Ajustar a proporção
        stratify=[data_dicts[i]['label'] for i in train_val_indices],
        random_state=seed
    )

    train_data = [data_dicts[i] for i in train_indices]
    valid_data = [data_dicts[i] for i in val_indices]
    test_data = [data_dicts[i] for i in test_indices]

    return {
        "train": train_data,
        "valid": valid_data,
        "test": test_data,
        "classes": classes,
        "class_to_idx": class_to_idx,
    }

def run_training_pipeline(app_config: Dict[str, Any]):
    """Função principal para executar o pipeline de treino e avaliação."""
    status_placeholder = st.empty()
    try:
        data_dir = app_config['data_dir']
        status_placeholder.info("A preparar e dividir os dados...")

        # Preparar e dividir os dados de forma robusta
        data_splits = prepare_data_splits(data_dir, app_config['train_split'], config.SEED)
        train_data = data_splits['train']
        valid_data = data_splits['valid']
        test_data = data_splits['test']
        classes = data_splits['classes']
        num_classes = len(classes)
        
        st.success(f"Dados divididos: {len(train_data)} treino, {len(valid_data)} validação, {len(test_data)} teste.")

        # --- MONAI Data Pipeline ---
        train_transforms_list = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys="image", func=lambda x: x[:3, :, :] if x.shape[0] > 3 else x), # Garante 3 canais
            Resized(keys=["image"], spatial_size=(app_config['image_size'], app_config['image_size'])),
            ScaleIntensityd(keys=["image"]),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ]
        
        if app_config['augmentation'] != 'Nenhum':
            train_transforms_list.extend([
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandRotate90d(keys=["image"], prob=0.5, max_k=3),
                RandZoomd(keys=["image"], prob=0.5, min_zoom=0.9, max_zoom=1.1),
            ])
            
        train_transform = Compose(train_transforms_list)

        val_transform = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys="image", func=lambda x: x[:3, :, :] if x.shape[0] > 3 else x),
            Resized(keys=["image"], spatial_size=(app_config['image_size'], app_config['image_size'])),
            ScaleIntensityd(keys=["image"]),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ])

        train_dataset = MONAIDataset(data=train_data, transform=train_transform)
        valid_dataset = MONAIDataset(data=valid_data, transform=val_transform)
        test_dataset = MONAIDataset(data=test_data, transform=val_transform)

        g = torch.Generator().manual_seed(config.SEED)
        train_loader = DataLoader(train_dataset, batch_size=app_config['batch_size'], shuffle=True, worker_init_fn=seed_worker, generator=g, num_workers=2)
        valid_loader = DataLoader(valid_dataset, batch_size=app_config['batch_size'], num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=app_config['batch_size'], num_workers=2)

        # --- Visualizations ---
        st.subheader("Análise do Conjunto de Dados")
        plot_class_distribution(train_dataset, classes, title="Distribuição de Classes (Treino)")
        if app_config['augmentation'] != 'Nenhum':
            visualize_augmented_data(train_loader)

        # --- Loss Function ---
        criterion = nn.CrossEntropyLoss()
        if app_config['use_weighted_loss']:
            class_counts = np.bincount([d['label'] for d in train_data], minlength=num_classes)
            class_weights = 1.0 / (class_counts + 1e-6)
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(config.DEVICE))

        # --- Model, Optimizer, Scheduler ---
        model = get_model(app_config['model_name'], num_classes, fine_tune=app_config['fine_tune'], dropout_rate=app_config['dropout_rate'])
        if model is None:
            st.error("Falha ao carregar o modelo.")
            return None
        display_model_architecture(model, app_config['model_name'])
        model.to(config.DEVICE)

        optimizer = get_optimizer(model, app_config['optimizer'], app_config['learning_rate'], app_config['l2_lambda'])
        scheduler = get_scheduler(optimizer, app_config['scheduler'], app_config['epochs'], len(train_loader), app_config['learning_rate'])
        
        # --- Training ---
        status_placeholder.info("A iniciar o treino do modelo...")
        train_results = train_loop(
            model, train_loader, valid_loader, criterion, optimizer, scheduler,
            app_config['epochs'], app_config['patience'], config.DEVICE,
            l1_lambda=app_config.get('l1_lambda', 0.0), # Compatibilidade
            status_placeholder=status_placeholder
        )
        status_placeholder.empty()

        best_model_wts = train_results['weights']
        history = train_results['history']
        model.load_state_dict(best_model_wts)

        # --- Evaluation ---
        st.write("### Curvas de Aprendizagem")
        plot_metrics(history)

        st.write("### Avaliação no Conjunto de Teste")
        eval_results = compute_metrics(model, test_loader, classes, config.DEVICE)
        metrics_report = eval_results["report"]
        st.pyplot(eval_results["figure"])
        interpret_results(metrics_report, history)

        st.write("### Análise de Erros")
        error_fig = error_analysis(model, test_loader, classes, config.DEVICE)
        if error_fig:
            st.pyplot(error_fig)
        else:
            st.success("✅ Nenhum erro de classificação foi encontrado no conjunto de teste!")

        gc.collect()
        return model, classes, history, metrics_report

    except Exception as e:
        st.error(f"Ocorreu um erro crítico durante o pipeline: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None
    finally:
        status_placeholder.empty()

def extract_features(model: nn.Module, data_dir: str, batch_size: int, classes: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """Extrai 'features' (embeddings) de um dataset."""
    feature_extractor = nn.Sequential(*list(model.children())[:-1]) # Remove a última camada (classificador)
    feature_extractor.eval()

    # MONAI dataset para extração
    image_files = []
    labels = []
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    for target_class in classes:
        class_dir = os.path.join(data_dir, target_class)
        if os.path.isdir(class_dir):
            for fname in os.listdir(class_dir):
                if not fname.startswith('.'):
                    image_files.append(os.path.join(class_dir, fname))
                    labels.append(class_to_idx[target_class])
    
    if not image_files:
        st.error("Nenhuma imagem encontrada para extração de features.")
        return None, None, []

    dataset = MONAIDataset(
        data=[{"image": img, "label": lab} for img, lab in zip(image_files, labels)],
        transform=Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys="image", func=lambda x: x[:3, :, :] if x.shape[0] > 3 else x),
            Resized(keys=["image"], spatial_size=(config.TRAINING_PARAMS['image_size'], config.TRAINING_PARAMS['image_size'])),
            ScaleIntensityd(keys=["image"]),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ])
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
    
    features, all_labels, paths = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="A extrair features"):
            inputs = batch['image'].to(config.DEVICE)
            outputs = feature_extractor(inputs)
            features.append(outputs.cpu().numpy().reshape(len(outputs), -1))
            all_labels.extend(batch['label'].numpy())
            paths.extend([meta['filename_or_obj'] for meta in batch['image_meta_dict']])

    return np.concatenate(features), np.array(all_labels), paths

def main():
    """Função principal da aplicação Streamlit."""
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
    
    with st.sidebar:
        if os.path.exists("logo.png"): st.image("logo.png", width=200)
        st.title("🔬 Configurações do Experimento")
        
        st.header("Modelo")
        model_name = st.selectbox("Arquitetura:", config.AVAILABLE_MODELS, key="model_name")
        fine_tune = st.checkbox("Fine-Tuning Completo", True, key="fine_tune")

        st.header("Treinamento")
        epochs = st.slider("Épocas:", 1, 100, config.TRAINING_PARAMS['epochs'], key="epochs")
        batch_size = st.select_slider("Tamanho do Lote:", [4, 8, 16, 32], config.TRAINING_PARAMS['batch_size'], key="batch_size")
        
        st.header("Otimização")
        optimizer = st.selectbox("Otimizador:", config.AVAILABLE_OPTIMIZERS, key="optimizer")
        learning_rate = st.select_slider("Taxa de Aprendizagem:", [1e-3, 1e-4, 1e-5, 1e-6], config.TRAINING_PARAMS['learning_rate'], key="lr", format="%.0e")
        scheduler = st.selectbox("Agendador de LR:", config.AVAILABLE_SCHEDULERS, key="scheduler")
        
        st.header("Regularização")
        l1_lambda = st.number_input("Regularização L1 (Lasso):", 0.0, 0.1, 0.0, 0.001, key="l1", format="%.4f")
        l2_lambda = st.number_input("Regularização L2 (Weight Decay):", 0.0, 0.1, config.TRAINING_PARAMS['l2_lambda'], 0.001, key="l2", format="%.4f")
        dropout_rate = st.slider("Taxa de Dropout:", 0.0, 0.9, 0.5, 0.1, key="dropout")
        patience = st.number_input("Paciência (Early Stopping):", 3, 20, config.TRAINING_PARAMS['patience'], key="patience")
        use_weighted_loss = st.checkbox("Usar Perda Ponderada", True, key="weighted_loss")
        
        st.header("Aumento de Dados")
        augmentation = st.selectbox("Técnica de Aumento:", config.AUGMENTATION_TECHNIQUES, key="augmentation")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Treinamento", "Análise de Clustering", "Avaliação de Imagem",
        "Comparação de Experimentos", "Análise Técnica da Arquitetura"
    ])

    with tab1:
        st.header("1. Fonte de Dados e Início")
        data_source = st.radio("Selecione a fonte dos dados:", ("Usar dataset local", "Fazer upload de ficheiro ZIP"))

        zip_file = None
        if data_source == "Fazer upload de ficheiro ZIP":
            zip_file = st.file_uploader("Carregue um ficheiro ZIP com as imagens organizadas em pastas por classe", type=["zip"])

        if st.button("🚀 Iniciar Treinamento"):
            app_config = {
                'model_name': model_name, 'fine_tune': fine_tune, 'epochs': epochs,
                'batch_size': batch_size, 'optimizer': optimizer, 'learning_rate': learning_rate,
                'scheduler': scheduler, 'l1_lambda': l1_lambda, 'l2_lambda': l2_lambda,
                'dropout_rate': dropout_rate, 'patience': patience, 'use_weighted_loss': use_weighted_loss,
                'augmentation': augmentation, 'train_split': config.TRAINING_PARAMS['train_split'],
                'image_size': config.TRAINING_PARAMS['image_size']
            }
            
            temp_dir_to_clean = None
            try:
                if zip_file:
                    st.info("A extrair ficheiro ZIP...")
                    temp_dir = tempfile.mkdtemp()
                    temp_dir_to_clean = temp_dir
                    with zipfile.ZipFile(zip_file, 'r') as z:
                        z.extractall(temp_dir)
                    
                    # Heurística para encontrar o diretório principal dentro do ZIP
                    extracted_contents = os.listdir(temp_dir)
                    if len(extracted_contents) == 1 and os.path.isdir(os.path.join(temp_dir, extracted_contents[0])):
                        data_dir = os.path.join(temp_dir, extracted_contents[0])
                    else:
                        data_dir = temp_dir
                    st.success(f"Ficheiro ZIP extraído para: {data_dir}")
                else:
                    data_dir = config.DATASET_PATH

                if os.path.isdir(data_dir):
                    app_config['data_dir'] = data_dir
                    display_environment_info()
                    train_result = run_training_pipeline(app_config)
                    if train_result:
                        st.session_state.model, st.session_state.classes, st.session_state.history, st.session_state.metrics = train_result
                        st.session_state.training_done = True
                        st.session_state.data_dir = data_dir
                        
                        # Guardar resultados
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        results_dir = 'results'
                        os.makedirs(results_dir, exist_ok=True)
                        results_filename = os.path.join(results_dir, f"experiment_{app_config['model_name']}_{timestamp}.json")
                        
                        # Remover tensores e outros objetos não serializáveis
                        serializable_metrics = {k: v for k, v in st.session_state.metrics.items() if isinstance(v, (dict, float, int))}
                        
                        results_data = {
                            "config": app_config,
                            "history": st.session_state.history,
                            "metrics": serializable_metrics
                        }
                        with open(results_filename, 'w') as f:
                            json.dump(results_data, f, indent=4)
                        
                        st.success(f"Experimento concluído! Resultados guardados em `{results_filename}`")
                        st.balloons()
                else:
                    st.error(f"Diretório de dados '{data_dir}' não encontrado.")
            
            finally:
                if temp_dir_to_clean and os.path.exists(temp_dir_to_clean):
                    shutil.rmtree(temp_dir_to_clean)

    with tab2:
        st.header("2. Análise de Clustering (Embeddings)")
        if not st.session_state.training_done:
            st.warning("Treine um modelo primeiro na aba 'Treinamento'.")
        else:
            if st.button("Analisar Clusters"):
                with st.spinner("A extrair features e a analisar clusters..."):
                    model = cast(nn.Module, st.session_state.model)
                    data_dir = cast(str, st.session_state.data_dir)
                    classes = cast(List[str], st.session_state.classes)

                    features, labels, paths = extract_features(model, data_dir, app_config['batch_size'], classes)
                    
                    if features is not None:
                        plot_interactive_embeddings(features, labels, paths, classes)

    with tab3:
        st.header("3. Avaliação e Interpretabilidade (XAI)")
        if not st.session_state.training_done:
            st.warning("Treine um modelo primeiro na aba 'Treinamento'.")
        else:
            model = cast(nn.Module, st.session_state.model)
            classes = cast(List[str], st.session_state.classes)
            
            xai_method = st.selectbox("Método de Interpretabilidade:", config.AVAILABLE_XAI_METHODS)
            eval_image_file = st.file_uploader("Upload de imagem para avaliação", type=["png", "jpg", "jpeg"], key="eval_uploader")
            
            if eval_image_file:
                image = Image.open(eval_image_file).convert("RGB")
                st.image(image, caption='Imagem para avaliação', use_container_width=False, width=300)
    
                # Adicione aqui a lógica de avaliação da imagem (evaluate_image) e visualização (visualize_activations)
                # ...

    with tab4:
        st.header("4. Comparação de Experimentos")
        # Adicione aqui a lógica para carregar e comparar ficheiros de resultados
        # ...

    with tab5:
        st.header("Análise Técnica da Arquitetura do Modelo")
        if not st.session_state.training_done:
            st.warning("Treine um modelo primeiro na aba 'Treinamento'.")
        else:
            # Lógica para gerar e exibir a análise técnica
            # ...

if __name__ == "__main__":
    main()
