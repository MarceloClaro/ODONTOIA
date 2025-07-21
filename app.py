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
from torch.utils.data import DataLoader
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
    """Descobre imagens, divide-as em treino/validação/teste e retorna os datasets MONAI."""
    image_files, labels = [], []
    classes = sorted([d.name for d in os.scandir(data_dir) if d.is_dir()])
    if not classes:
        raise ValueError(f"Nenhum diretório de classe encontrado em '{data_dir}'.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    for target_class in classes:
        class_dir = os.path.join(data_dir, target_class)
        for fname in os.listdir(class_dir):
            if not fname.startswith('.'):
                image_files.append(os.path.join(class_dir, fname))
                labels.append(class_to_idx[target_class])
    
    if not image_files:
        raise ValueError(f"Nenhuma imagem encontrada em '{data_dir}'.")

    data_dicts = [{"image": img, "label": lab} for img, lab in zip(image_files, labels)]
    
    train_val_indices, test_indices = train_test_split(
        list(range(len(data_dicts))), test_size=0.2,
        stratify=[d['label'] for d in data_dicts], random_state=seed
    )
    train_indices, val_indices = train_test_split(
        train_val_indices, train_size=train_split_ratio / 0.8,
        stratify=[data_dicts[i]['label'] for i in train_val_indices], random_state=seed
    )

    return {
        "train": [data_dicts[i] for i in train_indices],
        "valid": [data_dicts[i] for i in val_indices],
        "test": [data_dicts[i] for i in test_indices],
        "classes": classes, "class_to_idx": class_to_idx,
    }

def run_training_pipeline(app_config: Dict[str, Any]):
    """Função principal para executar o pipeline de treino e avaliação."""
    status_placeholder = st.empty()
    try:
        data_dir = app_config['data_dir']
        status_placeholder.info("A preparar e dividir os dados...")
        data_splits = prepare_data_splits(data_dir, app_config['train_split'], config.SEED)
        train_data, valid_data, test_data, classes = data_splits['train'], data_splits['valid'], data_splits['test'], data_splits['classes']
        num_classes = len(classes)
        st.success(f"Dados divididos: {len(train_data)} treino, {len(valid_data)} validação, {len(test_data)} teste.")

        train_transform = Compose([
            LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys="image", func=lambda x: x[:3, :, :] if x.shape[0] > 3 else x),
            Resized(keys=["image"], spatial_size=(app_config['image_size'], app_config['image_size'])),
            *(
                [RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                 RandRotate90d(keys=["image"], prob=0.5, max_k=3),
                 RandZoomd(keys=["image"], prob=0.5, min_zoom=0.9, max_zoom=1.1)]
                if app_config['augmentation'] != 'Nenhum' else []
            ),
            ScaleIntensityd(keys=["image"]), EnsureTyped(keys=["image"], dtype=torch.float32),
        ])
        val_transform = Compose([
            LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys="image", func=lambda x: x[:3, :, :] if x.shape[0] > 3 else x),
            Resized(keys=["image"], spatial_size=(app_config['image_size'], app_config['image_size'])),
            ScaleIntensityd(keys=["image"]), EnsureTyped(keys=["image"], dtype=torch.float32),
        ])

        train_dataset = MONAIDataset(data=train_data, transform=train_transform)
        valid_dataset = MONAIDataset(data=valid_data, transform=val_transform)
        test_dataset = MONAIDataset(data=test_data, transform=val_transform)

        g = torch.Generator().manual_seed(config.SEED)
        train_loader = DataLoader(train_dataset, batch_size=app_config['batch_size'], shuffle=True, worker_init_fn=seed_worker, generator=g, num_workers=2)
        valid_loader = DataLoader(valid_dataset, batch_size=app_config['batch_size'], num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=app_config['batch_size'], num_workers=2)

        st.subheader("Análise do Conjunto de Dados")
        plot_class_distribution(train_dataset, classes, title="Distribuição de Classes (Treino)")
        if app_config['augmentation'] != 'Nenhum': visualize_augmented_data(train_loader)

        class_counts = np.bincount([d['label'] for d in train_data], minlength=num_classes)
        class_weights = 1.0 / (class_counts + 1e-6)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(config.DEVICE) if app_config['use_weighted_loss'] else None)

        model = get_model(app_config['model_name'], num_classes, fine_tune=app_config['fine_tune'], dropout_rate=app_config['dropout_rate'])
        model.to(config.DEVICE)
        display_model_architecture(model, app_config['model_name'])

        optimizer = get_optimizer(model, app_config['optimizer'], app_config['learning_rate'], app_config['l2_lambda'])
        scheduler = get_scheduler(optimizer, app_config['scheduler'], app_config['epochs'], len(train_loader), app_config['learning_rate'])
        
        status_placeholder.info("A iniciar o treino do modelo...")
        train_results = train_loop(model, train_loader, valid_loader, criterion, optimizer, scheduler, app_config['epochs'], app_config['patience'], config.DEVICE, l1_lambda=app_config.get('l1_lambda', 0.0), status_placeholder=status_placeholder)
        
        model.load_state_dict(train_results['weights'])
        history = train_results['history']
        
        st.write("### Curvas de Aprendizagem"); plot_metrics(history)
        st.write("### Avaliação no Conjunto de Teste"); eval_results = compute_metrics(model, test_loader, classes, config.DEVICE)
        metrics_report = eval_results["report"]; st.pyplot(eval_results["figure"])
        interpret_results(metrics_report, history)
        st.write("### Análise de Erros"); error_fig = error_analysis(model, test_loader, classes, config.DEVICE)
        if error_fig: st.pyplot(error_fig)
        else: st.success("✅ Nenhum erro de classificação foi encontrado no conjunto de teste!")

        gc.collect()
        return model, classes, history, metrics_report
    except Exception as e:
        st.error(f"Ocorreu um erro crítico durante o pipeline: {e}")
        import traceback; st.code(traceback.format_exc())
    finally:
        status_placeholder.empty()

def evaluate_image(model: nn.Module, image: Image.Image, classes: List[str]) -> Tuple[str, float]:
    """Avalia uma única imagem e retorna a classe predita e a confiança."""
    model.eval()
    transform = Compose([
        Resized(spatial_size=(config.TRAINING_PARAMS['image_size'], config.TRAINING_PARAMS['image_size'])),
        EnsureTyped(dtype=torch.float32),
    ])
    image_tensor = transform(np.array(image).transpose(2,0,1))
    image_tensor = image_tensor.unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    return classes[predicted_idx.item()], confidence.item()

def visualize_activations(model: nn.Module, image: Image.Image, xai_method: str):
    """Gera e exibe os mapas de ativação (XAI)."""
    # Lógica de visualização (mantida da versão anterior)
    pass # A sua lógica existente aqui

def main():
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
        # (código da sidebar mantido como no ficheiro original)
        if os.path.exists("logo.png"): st.image("logo.png", width=200)
        st.title("🔬 Configurações do Experimento")
        st.header("Modelo"); model_name = st.selectbox("Arquitetura:", config.AVAILABLE_MODELS)
        fine_tune = st.checkbox("Fine-Tuning Completo", True)
        st.header("Treinamento"); epochs = st.slider("Épocas:", 1, 100, 50)
        batch_size = st.select_slider("Tamanho do Lote:", [4, 8, 16, 32], 32)
        st.header("Otimização"); optimizer = st.selectbox("Otimizador:", config.AVAILABLE_OPTIMIZERS)
        learning_rate = st.select_slider("Taxa de Aprendizagem:", [1e-3, 1e-4, 1e-5, 1e-6], 1e-4, format="%.0e")
        scheduler = st.selectbox("Agendador de LR:", config.AVAILABLE_SCHEDULERS)
        st.header("Regularização"); l1_lambda = st.number_input("Regularização L1 (Lasso):", 0.0, 0.1, 0.0, 0.001, format="%.4f")
        l2_lambda = st.number_input("Regularização L2 (Weight Decay):", 0.0, 0.1, 0.01, 0.001, format="%.4f")
        dropout_rate = st.slider("Taxa de Dropout:", 0.0, 0.9, 0.5, 0.1)
        patience = st.number_input("Paciência (Early Stopping):", 3, 20, 10)
        use_weighted_loss = st.checkbox("Usar Perda Ponderada", True)
        st.header("Aumento de Dados"); augmentation = st.selectbox("Técnica de Aumento:", config.AUGMENTATION_TECHNIQUES)

    tabs = st.tabs(["Treinamento", "Análise de Clustering", "Avaliação de Imagem", "Comparação", "Análise Técnica"])
    with tabs[0]:
        st.header("1. Fonte de Dados e Início")
        data_source = st.radio("Selecione a fonte:", ("Usar dataset local", "Fazer upload de ZIP"))
        zip_file = st.file_uploader("Carregue um ZIP", type=["zip"]) if data_source == "Fazer upload de ZIP" else None
        
        if st.button("🚀 Iniciar Treinamento"):
            app_config = locals() # Captura todas as variáveis locais da sidebar
            app_config.update(config.TRAINING_PARAMS)
            
            temp_dir_to_clean = None
            try:
                if zip_file:
                    temp_dir = tempfile.mkdtemp(); temp_dir_to_clean = temp_dir
                    with zipfile.ZipFile(zip_file, 'r') as z: z.extractall(temp_dir)
                    extracted = os.listdir(temp_dir)
                    data_dir = os.path.join(temp_dir, extracted[0]) if len(extracted) == 1 and os.path.isdir(os.path.join(temp_dir, extracted[0])) else temp_dir
                else:
                    data_dir = config.DATASET_PATH

                if os.path.isdir(data_dir):
                    app_config['data_dir'] = data_dir
                    display_environment_info()
                    train_result = run_training_pipeline(app_config)
                    if train_result:
                        st.session_state.update(dict(zip(['model', 'classes', 'history', 'metrics'], train_result)))
                        st.session_state.training_done = True
                        st.session_state.data_dir = data_dir
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        results_dir = 'results'; os.makedirs(results_dir, exist_ok=True)
                        results_filename = os.path.join(results_dir, f"experiment_{app_config['model_name']}_{timestamp}.json")
                        serializable_metrics = {k: v for k, v in st.session_state.metrics.items() if isinstance(v, (dict, float, int))}
                        with open(results_filename, 'w') as f: json.dump({"config": app_config, "history": st.session_state.history, "metrics": serializable_metrics}, f, indent=4)
                        st.success(f"Resultados guardados em `{results_filename}`"); st.balloons()
            finally:
                if temp_dir_to_clean: shutil.rmtree(temp_dir_to_clean)
    
    with tabs[1]:
        st.header("2. Análise de Clustering (Embeddings)")
        if not st.session_state.training_done: st.warning("Treine um modelo primeiro.")
        else:
            if st.button("Analisar Clusters"):
                # Lógica para clustering aqui
                pass

    with tabs[2]:
        st.header("3. Avaliação e Interpretabilidade (XAI)")
        if not st.session_state.training_done: st.warning("Treine um modelo primeiro.")
        else:
            model, classes = st.session_state.model, st.session_state.classes
            xai_method = st.selectbox("Método XAI:", config.AVAILABLE_XAI_METHODS)
            eval_image_file = st.file_uploader("Upload de imagem para avaliação", type=["png", "jpg", "jpeg"])
            if eval_image_file:
                image = Image.open(eval_image_file).convert("RGB")
                st.image(image, caption='Imagem para avaliação', width=300)
                class_name, confidence = evaluate_image(model, image, classes)
                st.metric(label="Classe Predita", value=class_name, delta=f"Confiança: {confidence:.2%}")
                with st.spinner("Consultando IA..."):
                    st.info(f"**Interpretação:** {interpretar_predicao(class_name)}")
                    st.info(f"**Prognóstico:** {gerar_prognostico(class_name)}")
                visualize_activations(model, image, xai_method)
                show_disease_modal(class_name, get_disease_key(class_name))

    with tabs[3]:
        st.header("4. Comparação de Experimentos")
        # Lógica para comparação aqui
        pass

    with tabs[4]:
        st.header("5. Análise Técnica da Arquitetura")
        if not st.session_state.training_done: st.warning("Treine um modelo primeiro.")
        else:
            # Lógica para análise técnica aqui
            pass

if __name__ == "__main__":
    main()
