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

        image_size = app_config.get('image_size', config.TRAINING_PARAMS['image_size'])
        train_transform = Compose([
            LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys="image", func=lambda x: x[:3, :, :] if x.shape[0] > 3 else x),
            Resized(keys=["image"], spatial_size=(image_size, image_size)),
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
            Resized(keys=["image"], spatial_size=(image_size, image_size)),
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
    # A transformação MONAI espera um array numpy com canais primeiro (C, H, W)
    image_np = np.array(image).transpose(2, 0, 1)
    image_tensor = transform(image_np)
    image_tensor = image_tensor.unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    return classes[predicted_idx.item()], confidence.item()

def visualize_activations(model: nn.Module, image: Image.Image, xai_method: str):
    """Gera e exibe os mapas de ativação (XAI)."""
    model.eval()
    target_layer = model.layer4 if hasattr(model, 'layer4') else model.features if hasattr(model, 'features') else None
    if target_layer is None:
        st.error("Não foi possível encontrar a camada alvo para XAI.")
        return

    transform = Compose([
        Resized(spatial_size=(config.TRAINING_PARAMS['image_size'], config.TRAINING_PARAMS['image_size'])),
        EnsureTyped(dtype=torch.float32),
    ])
    image_np = np.array(image).transpose(2, 0, 1)
    input_tensor = transform(image_np).unsqueeze(0).to(config.DEVICE)
    
    try:
        cam_extractor_class = {'SmoothGradCAMpp': SmoothGradCAMpp, 'ScoreCAM': ScoreCAM, 'LayerCAM': LayerCAM}.get(xai_method)
        if not cam_extractor_class:
            st.error(f"Método XAI '{xai_method}' não suportado.")
            return

        with cam_extractor_class(model, target_layer) as cam_extractor:
            out = model(input_tensor)
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        
        # Visualização
        result = to_pil_image(activation_map[0].squeeze(0).cpu(), mode='F')
        resized_map = result.resize(image.size, Image.Resampling.BICUBIC)
        map_np = (np.array(resized_map) - np.min(resized_map)) / (np.max(resized_map) - np.min(resized_map))
        heatmap = cv2.applyColorMap(np.uint8(255 * map_np), cv2.COLORMAP_JET)
        superimposed_img = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) * 0.4 + np.array(image) * 0.6

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(image); ax1.set_title('Original'); ax1.axis('off')
        ax2.imshow(np.uint8(superimposed_img)); ax2.set_title(xai_method); ax2.axis('off')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erro ao gerar o mapa de ativação: {e}")

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
        if os.path.exists("logo.png"): st.image("logo.png", width=200)
        st.title("🔬 Configurações do Experimento")
        st.header("Modelo"); model_name = st.selectbox("Arquitetura:", config.AVAILABLE_MODELS, key='sb_model')
        fine_tune = st.checkbox("Fine-Tuning Completo", True, key='cb_finetune')
        st.header("Treinamento"); epochs = st.slider("Épocas:", 1, 100, 50, key='sl_epochs')
        batch_size = st.select_slider("Tamanho do Lote:", [4, 8, 16, 32], 32, key='sl_batch')
        st.header("Otimização"); optimizer = st.selectbox("Otimizador:", config.AVAILABLE_OPTIMIZERS, key='sb_opt')
        learning_rate = st.select_slider("Taxa de Aprendizagem:", [1e-3, 1e-4, 1e-5, 1e-6], 1e-4, format="%.0e", key='sl_lr')
        scheduler = st.selectbox("Agendador de LR:", config.AVAILABLE_SCHEDULERS, key='sb_sched')
        st.header("Regularização"); l1_lambda = st.number_input("L1 (Lasso):", 0.0, 0.1, 0.0, 0.001, format="%.4f", key='ni_l1')
        l2_lambda = st.number_input("L2 (Weight Decay):", 0.0, 0.1, 0.01, 0.001, format="%.4f", key='ni_l2')
        dropout_rate = st.slider("Dropout:", 0.0, 0.9, 0.5, 0.1, key='sl_drop')
        patience = st.number_input("Paciência:", 3, 20, 10, key='ni_patience')
        use_weighted_loss = st.checkbox("Usar Perda Ponderada", True, key='cb_weightloss')
        st.header("Aumento de Dados"); augmentation = st.selectbox("Técnica de Aumento:", config.AUGMENTATION_TECHNIQUES, key='sb_aug')

    tabs = st.tabs(["Treinamento", "Análise de Clustering", "Avaliação de Imagem", "Comparação", "Análise Técnica"])
    with tabs[0]:
        st.header("1. Fonte de Dados e Início")
        data_source = st.radio("Selecione a fonte:", ("Usar dataset local", "Fazer upload de ZIP"), key='rad_source')
        zip_file = st.file_uploader("Carregue um ZIP", type=["zip"], key='fu_zip') if data_source == "Fazer upload de ZIP" else None
        
        if st.button("🚀 Iniciar Treinamento", key='bt_train'):
            app_config = {k: v for k, v in locals().items() if k not in ['tabs', 'data_source', 'zip_file', 'app_config']}
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
                        st.session_state.model, st.session_state.classes, st.session_state.history, st.session_state.metrics = train_result
                        st.session_state.training_done = True
                        st.session_state.data_dir = data_dir
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        results_dir = 'results'; os.makedirs(results_dir, exist_ok=True)
                        results_filename = os.path.join(results_dir, f"experiment_{model_name}_{timestamp}.json")
                        serializable_metrics = {k: v for k, v in st.session_state.metrics.items() if isinstance(v, (dict, float, int))}
                        
                        config_to_save = {k: v for k, v in app_config.items() if isinstance(v, (str, int, float, bool))}

                        with open(results_filename, 'w') as f: json.dump({"config": config_to_save, "history": st.session_state.history, "metrics": serializable_metrics}, f, indent=4)
                        st.success(f"Resultados guardados em `{results_filename}`"); st.balloons()
            finally:
                if temp_dir_to_clean: shutil.rmtree(temp_dir_to_clean)
    
    with tabs[1]:
        st.header("2. Análise de Clustering (Embeddings)")
        if not st.session_state.training_done: st.warning("Treine um modelo primeiro.")
        else:
            if st.button("Analisar Clusters", key='bt_cluster'):
                st.info("Esta funcionalidade está em desenvolvimento.")
                # Lógica para clustering aqui
                
    with tabs[2]:
        st.header("3. Avaliação e Interpretabilidade (XAI)")
        if not st.session_state.training_done: st.warning("Treine um modelo primeiro.")
        else:
            model, classes = st.session_state.model, st.session_state.classes
            xai_method = st.selectbox("Método XAI:", config.AVAILABLE_XAI_METHODS, key='sb_xai')
            eval_image_file = st.file_uploader("Upload de imagem para avaliação", type=["png", "jpg", "jpeg"], key='fu_eval')
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
        results_dir = 'results'
        if not os.path.exists(results_dir):
            st.info("Nenhum resultado de experimento guardado.")
        else:
            results_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
            selected_files = st.multiselect("Selecione os experimentos para comparar:", results_files, key='ms_compare')
            if selected_files:
                # Lógica para comparação aqui
                pass

    with tabs[4]:
        st.header("5. Análise Técnica da Arquitetura")
        if not st.session_state.training_done: st.warning("Treine um modelo primeiro.")
        else:
            model_name_used = st.session_state.model.__class__.__name__
            prompt = f"Explique de maneira técnica a arquitetura de rede neural {model_name_used}..."
            with st.spinner("Gerando análise técnica..."):
                st.markdown(consulta_groq(prompt, temperature=0.3))

if __name__ == "__main__":
    main()
