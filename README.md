# Detecção de Lesões Bucais Utilizando Redes Neurais Convolucionais e Clustering

## Visão Geral
Este projeto utiliza redes neurais convolucionais (CNNs) pré-treinadas para a detecção de lesões bucais em imagens, complementado por técnicas de agrupamento (clustering) para análise de padrões latentes nos dados. Ele também implementa estratégias avançadas de visualização de ativação para destacar as regiões das imagens que mais influenciam as previsões do modelo, utilizando **Grad-CAM** (Gradient-weighted Class Activation Mapping). O objetivo principal é detectar lesões bucais de forma precisa e explicar os resultados de forma compreensível.

## 🆕 **Novo Recurso: Sistema de Consulta Acadêmica Inteligente com IA**

### 🤖 Módulo LLM Modal Integrado
O sistema agora inclui um módulo de inteligência artificial avançado que oferece:

#### 📚 **Descrições Clínicas Detalhadas**
- Informações médicas precisas sobre 7 doenças bucais do dataset MOD
- Sintomas, causas e tratamentos baseados em literatura científica
- Terminologia médica apropriada para profissionais da saúde

#### 🔬 **Referências do PubMed**
- Busca automática de artigos científicos relevantes
- Integração direta com a base de dados PubMed/MEDLINE
- Acesso a abstracts e links para artigos completos
- Referências atualizadas para suporte à prática clínica

#### 🎯 **Análise Inteligente**
- Insights clínicos gerados por IA
- Correlações entre achados visuais e manifestações clínicas
- Suporte à tomada de decisão diagnóstica

#### 📖 **Doenças Cobertas**
1. **Gangivoestomatite** (Gingivostomatitis)
2. **Aftas** (Aphthous Stomatitis)
3. **Herpes Labial** (Herpes Simplex Labialis)
4. **Líquen Plano Oral** (Oral Lichen Planus)
5. **Candidíase Oral** (Oral Candidiasis)
6. **Câncer de Boca** (Mouth Cancer)
7. **Câncer Oral** (Oral Carcinoma)

### 🚀 **Como Usar o Novo Recurso**
1. **Durante o treinamento:** Explore informações sobre as classes de doenças no painel lateral
2. **Após a predição:** Clique em "Ver Informações" para detalhes acadêmicos da doença identificada
3. **Consulta independente:** Use o seletor no painel lateral para estudar qualquer doença

### 🔧 **Implementação Técnica**
- **Módulo**: `llm_modal.py` - Sistema completo de consulta acadêmica
- **Integração**: API PubMed para busca de referências científicas
- **Interface**: Modal interativo com 3 abas (Descrição, Referências, Análise LLM)
- **Dados**: Base de conhecimento médico estruturada para cada doença

## Objetivos

- **Classificação de Lesões Bucais**: Treinamento de CNNs pré-treinadas (ResNet18, ResNet50, DenseNet121) para identificar diferentes tipos de lesões bucais em imagens.
- **Clustering**: Uso de técnicas de clustering (K-Means e Clustering Hierárquico) para encontrar padrões e agrupar imagens similares com base nas representações aprendidas pelo modelo.
- **Grad-CAM**: Visualização das áreas de interesse nas imagens que ativam o modelo durante o processo de classificação.
- **Perda Ponderada**: Implementação de perda ponderada para lidar com classes desbalanceadas no conjunto de dados, garantindo que classes minoritárias sejam adequadamente representadas.
- **🆕 Consulta Acadêmica**: Sistema inteligente para consulta de informações médicas detalhadas e referências científicas.
  
## Requisitos Técnicos

- **Python**: 3.7+
- **Framework**: PyTorch
- **Interface Gráfica**: Streamlit
- **Bibliotecas**:
  - torchvision
  - torchcam
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - OpenCV
  - PIL (Python Imaging Library)
  - requests (para integração PubMed)

## Instalação e Execução

### Passos para instalação:
1. **Clonar o repositório:**
   ```bash
   git clone https://github.com/seu-repositorio.git
   ```
   
2. **Acessar o diretório do projeto:**
   ```bash
   cd detect-lesoes-bucais
   ```

3. **Criar ambiente virtual e instalar dependências:**
   ```bash
   python -m venv env
   source env/bin/activate  # No Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Iniciar a aplicação:**
   ```bash
   streamlit run app.py
   ```

## Arquitetura do Projeto

O projeto é dividido em três componentes principais:

1. **Treinamento Supervisionado**: Utiliza redes neurais convolucionais pré-treinadas em grandes conjuntos de dados (ex. ImageNet) e ajusta as últimas camadas para aprender as características de lesões bucais. A técnica de aumento de dados (data augmentation) é aplicada para gerar variações de imagens e melhorar a robustez do modelo.

2. **Clustering para Análise Não-Supervisionada**: Após o treinamento supervisionado, o modelo extrai as representações intermediárias (features) de cada imagem. Essas features são, então, usadas como entrada para algoritmos de clustering (K-Means e Clustering Hierárquico) para identificar agrupamentos de imagens baseados em similaridades latentes.

3. **Visualização das Ativações com Grad-CAM**: Esta técnica é utilizada para gerar mapas de calor que destacam as regiões da imagem que mais ativaram o modelo durante a classificação, proporcionando uma explicação visual das decisões do modelo.

## Pipeline de Treinamento

1. **Definição de Transformações de Dados**:
   - **Aumento de Dados (Data Augmentation)**: Inclui transformações como flips horizontais, rotações, jitter de cor, e recortes aleatórios para aumentar a variabilidade das imagens de treino.
   - **Normalização**: Aplicada para garantir que os valores de pixel estejam em uma escala adequada para o treinamento.

2. **Carregamento do Dataset**:
   - O dataset é carregado utilizando a classe `ImageFolder` do `torchvision`, que organiza as imagens em pastas nomeadas de acordo com as classes. Isso facilita a organização e o pré-processamento das imagens.

3. **Treinamento do Modelo**:
   - Modelos CNN pré-treinados (ResNet18, ResNet50, DenseNet121) são carregados e ajustados para a tarefa de classificação. As camadas finais são modificadas para o número de classes no dataset.
   - **Perda Ponderada**: Para lidar com o desbalanceamento das classes, pesos são calculados com base na frequência das classes no conjunto de treino e aplicados à função de perda.
   
4. **Avaliação e Visualização**:
   - **Métricas**: Matrizes de confusão, AUC-ROC e relatórios de classificação são gerados para avaliar o desempenho.
   - **Ativações Grad-CAM**: Mapas de ativação são gerados para visualizar quais regiões das imagens mais influenciam as decisões do modelo.

## Principais Funções

### `train_model()`
Treina o modelo CNN selecionado nas imagens de lesões bucais. Este método divide o conjunto de dados em treino, validação e teste, utilizando o PyTorch para otimização do modelo.

**Parâmetros**:
- `data_dir`: Caminho para o diretório contendo o dataset de imagens.
- `num_classes`: Número de classes a serem classificadas.
- `model_name`: Nome do modelo pré-treinado a ser utilizado.
- `fine_tune`: Se o ajuste fino (fine-tuning) será aplicado a todas as camadas ou apenas às finais.
- `epochs`: Número de épocas de treinamento.
- `learning_rate`: Taxa de aprendizado.
- `batch_size`: Tamanho do lote para o DataLoader.
- `use_weighted_loss`: Se a perda ponderada será usada para classes desbalanceadas.

### `visualize_data()`
Exibe visualmente um subconjunto de imagens do dataset junto com suas respectivas classes, para uma inspeção visual rápida.

### `get_model()`
Retorna o modelo CNN pré-treinado ajustado para a tarefa de classificação de lesões bucais. Suporta ResNet18, ResNet50 e DenseNet121.

### `perform_clustering()`
Aplica os algoritmos de clustering **K-Means** e **Clustering Hierárquico** nas features extraídas pelo modelo, gerando rótulos de agrupamento para análise.

### `compute_metrics()`
Gera as métricas de avaliação como matriz de confusão, relatório de classificação e curva ROC. Essencial para validar o desempenho do modelo após o treinamento.

### `visualize_activations()`
Utiliza Grad-CAM para gerar um mapa de ativação destacando as áreas mais importantes de uma imagem na tomada de decisão do modelo.

## Visualização de Dados

### Grad-CAM
O Grad-CAM é aplicado para gerar visualizações das áreas da imagem que mais ativaram a rede neural. Isso ajuda a explicar as decisões da rede e é essencial para entender como o modelo toma decisões, o que é particularmente importante em aplicações médicas.

### Clustering
Após o treinamento, os embeddings das imagens são usados em algoritmos de clustering (K-Means e Clustering Hierárquico). Isso permite uma análise adicional para verificar se há padrões latentes ou agrupamentos de lesões semelhantes.

## Exemplos de Execução

1. **Treinamento Supervisionado**:
   Após carregar o dataset, selecione o modelo, defina os hiperparâmetros e inicie o treinamento.

   ```python
   model, classes = train_model(
       data_dir='data/lesoes_bucais',
       num_classes=5,
       model_name='ResNet18',
       fine_tune=True,
       epochs=50,
       learning_rate=0.001,
       batch_size=32,
       train_split=0.7,
       valid_split=0.15,
       use_weighted_loss=True,
   )
   ```

2. **Clustering**:
   Após o treinamento do modelo, utilize o clustering para identificar padrões de agrupamento.

   ```python
   features, labels = extract_features(dataset, model, batch_size=32)
   hierarchical_labels, kmeans_labels = perform_clustering(features, num_clusters=5)
   visualize_clusters(features, labels, hierarchical_labels, kmeans_labels, classes)
   ```

## Considerações Finais

Este projeto fornece uma abordagem robusta para a detecção de lesões bucais utilizando aprendizado profundo. A combinação de CNNs pré-treinadas com técnicas de clustering e Grad-CAM oferece não só uma solução precisa, mas também interpretável, permitindo entender as decisões do modelo e os agrupamentos formados. A implementação de técnicas como **perda ponderada** garante que as classes minoritárias sejam adequadamente tratadas, aumentando a confiabilidade dos resultados.

## Referências

- Huang, G., Liu, Z., Maaten, L., & Weinberger, K. (2017). **Densely connected convolutional networks**. CVPR.
- Buda, M., Maki, A., & Mazurowski, M. (2018). **A systematic study of the class imbalance problem in convolutional neural networks**. Neural Networks, 106, 249-259.
- Shorten, C., & Khoshgoftaar, T. M. (2019). **A survey on image data augmentation for deep learning**. Journal of Big Data.
  
Para mais detalhes, entre em contato com o professor e orientador:
[Guiador - Marcelo Claro](https://www.instagram.com/marceloclaro

.geomaker/)
