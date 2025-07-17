# ODONTO.IA - Detecção de Lesões Bucais com IA

[![Deployment](https://img.shields.io/badge/Deployment-Live-brightgreen)](https://odontoia.streamlit.app/)

## 1. Visão Geral

Este projeto utiliza Redes Neurais Convolucionais (CNNs) para classificar lesões bucais em imagens. A aplicação, construída com Streamlit e PyTorch, permite treinar, avaliar e visualizar o desempenho de diferentes arquiteturas de modelos. Além disso, integra um sistema de consulta acadêmica baseado em LLM para fornecer informações detalhadas e referências científicas sobre as doenças detectadas.

## 2. Ambiente de Desenvolvimento

Siga os passos abaixo para configurar o ambiente de desenvolvimento local.

### 2.1. Pré-requisitos

- Python 3.8+
- `pip` e `venv`

### 2.2. Instalação

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/seu-usuario/ODONTOIA.git
    cd ODONTOIA
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **Instale as dependências:**
    O arquivo `requirements.txt` contém todas as bibliotecas necessárias.
    ```bash
    pip install -r requirements.txt
    ```

### 2.3. Executando a Aplicação

Para iniciar a interface do Streamlit, execute:

```bash
streamlit run app.py
```

A aplicação será aberta em seu navegador padrão.

## 3. Estrutura do Projeto

O projeto está organizado nos seguintes arquivos principais:

-   **`app.py`**: Ponto de entrada da aplicação Streamlit. Controla a interface do usuário, orquestra o pipeline de treinamento e avaliação, e integra todos os outros módulos.
-   **`config.py`**: Arquivo central de configurações. Define hiperparâmetros, modelos disponíveis, otimizadores, transformações de dados e outras constantes globais. **Este é o primeiro lugar para procurar ao ajustar um experimento.**
-   **`trainer.py`**: Contém a lógica principal de treinamento (`train_loop`), avaliação (`compute_metrics`), e análise de erros.
-   **`models.py`**: Define a função `get_model()` que carrega arquiteturas de CNN pré-treinadas (ResNet, DenseNet) e as adapta para a tarefa de classificação.
-   **`utils.py`**: Funções utilitárias para reprodutibilidade (`set_seed`), visualização de dados (`visualize_data`, `plot_metrics`) e outras tarefas de suporte.
-   **`llm_modal.py`**: Implementa o sistema de consulta acadêmica. Contém a classe `DentalDiseaseReference` que busca informações no PubMed e gera descrições detalhadas das doenças.
-   **`requirements.txt`**: Lista de todas as dependências do Python.
-   **`dataset/`**: Diretório padrão para os dados de imagem, embora um arquivo ZIP possa ser usado através da interface.
-   **`results/`**: Diretório onde os resultados dos experimentos (em formato JSON) são salvos.

## 4. Pipeline de Treinamento e Avaliação

O fluxo de trabalho principal é gerenciado pelo `app.py` e pode ser resumido nos seguintes passos:

1.  **Configuração do Experimento**: O usuário seleciona os hiperparâmetros na barra lateral do Streamlit (modelo, taxa de aprendizado, otimizador, etc.).
2.  **Carregamento de Dados**: Os dados são carregados do diretório especificado em `config.DATASET_PATH` ou de um arquivo ZIP enviado pelo usuário.
3.  **Divisão dos Dados**: O dataset é dividido em conjuntos de treino, validação e teste de forma estratificada.
4.  **Aumento de Dados (Data Augmentation)**: As transformações definidas em `config.py` são aplicadas ao conjunto de treino.
5.  **Início do Treinamento**: O `run_training_pipeline()` é chamado.
    -   O modelo é instanciado via `models.get_model()`.
    -   O otimizador e o scheduler são criados via `trainer.get_optimizer()` e `trainer.get_scheduler()`.
    -   O loop de treinamento (`trainer.train_loop`) é executado, iterando através das épocas, calculando a perda, atualizando os pesos e validando o modelo. O Early Stopping é usado para evitar overfitting.
6.  **Avaliação**: Após o treinamento, o melhor modelo é avaliado no conjunto de teste usando `trainer.compute_metrics()`.
7.  **Salvamento dos Resultados**: As configurações, o histórico de treinamento e as métricas finais são salvas em um arquivo JSON no diretório `results/`.

## 5. Funcionalidades Avançadas

### 5.1. Explicação por IA (XAI)

A aplicação utiliza métodos como `Grad-CAM` para visualizar quais partes da imagem o modelo está "olhando" para fazer uma predição. Isso é implementado na função `visualize_activations()` em `app.py`.

### 5.2. Análise de Clustering

Após o treinamento, é possível extrair *features* (embeddings) das imagens usando a penúltima camada do modelo treinado. Esses embeddings são então usados para agrupar imagens semelhantes usando algoritmos como K-Means. Isso ajuda a descobrir padrões latentes nos dados.

### 5.3. Módulo LLM Acadêmico

O `llm_modal.py` fornece um recurso de consulta que:
-   Busca artigos científicos relevantes no PubMed usando a API E-utilities.
-   Apresenta descrições clínicas, sintomas, causas e tratamentos para as doenças.
-   Gera insights clínicos baseados em IA.

## 6. Como Contribuir

Para adicionar novas funcionalidades ou corrigir bugs:

-   **Novos Modelos**: Adicione o nome do modelo à lista `AVAILABLE_MODELS` em `config.py` e atualize a função `get_model()` em `models.py` para lidar com a nova arquitetura.
-   **Novos Otimizadores/Schedulers**: Adicione o nome às listas `AVAILABLE_OPTIMIZERS` ou `AVAILABLE_SCHEDULERS` em `config.py` e atualize as funções `get_optimizer()`/`get_scheduler()` em `trainer.py`.
-   **Estilo de Código**: Siga o estilo de código existente (PEP 8) e adicione docstrings claras às novas funções.

## 7. Contato

Para mais detalhes, entre em contato com o professor e orientador:
[Marcelo Claro](https://www.instagram.com/marceloclaro.geomaker/)
