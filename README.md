# Detecção de Lesões Bucais Utilizando Redes Neurais e Clustering

## Introdução
Este projeto tem como objetivo a detecção de lesões bucais em imagens utilizando técnicas avançadas de aprendizado profundo (deep learning) e algoritmos de clustering. A aplicação usa modelos de redes neurais convolucionais (CNN) pré-treinados, como ResNet18, ResNet50, e DenseNet121, para a classificação de imagens bucais, além de aplicar técnicas de clustering para análise comparativa.

O código está implementado em **PyTorch** e utiliza o **Streamlit** como interface interativa para visualização dos resultados. O projeto inclui transformações de dados, visualizações de métricas de performance, e análise de clusters utilizando **PCA**.

## Funcionalidades Principais
- **Classificação de Imagens**: Treinamento de modelos CNN pré-treinados para a detecção de lesões bucais.
- **Clustering**: Aplicação de algoritmos de clustering (K-Means e Hierarchical Clustering) nos embeddings extraídos das imagens para análise de padrões.
- **Visualização de Ativações**: Utilização de Grad-CAM para visualização das regiões ativadas pelo modelo na detecção de lesões.
- **Suporte a Dados Desbalanceados**: Implementação de perda ponderada para tratar o desbalanceamento de classes no conjunto de dados.
- **Transformações de Dados**: Aumento de dados com transformações como flip horizontal, rotação, recorte, entre outras.

## Requisitos
- Python 3.7+
- PyTorch
- Streamlit
- Scikit-learn
- Torchvision
- Seaborn
- Matplotlib
- OpenCV
- PIL (Python Imaging Library)

## Instalação
Para rodar este projeto localmente, siga os passos abaixo:

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-repositorio.git
   ```
   
2. Navegue até o diretório do projeto:
   ```bash
   cd detect-lesoes-bucais
   ```

3. Crie um ambiente virtual e instale as dependências:
   ```bash
   python -m venv env
   source env/bin/activate  # No Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

4. Rode a aplicação Streamlit:
   ```bash
   streamlit run app.py
   ```

## Uso
1. **Carregar Imagens**: Faça o upload de um arquivo ZIP contendo as imagens bucais organizadas em subdiretórios para cada classe.
   
2. **Configurações de Treinamento**: Selecione o modelo pré-treinado, ajuste o número de classes, as porcentagens de dados para treino e validação, e outros parâmetros na barra lateral.

3. **Treinamento**: O modelo será treinado nas imagens carregadas. Durante o treinamento, métricas como perda e acurácia serão exibidas, e gráficos de desempenho serão gerados.

4. **Clustering**: Após o treinamento, o modelo extrai os embeddings das imagens para aplicar algoritmos de clustering, ajudando na análise de padrões ocultos.

5. **Visualizar Ativações**: Você pode visualizar as ativações da rede utilizando a técnica Grad-CAM para entender quais partes das imagens influenciaram a decisão do modelo.

## Estrutura do Código

### `train_model()`
Treina um modelo de rede neural convolucional pré-treinado com dados bucais. Realiza a divisão do dataset em treino, validação e teste, e retorna o modelo treinado.

### `visualize_data()`
Exibe algumas imagens do dataset para inspeção visual.

### `get_model()`
Retorna o modelo CNN pré-treinado (ResNet ou DenseNet) com as camadas finais ajustadas para o número de classes definidas.

### `compute_metrics()`
Calcula e exibe métricas de performance como a matriz de confusão e o relatório de classificação.

### `perform_clustering()`
Aplica algoritmos de clustering como K-Means e Clustering Hierárquico nas características extraídas pelo modelo.

### `visualize_clusters()`
Exibe a visualização dos clusters gerados com redução de dimensionalidade usando PCA.

### `visualize_activations()`
Utiliza Grad-CAM para visualizar as ativações da rede, destacando as áreas da imagem mais relevantes para a predição.

## Referências
- Huang, G., Liu, Z., Maaten, L., & Weinberger, K. (2017). Densely connected convolutional networks.
- Shorten, C. & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning.
- Petrovska, B., Atanasova-Pacemska, T., Corizzo, R., Mignone, P., Lameski, P., & Zdravevski, E. (2020). Aerial scene classification through fine-tuning.

Para mais detalhes e contato: 
[Guiador - Marcelo Claro](https://www.instagram.com/marceloclaro.geomaker/)
