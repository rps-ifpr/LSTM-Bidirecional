import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from scipy.stats import zscore
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuração inicial do Streamlit
st.set_page_config(layout="wide")
st.title('Análise de Componentes Principais (PCA) e Previsão com LSTM')


# Função para carregar os dados processados
def carregar_dados_processados(caminho):
    try:
        data = pd.read_csv(caminho, delimiter=';', decimal=',', encoding='latin1')
        st.write("Dados processados carregados com sucesso:")
        st.write(data.head())  # Mostrar uma amostra dos dados carregados para verificar a consistência
    except Exception as e:
        st.error(f"Erro ao carregar os dados processados. Detalhes: {e}")
        return None
    return data


# Função para aplicar PCA e K-Means e visualizar os resultados
def aplicar_pca_kmeans(data):
    # Preparar os dados para PCA
    features = ['PRECIPITAÇÃO TOTAL', 'TEMPERATURA DO AR BULBO SECO (°C)', 'TEMP MAX (°C)',
                'TEMP MÍN (°C)', 'UMIDADE RELATIVA DO AR (%)', 'PRESSAO ATMOSFERICA', 'RADIACAO GLOBAL (Kj/m²)']

    # Verificar se todas as colunas estão presentes
    if not all(feature in data.columns for feature in features):
        st.error("Algumas das colunas especificadas não estão presentes no DataFrame.")
        st.stop()

    # Aplicar a padronização dos dados
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])

    # Aplicar PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data_scaled)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

    # Plotar os componentes principais
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(principalDf['PC1'], principalDf['PC2'], alpha=0.5)
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.set_title('PCA - Componentes Principais')
    ax.grid(True)
    st.pyplot(fig)

    # Identificar possíveis outliers usando score Z
    z_scores = np.abs(zscore(principalDf))
    outliers = (z_scores > 3).any(axis=1)
    outlier_data = principalDf[outliers]

    # Plotar os componentes principais com outliers destacados
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(principalDf['PC1'], principalDf['PC2'], alpha=0.5)
    ax.scatter(outlier_data['PC1'], outlier_data['PC2'], color='red', label='Outliers')  # Outliers marcados em vermelho
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.set_title('PCA - Outliers destacados')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Aplicar K-means para identificar agrupamentos
    kmeans = KMeans(n_clusters=3)
    principalDf['cluster'] = kmeans.fit_predict(principalDf[['PC1', 'PC2']])

    # Plotar os clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in principalDf['cluster'].unique():
        cluster_data = principalDf[principalDf['cluster'] == cluster]
        ax.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}')
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.set_title('PCA - Clusters')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


# Função para normalizar os dados para o modelo LSTM
def normalizar_dados(data, columns):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[columns])
    return data_scaled, scaler


# Função para criar sequências para a LSTM
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps, 0])  # Prevendo com base na precipitação (como exemplo)
    return np.array(X), np.array(y)


# Função para construir o modelo LSTM Bidirecional
def construir_modelo(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(50, activation='relu')))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


# Função para treinar o modelo
def treinar_modelo(model, X_train, y_train, save_path):
    checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping, checkpoint],
                        verbose=1)
    return history


# Função para avaliar o modelo
def avaliar_modelo(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if np.isnan(y_test).any() or np.isnan(y_pred).any():
        raise ValueError("Existem valores NaN nas previsões ou nos dados de teste!")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    return y_pred, rmse, mae, mape, r2


# Função para visualizar as previsões
def visualizar_previsoes(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(y_test, label='Dados Reais', color='blue')
    ax.plot(y_pred, label='Previsões', color='red', linestyle='--')
    ax.set_title('Previsão de Irrigação com LSTM Bidirecional')
    ax.set_xlabel('Tempo')
    ax.set_ylabel('Precipitação Normalizada')
    ax.legend()
    st.pyplot(fig)


# Definir caminhos conforme a estrutura de diretórios
caminho_processado = '../data/processed/INMET_S_SC_A863_ITUPORANGA_PROCESSADO.CSV'
caminho_modelo = '../models/best_model.h5'

# Carregar os dados processados
data_processada = carregar_dados_processados(caminho_processado)

if data_processada is not None:
    # Aplicar PCA e K-Means
    aplicar_pca_kmeans(data_processada)

    # Selecionar as colunas de interesse para o modelo LSTM
    columns_of_interest = ['PRECIPITAÇÃO TOTAL', 'TEMPERATURA DO AR BULBO SECO (°C)', 'TEMP MAX (°C)',
                           'TEMP MÍN (°C)', 'UMIDADE RELATIVA DO AR (%)', 'PRESSAO ATMOSFERICA',
                           'RADIACAO GLOBAL (KJ/M²)', 'hora', 'dia_da_semana', 'mes',
                           'precipitação_média_móvel', 'diferença_temp_ar']

    # Normalização dos dados para LSTM
    data_scaled, scaler = normalizar_dados(data_processada, columns_of_interest)

    # Criar sequências de entrada e saída para o modelo LSTM
    n_steps = 24  # Previsão com base nas últimas 24 horas
    X, y = create_sequences(data_scaled, n_steps)

    # Dividir os dados em treinamento e teste
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Construção e treinamento do modelo LSTM
    input_shape = (n_steps, X.shape[2])
    model = construir_modelo(input_shape)
    treinar_modelo(model, X_train, y_train, caminho_modelo)

    # Avaliação do modelo
    try:
        y_pred, rmse, mae, mape, r2 = avaliar_modelo(model, X_test, y_test)
        st.write(f"Erro Quadrático Médio (RMSE): {rmse:.2f}")
        st.write(f"Erro Médio Absoluto (MAE): {mae:.2f}")
        st.write(f"Erro Percentual Absoluto Médio (MAPE): {mape:.2f}%")
        st.write(f"Coeficiente de Determinação (R²): {r2:.2f}")

        # Visualizar previsões
        visualizar_previsoes(y_test, y_pred)

    except ValueError as e:
        st.error(f"Erro durante a avaliação do modelo: {e}")
