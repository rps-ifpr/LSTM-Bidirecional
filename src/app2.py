import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuração inicial do Streamlit
st.set_page_config(layout="wide")
st.title('Previsão de Irrigação com LSTM Bidirecional')


# Função para carregar e preparar os dados
def carregar_dados(caminho):
    data = pd.read_csv(caminho, delimiter=';', decimal=',')
    data['datetime'] = pd.to_datetime(data['Data'] + ' ' + data['Hora'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    data.drop(['Data', 'Hora'], axis=1, inplace=True)

    # Convertendo as colunas relevantes para numérico
    cols_to_numeric = ['PRECIPITAÇÃO TOTAL', 'TEMPERATURA DO AR BULBO SECO (°C)', 'TEMP MAX (°C)',
                       'TEMP MÍN (°C)', 'UMIDADE RELATIVA DO AR (%)', 'PRESSAO ATMOSFERICA',
                       'RADIACAO GLOBAL (Kj/m²)']
    for col in cols_to_numeric:
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Converte para numérico, coercing errors to NaN

    # Feature Engineering
    data['hora'] = data['datetime'].dt.hour
    data['dia_da_semana'] = data['datetime'].dt.dayofweek
    data['mes'] = data['datetime'].dt.month
    data['precipitação_média_móvel'] = data['PRECIPITAÇÃO TOTAL'].rolling(window=3).mean()
    data['diferença_temp_ar'] = data['TEMPERATURA DO AR BULBO SECO (°C)'].diff()

    # Remover ou substituir NaNs
    data.fillna(method='ffill', inplace=True)  # Forward fill para NaNs iniciais
    data.dropna(inplace=True)  # Remove linhas restantes com NaN

    return data


# Função para normalizar os dados
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
def treinar_modelo(model, X_train, y_train):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    return history


# Função para avaliar o modelo
def avaliar_modelo(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # Verificar NaNs em y_test e y_pred
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


# Caminho do arquivo processado
caminho_processado = '../data/processed/INMET_S_SC_A863_ITUPORANGA_PROCESSADO.CSV'

# Carregar e preparar os dados
data = carregar_dados(caminho_processado)

# Selecionar as colunas de interesse
columns_of_interest = ['PRECIPITAÇÃO TOTAL', 'TEMPERATURA DO AR BULBO SECO (°C)', 'TEMP MAX (°C)',
                       'TEMP MÍN (°C)', 'UMIDADE RELATIVA DO AR (%)', 'PRESSAO ATMOSFERICA', 'RADIACAO GLOBAL (Kj/m²)',
                       'hora', 'dia_da_semana', 'mes', 'precipitação_média_móvel', 'diferença_temp_ar']

# Normalização dos dados
data_scaled, scaler = normalizar_dados(data, columns_of_interest)

# Criar sequências de entrada e saída
n_steps = 24  # Previsão com base nas últimas 24 horas
X, y = create_sequences(data_scaled, n_steps)

# Dividir os dados em treinamento e teste
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Construção e treinamento do modelo
input_shape = (n_steps, X.shape[2])
model = construir_modelo(input_shape)
treinar_modelo(model, X_train, y_train)

# Avaliação do modelo
try:
    y_pred, rmse, mae, mape, r2 = avaliar_modelo(model, X_test, y_test)
    # Exibir resultados de avaliação
    st.write(f"Erro Quadrático Médio (RMSE): {rmse:.2f}")
    st.write(f"Erro Médio Absoluto (MAE): {mae:.2f}")
    st.write(f"Erro Percentual Absoluto Médio (MAPE): {mape:.2f}%")
    st.write(f"Coeficiente de Determinação (R²): {r2:.2f}")

    # Visualizar previsões
    visualizar_previsoes(y_test, y_pred)

except ValueError as e:
    st.error(f"Erro durante a avaliação do modelo: {e}")
