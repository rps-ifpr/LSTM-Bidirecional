import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configuração inicial do Streamlit
st.set_page_config(layout="wide")
st.title('Previsão de Irrigação com LSTM Bidirecional')

# Carregar e preparar os dados
caminho_processado = '../data/processed/INMET_S_SC_A863_ITUPORANGA_PROCESSADO.CSV'
data = pd.read_csv(caminho_processado, delimiter=';', decimal=',')
data['datetime'] = pd.to_datetime(data['Data'] + ' ' + data['Hora'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
data.drop(['Data', 'Hora'], axis=1, inplace=True)

# Selecionar as colunas de interesse
columns_of_interest = ['PRECIPITAÇÃO TOTAL', 'TEMPERATURA DO AR BULBO SECO (°C)', 'TEMP MAX (°C)',
                       'TEMP MÍN (°C)', 'UMIDADE RELATIVA DO AR (%)', 'PRESSAO ATMOSFERICA', 'RADIACAO GLOBAL (Kj/m²)']

# Normalização dos dados
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[columns_of_interest])

# Dividir os dados em sequência de entrada e rótulo (target)
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps, 0])  # Prevendo com base na precipitação (como exemplo)
    return np.array(X), np.array(y)

n_steps = 24  # Exemplo: Previsão com base nas últimas 24 horas
X, y = create_sequences(data_scaled, n_steps)

# Dividir os dados em treinamento e teste
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Construção do modelo LSTM Bidirecional
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=True), input_shape=(n_steps, X.shape[2])))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(50, activation='relu')))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Treinamento do modelo
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Avaliação do modelo
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

st.write(f"Erro Quadrático Médio (RMSE): {rmse:.2f}")
st.write(f"Erro Médio Absoluto (MAE): {mae:.2f}")

# Visualização das previsões
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(y_test, label='Dados Reais', color='blue')
ax.plot(y_pred, label='Previsões', color='red', linestyle='--')
ax.set_title('Previsão de Irrigação com LSTM Bidirecional')
ax.set_xlabel('Tempo')
ax.set_ylabel('Precipitação Normalizada')
ax.legend()
st.pyplot(fig)
