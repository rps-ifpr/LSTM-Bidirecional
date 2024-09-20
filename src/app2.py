import streamlit as st
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Configuração inicial do Streamlit
st.set_page_config(layout="wide")
st.title('Treinamento de Modelo LSTM Bidirecional')

# Diretórios de dados e modelos
base_dir = './data/'
model_dir = './models'

# Criação de diretórios se não existirem
os.makedirs(model_dir, exist_ok=True)
for dir in [base_dir, os.path.join(base_dir, 'training'), os.path.join(base_dir, 'test'), os.path.join(base_dir, 'validation')]:
    os.makedirs(dir, exist_ok=True)

train_dir = os.path.join(base_dir, 'training')
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')

# Função para carregar e preparar dados
def load_data(directory):
    data_frames = []
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path, sep=';', decimal=',')
                    data_frames.append(df)
                except Exception as e:
                    st.error(f'Erro ao carregar {file}: {e}')
        return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()
    else:
        st.error(f'Diretório não encontrado: {directory}')
        return pd.DataFrame()

# Botão para carregar e preparar os dados
if st.button('Carregar e Preparar Dados'):
    X_train = load_data(train_dir)
    X_test = load_data(test_dir)
    X_val = load_data(validation_dir)

    if not X_train.empty:
        y_train = X_train.pop(X_train.columns[-1]).values
        y_test = X_test.pop(X_test.columns[-1]).values
        y_val = X_val.pop(X_val.columns[-1]).values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_val_scaled = scaler.transform(X_val)

        st.success('Dados carregados e preprocessados com sucesso!')

        # Definição do modelo LSTM
        model = Sequential([
            Bidirectional(LSTM(50, return_sequences=True, input_shape=(X_train_scaled.shape[1], 1))),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        checkpoint_path = os.path.join(model_dir, 'best_model.h5')
        checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Botão para iniciar o treinamento do modelo
        if st.button('Treinar Modelo'):
            with st.spinner('Treinando o modelo...'):
                history = model.fit(X_train_scaled, y_train, epochs=50, validation_data=(X_val_scaled, y_val), callbacks=[checkpoint, early_stop], verbose=1)
                st.success('Treinamento concluído!')
                model.load_weights(checkpoint_path)
                test_loss = model.evaluate(X_test_scaled, y_test)
                st.write(f'Perda no teste: {test_loss}')

                # Visualização das perdas
                fig, ax = plt.subplots()
                ax.plot(history.history['loss'], label='Perda de Treinamento')
                ax.plot(history.history['val_loss'], label='Perda de Validação')
                ax.set_title('Perda durante o Treinamento')
                ax.set_xlabel('Época')
                ax.set_ylabel('Perda')
                ax.legend()
                st.pyplot(fig)
