import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import streamlit as st
import os

# Caminho da pasta onde seus arquivos processados estão localizados (relativo ao script)
CAMINHO_PROCESSADO = 'data/processed'

# Função para analisar e visualizar um único arquivo CSV
def analisar_dados_meteorologicos(arquivo):
    data = pd.read_csv(
        arquivo,
        delimiter=';',
        decimal=','
    )

    # Combinar as colunas 'Data' e 'Hora' em uma única coluna datetime
    data['datetime'] = pd.to_datetime(
        data['Data'] + ' ' + data['Hora'],
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce'
    )

    # Remover as colunas 'Data' e 'Hora' se não forem mais necessárias
    data.drop(['Data', 'Hora'], axis=1, inplace=True)

    # Selecionar as colunas de interesse
    columns_of_interest = [
        'PRECIPITAÇÃO TOTAL',
        'TEMPERATURA DO AR - BULBO SECO, AUTOMÁTICA (°C)',
        'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)',
        'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)',
        'UMIDADE RELATIVA DO AR, AUTOMÁTICA (%)',
        'PRESSÃO ATMOSFÉRICA AO NÍVEL DA ESTAÇÃO, AUTOMÁTICA (mB)',
        'RADIAÇÃO GLOBAL (Kj/m²)'
    ]

    # Criar um dicionário de abreviações para melhorar a legibilidade dos gráficos
    abbreviations = {
        'PRECIPITAÇÃO TOTAL': 'Precipitação',
        'TEMPERATURA DO AR - BULBO SECO, AUTOMÁTICA (°C)': 'Temp. Ar (°C)',
        'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'Temp. Máx. (°C)',
        'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'Temp. Mín. (°C)',
        'UMIDADE RELATIVA DO AR, AUTOMÁTICA (%)': 'Umidade (%)',
        'PRESSÃO ATMOSFÉRICA AO NÍVEL DA ESTAÇÃO, AUTOMÁTICA (mB)': 'Pressão Atm.',
        'RADIAÇÃO GLOBAL (Kj/m²)': 'Radiação (Kj/m²)'
    }

    # Renomear as colunas para as abreviações
    data.rename(columns=abbreviations, inplace=True)
    columns_of_interest = list(abbreviations.values())

    # Reordenar as colunas
    data = data[['datetime'] + columns_of_interest]

    # Verificar e tratar valores nulos
    data.dropna(inplace=True)

    # Normalização dos dados
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[columns_of_interest])

    # Criar DataFrame com os dados normalizados
    data_normalized = pd.DataFrame(data_scaled, columns=columns_of_interest)
    data_normalized['datetime'] = data['datetime'].values

    # ---------------------------------------------
    # 1. Gráfico de Série Temporal Multivariada
    # ---------------------------------------------
    st.header('1. Gráfico de Série Temporal Multivariada')

    # Seleção das variáveis para exibição
    variables_to_plot = st.multiselect(
        'Selecione as variáveis que deseja visualizar:',
        options=columns_of_interest,
        default=columns_of_interest
    )

    # Verificar se há variáveis selecionadas
    if variables_to_plot:
        # Criar figura e eixos
        fig, axs = plt.subplots(len(variables_to_plot), 1, figsize=(15, 3 * len(variables_to_plot)), sharex=True)

        # Se apenas uma variável for selecionada, axs não será uma lista
        if len(variables_to_plot) == 1:
            axs = [axs]

        # Plotar cada variável selecionada
        for ax, column in zip(axs, variables_to_plot):
            ax.plot(data_normalized['datetime'], data_normalized[column], label=column)
            ax.set_title(f'{column} ao Longo do Tempo', fontsize=12)
            ax.set_ylabel('Valor Normalizado', fontsize=10)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True)

        # Configurações finais
        plt.xlabel('Data e Hora', fontsize=12)
        plt.tight_layout()

        # Exibir o gráfico no Streamlit
        st.pyplot(fig)
    else:
        st.warning('Por favor, selecione pelo menos uma variável para visualizar o gráfico.')

    # ---------------------------------------------
    # 2. Matriz de Correlação com Heatmap
    # ---------------------------------------------
    st.header('2. Matriz de Correlação entre Variáveis')

    # Recalcular a matriz de correlação com as variáveis selecionadas
    correlation_matrix = data_normalized[columns_of_interest].corr()

    # Criar o heatmap de correlação
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        xticklabels=columns_of_interest,
        yticklabels=columns_of_interest,
        ax=ax_corr
    )
    ax_corr.set_title('Matriz de Correlação entre Variáveis', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    # Exibir o gráfico no Streamlit
    st.pyplot(fig_corr)

    # ---------------------------------------------
    # 3. Análise de Componentes Principais (PCA)
    # ---------------------------------------------
    st.header('3. Análise de Componentes Principais (PCA)')

    # Aplicar PCA aos dados normalizados
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_normalized[columns_of_interest])

    # Criar um DataFrame com os componentes principais
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # Plotar o scatter plot dos dois primeiros componentes principais
    fig_pca, ax_pca = plt.subplots(figsize=(10, 7))
    scatter = ax_pca.scatter(pca_df['PC1'], pca_df['PC2'], c=data_normalized['Temp. Ar (°C)'], cmap='viridis', alpha=0.5)
    ax_pca.set_title('Análise de Componentes Principais (PCA)', fontsize=14)
    ax_pca.set_xlabel(f'PC1 - {pca.explained_variance_ratio_[0] * 100:.2f}% da Variância', fontsize=12)
    ax_pca.set_ylabel(f'PC2 - {pca.explained_variance_ratio_[1] * 100:.2f}% da Variância', fontsize=12)
    plt.colorbar(scatter, ax=ax_pca, label='Temp. Ar (°C)')
    plt.tight_layout()

    # Exibir o gráfico no Streamlit
    st.pyplot(fig_pca)

    # Exibir a variância explicada pelos componentes principais
    st.write(f'Variância explicada pelo PC1: {pca.explained_variance_ratio_[0] * 100:.2f}%')
    st.write(f'Variância explicada pelo PC2: {pca.explained_variance_ratio_[1] * 100:.2f}%')


# Configuração inicial do Streamlit
st.set_page_config(layout="wide")
st.title('Análise Exploratória de Dados Meteorológicos')

# Obter a lista de arquivos CSV na pasta 'processed'
arquivos_csv = [f for f in os.listdir(CAMINHO_PROCESSADO) if f.endswith('.csv')]

# Iterar sobre cada arquivo CSV e realizar a análise
for arquivo in arquivos_csv:
    # Exibir o nome do arquivo que está sendo processado
    st.header(f"Analisando arquivo: {arquivo}")

    # Criar o caminho completo para o arquivo
    caminho_arquivo = os.path.join(CAMINHO_PROCESSADO, arquivo)

    # Chamar a função de análise
    analisar_dados_meteorologicos(caminho_arquivo)

    # Adicionar um espaço visual entre as análises de cada arquivo
    st.markdown("---")