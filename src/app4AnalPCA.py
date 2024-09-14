import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import zscore

# Configuração inicial do Streamlit
st.set_page_config(layout="wide")
st.title('Análise de Componentes Principais (PCA)')

# Carregar os dados com o delimitador correto
data = pd.read_csv('../data/processed/INMET_S_SC_A863_ITUPORANGA_PROCESSADO.CSV', delimiter=';')

# Imprimir os nomes das colunas para verificar
st.write("Colunas disponíveis no DataFrame:", data.columns.tolist())

# Preparar os dados para PCA
features = ['PRECIPITAÇÃO TOTAL', 'TEMPERATURA DO AR BULBO SECO (°C)', 'TEMP MAX (°C)', 'TEMP MÍN (°C)', 'UMIDADE RELATIVA DO AR (%)', 'PRESSAO ATMOSFERICA', 'RADIACAO GLOBAL (Kj/m²)']
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
