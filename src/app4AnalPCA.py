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

# Carregar os dados
data = pd.read_csv('../data/processed/INMET_S_SC_A863_ITUPORANGA_PROCESSADO.CSV', delimiter=';')

# Verificação das colunas disponíveis
st.write("Colunas disponíveis no DataFrame:", data.columns.tolist())

# Preparação dos dados para PCA
features = ['PRECIPITAÇÃO TOTAL', 'TEMPERATURA DO AR BULBO SECO (°C)', 'TEMP MAX (°C)', 'TEMP MÍN (°C)', 'UMIDADE RELATIVA DO AR (%)', 'PRESSAO ATMOSFERICA', 'RADIACAO GLOBAL (Kj/m²)']
if not all(feature in data.columns for feature in features):
    st.error("Algumas das colunas especificadas não estão presentes no DataFrame.")
    st.stop()

# Padronização dos dados
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Aplicação do PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data_scaled)
principalDf = pd.DataFrame(principalComponents, columns=['PC1', 'PC2'])

# Explicação dos eixos
explained_variance = pca.explained_variance_ratio_

# Plotagem dos componentes principais
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(principalDf['PC1'], principalDf['PC2'], alpha=0.5)
ax.set_xlabel(f'Componente Princ. 1 (Exp{explained_variance[0]*100:.2f}% da variância)')
ax.set_ylabel(f'Componente Princ. 2 (Exp{explained_variance[1]*100:.2f}% da variância)')
ax.set_title('PCA - Componentes Principais')
ax.grid(True)
st.pyplot(fig)

# Identificação e plotagem de outliers
z_scores = np.abs(zscore(principalDf))
outliers = (z_scores > 3).any(axis=1)
outlier_data = principalDf[outliers]

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(principalDf['PC1'], principalDf['PC2'], alpha=0.5)
ax.scatter(outlier_data['PC1'], outlier_data['PC2'], color='red', label='Outliers (Z > 3)')
ax.set_xlabel('Componente Princ. 1')
ax.set_ylabel('Componente Princ. 2')
ax.set_title('PCA - Outliers Destacados')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Aplicação do K-means e plotagem de clusters
kmeans = KMeans(n_clusters=3)
principalDf['cluster'] = kmeans.fit_predict(principalDf[['PC1', 'PC2']])

fig, ax = plt.subplots(figsize=(10, 6))
for cluster in principalDf['cluster'].unique():
    cluster_data = principalDf[principalDf['cluster'] == cluster]
    ax.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}')
ax.set_xlabel('Componente Princ. 1')
ax.set_ylabel('Componente Princ. 2')
ax.set_title('PCA - Clusters')
ax.legend()
ax.grid(True)
st.pyplot(fig)

