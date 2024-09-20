#Seleção e Salvamento de Dados Processados
import pandas as pd

# Caminhos dos arquivos
caminho_limpo = '../data/raw/INMET_S_SC_A863_ITUPORANGA_LIMPO.CSV'
caminho_processado = '../data/processed/INMET_S_SC_A863_ITUPORANGA_PROCESSADO.CSV'

# Carregar o arquivo limpo
data = pd.read_csv(caminho_limpo, delimiter=';', decimal=',')

# Selecionar apenas as colunas relevantes
columns_of_interest = [
    'Data', 'Hora', 'PRECIPITAÇÃO TOTAL', 'TEMPERATURA DO AR BULBO SECO (°C)',
    'TEMP MAX (°C)', 'TEMP MÍN (°C)', 'UMIDADE RELATIVA DO AR (%)',
    'PRESSAO ATMOSFERICA', 'RADIACAO GLOBAL (Kj/m²)'
]
filtered_data = data[columns_of_interest]

# Remover linhas com dados faltantes
filtered_data = filtered_data.dropna()

# Salvar o novo arquivo CSV
filtered_data.to_csv(caminho_processado, index=False, sep=';', decimal=',')

print(f'Arquivo salvo com sucesso em: {caminho_processado}')
