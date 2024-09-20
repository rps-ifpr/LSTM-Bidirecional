import os
import pandas as pd

# Caminhos dos arquivos
caminho_original = '../data/raw/INMET_S_SC_A863_ITUPORANGA.CSV'
caminho_limpo = '../data/raw/INMET_S_SC_A863_ITUPORANGA_LIMPO.CSV'

# Verifica a existência do arquivo e processa se encontrado
if os.path.exists(caminho_original):
    print("Arquivo encontrado, processando...")
    dados = pd.read_csv(caminho_original, encoding='latin1', skiprows=8, sep=';')
    dados.columns = [
        # Renomeando todas as colunas relevantes diretamente aqui
        'Data', 'Hora', 'PRECIPITAÇÃO TOTAL', 'PRESSAO ATMOSFERICA',
        'PRESSÃO ATMOSFERICA MAX', 'PRESSÃO ATMOSFERICA MIN',
        'RADIACAO GLOBAL', 'TEMPERATURA DO AR', 'TEMP PONTO DE ORVALHO',
        'TEMP MAX', 'TEMP MÍN', 'TEMP ORVALHO MAX', 'TEMP ORVALHO MIN',
        'UMIDADE REL. MAX', 'UMIDADE REL. MIN', 'UMIDADE RELATIVA DO AR',
        'VENTO DIREÇÃO', 'VENTO RAJADA MAX', 'VENTO VELOCIDADE'
    ]
    dados['Hora'] = pd.to_datetime(dados['Hora'].str.replace(' UTC', ''), format='%H%M').dt.time
    dados['Data'] = pd.to_datetime(dados['Data'], format='%Y/%m/%d')
    for coluna in dados.select_dtypes(include=['object']).columns:
        dados[coluna] = pd.to_numeric(dados[coluna].str.replace(',', '.'), errors='coerce')
    dados.ffill(inplace=True)
    dados.to_csv(caminho_limpo, index=False, sep=';')
    print("Dados limpos salvos com sucesso.")
else:
    print(f"Arquivo não encontrado: {caminho_original}")
