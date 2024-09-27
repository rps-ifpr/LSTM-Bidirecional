import os
import pandas as pd

def processar_dados_limpos(diretorio_limpos, diretorio_processados):
    """
    Processa arquivos CSV limpos, selecionando colunas relevantes e salvando em um novo arquivo.
    """
    for filename in os.listdir(diretorio_limpos):
        if filename.endswith('_LIMPO.CSV'):
            caminho_limpo = os.path.join(diretorio_limpos, filename)
            caminho_processado = os.path.join(diretorio_processados, filename.replace('_LIMPO.CSV', '_PROCESSADO.CSV'))
            print(f"Processando arquivo: {filename}")

            # Carregar o arquivo limpo
            try:
                data = pd.read_csv(caminho_limpo, delimiter=';', decimal=',')
            except Exception as e:
                print(f"Erro ao ler o arquivo {filename}: {e}")
                continue

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
            try:
                filtered_data.to_csv(caminho_processado, index=False, sep=';', decimal=',')
                print(f'Arquivo {filename} salvo com sucesso em: {caminho_processado}')
            except Exception as e:
                print(f"Erro ao salvar o arquivo {filename}: {e}")


# Caminhos dos diretórios
diretorio_limpos = '../data/raw'
diretorio_processados = '../data/processed'

# Processar os arquivos limpos
processar_dados_limpos(diretorio_limpos, diretorio_processados)
