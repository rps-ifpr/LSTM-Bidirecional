import os
import pandas as pd

def processar_arquivo(caminho_original, caminho_limpo):
    """
    Processa um arquivo CSV do INMET, limpando e salvando os dados em um novo arquivo.
    """
    # Mostrar o caminho absoluto do arquivo original para confirmar a localização
    caminho_absoluto = os.path.abspath(caminho_original)
    print(f"Tentando acessar o arquivo em: {caminho_absoluto}")

    # Verificar se o arquivo existe
    if os.path.exists(caminho_absoluto):
        print("Arquivo encontrado.")
        try:
            # Ler o arquivo, ignorando as primeiras 8 linhas que contêm metadados
            dados = pd.read_csv(caminho_absoluto, encoding='latin1', skiprows=8, sep=';')

            # Renomear as colunas conforme especificado
            colunas_renomeadas = {
                'Data': 'Data',
                'Hora UTC': 'Hora',
                'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'PRECIPITAÇÃO TOTAL',
                'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': 'PRESSAO ATMOSFERICA',
                'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)': 'PRESSÃO ATMOSFERICA MAX (mB)',
                'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)': 'PRESSÃO ATMOSFERICA MIN (mB)',
                'RADIACAO GLOBAL (Kj/m²)': 'RADIACAO GLOBAL (Kj/m²)',
                'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'TEMPERATURA DO AR BULBO SECO (°C)',
                'TEMPERATURA DO PONTO DE ORVALHO (°C)': 'TEMP PONTO DE ORVALHO (°C)',
                'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'TEMP MAX (°C)',
                'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'TEMP MÍN (°C)',
                'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)': 'TEMP ORVALHO MAX (°C)',
                'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)': 'TEMP ORVALHO MIN (°C)',
                'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)': 'UMIDADE REL. MAX (%)',
                'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)': 'UMIDADE REL. MIN (%)',
                'UMIDADE RELATIVA DO AR, HORARIA (%)': 'UMIDADE RELATIVA DO AR (%)',
                'VENTO, DIREÇÃO HORARIA (gr) (° (gr))': 'VENTO DIREÇÃO (gr)°',
                'VENTO, RAJADA MAXIMA (m/s)': 'VENTO RAJADA (m/s)',
                'VENTO, VELOCIDADE HORARIA (m/s)': 'VENTO VELOCIDADE (m/s)'
            }
            dados.rename(columns=colunas_renomeadas, inplace=True)

            # Garantir que a coluna Hora seja string antes de usar .str.replace()
            if 'Hora' in dados.columns:
                dados['Hora'] = dados['Hora'].astype(str).str.replace(' UTC', '')
                # Converter para tipo time
                dados['Hora'] = pd.to_datetime(dados['Hora'], format='%H%M').dt.time

            # Converter a coluna de Data para o tipo datetime
            if 'Data' in dados.columns:
                dados['Data'] = pd.to_datetime(dados['Data'], format='%Y/%m/%d')

            # Convertendo colunas numéricas de string com vírgula decimal para float
            colunas_numericas = [col for col in dados.columns if dados[col].dtype == object]
            for col in colunas_numericas:
                try:
                    dados[col] = pd.to_numeric(dados[col].str.replace(',', '.'), errors='coerce')
                except Exception as e:
                    print(f"Erro ao converter coluna {col} para numérico: {e}")

            # Preencher valores nulos com forward fill
            dados.ffill(inplace=True)  # Uso de ffill() diretamente

            # Salva o dataframe limpo em um novo arquivo CSV
            dados.to_csv(caminho_limpo, index=False, sep=';')

            print("Arquivo limpo salvo com sucesso em:", caminho_limpo)
            print(dados.head())
            print(dados.dtypes)  # Exibir tipos de dados para confirmação

        except Exception as e:
            print(f"Erro ao processar o arquivo: {e}")
    else:
        print("Arquivo não encontrado. Verifique o caminho:", caminho_absoluto)


# Defina o diretório raiz dos seus arquivos de dados
diretorio_raiz = '../data/raw' 

# Obtenha uma lista de todos os arquivos CSV no diretório
arquivos_csv = [f for f in os.listdir(diretorio_raiz) if f.endswith('.CSV')]

# Itere sobre cada arquivo e processe-o
for arquivo in arquivos_csv:
    caminho_original = os.path.join(diretorio_raiz, arquivo)
    caminho_limpo = os.path.join(diretorio_raiz, arquivo.replace('.CSV', '_LIMPO.CSV'))
    processar_arquivo(caminho_original, caminho_limpo)