#Carregar, preparar e dividir os dados em conjuntos de treinamento, teste e validação
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Configuração de diretórios para salvar os dados divididos
base_dir = '../data/'
train_dir = os.path.join(base_dir, 'training')
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')

# Criar diretórios se não existirem
for path in [train_dir, test_dir, validation_dir]:
    os.makedirs(path, exist_ok=True)

# Função para carregar e preparar os dados
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path, delimiter=';', decimal=',')
    data['datetime'] = pd.to_datetime(data['Data'] + ' ' + data['Hora'], format='%Y-%m-%d %H:%M:%S')
    data.drop(['Data', 'Hora'], axis=1, inplace=True)
    return data

# Caminho do arquivo de dados original
data_path = '../data/processed/INMET_S_SC_A863_ITUPORANGA_PROCESSADO.CSV'
data = load_and_prepare_data(data_path)

# Dividir dados em treinamento, teste e validação
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)  # 80% treino, 20% temp
test_data, validation_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # 10% teste, 10% validação

# Salvar os dados nos diretórios correspondentes
train_data.to_csv(os.path.join(train_dir, 'training_data.csv'), index=False, sep=';', decimal=',')
test_data.to_csv(os.path.join(test_dir, 'test_data.csv'), index=False, sep=';', decimal=',')
validation_data.to_csv(os.path.join(validation_dir, 'validation_data.csv'), index=False, sep=';', decimal=',')

print("Dados divididos e salvos com sucesso.")
