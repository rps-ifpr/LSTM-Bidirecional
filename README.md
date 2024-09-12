# Implementação de Redes Neurais Recorrentes (LSTM Bidirecionais) para Gestão de Recursos Hídricos em Agricultura de Precisão

Este repositório contém todos os componentes de um projeto que aplica uma rede neural Long Short-Term Memory (LSTM) Bidirecional para otimizar a irrigação de precisão em cultivos agrícolas, integrando dados de clima e solo. O objetivo é melhorar a eficiência da irrigação através de modelos preditivos avançados e técnicas de aprendizado profundo.

## Estrutura do Diretório

```plaintext
/projeto-lstm-bidirecional-irrigacao/
│
├── .git/                     # Controle de versão Git
├── .venv/                    # Ambiente virtual Python para isolamento de dependências
├── .idea/                    # Configurações do IDE (IntelliJ, PyCharm, etc.)
├── .gitignore                # Especifica arquivos intencionalmente não rastreados para ignorar no controle de versão
├── README.md                 # Documentação inicial, descreve o projeto, como instalar e executar
├── requirements.txt          # Lista de dependências Python para reproduzir o ambiente de desenvolvimento
├── LICENSE                   # Arquivo de licença para o projeto
│
├── data/                     # Dados usados no projeto
│   ├── raw/                  # Dados brutos não modificados
│   └── processed/            # Dados processados e limpos
│
├── src/                      # Códigos-fonte para o projeto
│   ├── main.py               # Script principal para executar o modelo
│   ├── model.py              # Definições do modelo LSTM Bidirecional
│   ├── data_preprocessing.py # Scripts para limpeza e preparação dos dados
│   ├── utilities.py          # Funções auxiliares
│   └── app.py                # Aplicativo Streamlit
│
├── notebooks/                # Jupyter notebooks para análise e demonstração
│   ├── Exploratory_Data_Analysis.ipynb
│   └── Model_Training_and_Evaluation.ipynb
│
├── docs/                     # Documentação adicional para o projeto
│   ├── setup.md
│   └── usage.md
│
└── tests/                    # Testes para assegurar a funcionalidade dos componentes do projeto
    ├── test_data_preprocessing.py
    └── test_model.py
