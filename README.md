# API do Ragas 

## 1. Instalação do gerenciador de dependências

Primeiramente é necessário baixar o gerenciador de dependências uv, que pode ser instalado com o comando
abaixo, que deve ser usado no powershell:

`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

## 2. Configurando e inicializando a API

### 2.1. Instalação das Dependências

Para utilizar-se a API, primeiro abra o diretorio implementacao_ragas no terminal e rode o comando responsável por baixar as dependências necessárias:

`uv sync`

## 2.2. Inserção da api key.
Após a instalação das dependências, mude o nome do arquivo .env.example para .env e insira corretamente sua
API key do gemini.

## 2.3. Colocando a API em funcionamento
Por fim, para rodar o programa execute o comando 

`uv run -m evaluation_api`

## 3. Configurando e inicializando o front da aplicação

### 3.1. Instalação das Dependências

Abra o diretorio ExtensaoV_RAG_Simples no terminal e rode os comandos responsáveis por baixar as dependências necessárias:

`uv pip install requirements.txt`

### 3.2. Inserção da api key.

Mude o nome do arquivo .env.example para .env e insira corretamente sua API key do gemini.

### 3.3. Executando o programa.

Rode o comando abaixo dentro do diretório ExtensaoV_RAG_Simples no terminal:

`uv run streamlit app_generic.py`