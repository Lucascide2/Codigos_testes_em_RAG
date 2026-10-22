# API do Ragas 

## 1. Instalação do gerenciador de dependências

Primeiramente é necessário baixar o gerenciador de dependências uv, que pode ser instalado com o comando
abaixo, que deve ser usado no powershell:

`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

## 2. Instalação das Dependências

Para utilizar-se a API, primeiro rode o comando responsável por baixar as dependências necessárias:

`uv sync`

## 3. Inserção da api key.
Após a instalação das dependências, mude o nome do arquivo .env.example para .env e insira corretamente sua
API key do gemini.

## 4. Colocando a API em funcionamento
Por fim, para rodar o programa execute o comando 

`uv run -m evaluation_api`