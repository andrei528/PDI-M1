# Deploy

Este projeto usa `venv` para criar um ambiente virtual isolado e instalar as dependências listadas em `requirements.txt`.
Os comandos abaixo foram pensados para Windows com PowerShell.

## 1. Acesse a pasta do projeto

```powershell
cd "c:\Users\Usuario\caminho-da-pasta"
```

## 2. Crie o ambiente virtual

Se ainda não existir uma pasta `.venv`, crie o ambiente com:

```powershell
python -m venv .venv
```

## 3. Ative o ambiente virtual

```powershell
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a execução de scripts, rode este comando e tente ativar novamente:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 4. Instale as dependencias do `requirements.txt`

Com o ambiente virtual ativado, execute:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Execute o projeto

```powershell
python main.py
```

## 6. Desative o ambiente virtual

Quando terminar:

```powershell
deactivate
```

## Resumo rápido

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```
