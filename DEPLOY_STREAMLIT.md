# Deploy no Streamlit Community Cloud

## 1) Pré-requisitos
- Conta no GitHub
- Conta no Streamlit Cloud: https://share.streamlit.io/

## 2) Subir o projeto no GitHub
No diretório do projeto, rode:

```powershell
git init
git add .
git commit -m "App pronto para deploy"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/SEU_REPO.git
git push -u origin main
```

Se o repositório já existe, faça apenas `git add`, `git commit`, `git push`.

## 3) Criar o app no Streamlit Cloud
1. Acesse https://share.streamlit.io/
2. Clique em `New app`
3. Selecione:
   - Repository: seu repositório
   - Branch: `main`
   - Main file path: `portfolio_app.py`
4. Clique em `Deploy`

## 4) Quando o deploy terminar
- O Streamlit vai gerar uma URL pública.
- Compartilhe essa URL com quem vai usar.

## 5) Atualizar versões
Sempre que alterar o código:

```powershell
git add .
git commit -m "Atualização"
git push
```

O Streamlit Cloud redeploya automaticamente.

## Observações
- O app já está configurado para baixar PDF direto no navegador.
- Em Streamlit Cloud, arquivos locais (`cache/`, `reports/`, `carteiras.json`) podem não ser persistentes entre reinícios.
