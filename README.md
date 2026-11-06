# Experimentos de Cinesiologia

Página web com os experimentos da disciplina de Cinesiologia (LEMOH · UFPA), um por aba:
Planos e eixos, Ombro, Cotovelo, Punho e mão, Joelho, Tornozelo, Cervical, Marcha e Equilíbrio.

Tudo roda no navegador, sem instalar nada e sem MATLAB. Cada aba já abre com os dados de exemplo
e aceita arquivos próprios pelo botão **Carregar arquivo**.

## Link de cada aba

Depois de publicado, o endereço fica `https://SEU-USUARIO.github.io/cinesiologia/` e cada aba tem link próprio:

| Aba | Final do link |
|---|---|
| Planos e eixos | `#planos-e-eixos` |
| Ombro | `#ombro` |
| Cotovelo | `#cotovelo` |
| Punho e mão | `#punho-e-mao` |
| Joelho | `#joelho` |
| Tornozelo | `#tornozelo` |
| Cervical | `#cervical` |
| Marcha | `#marcha` |
| Equilíbrio | `#equilibrio` |

## Formato dos arquivos dos alunos

| Experimento | Arquivo | Colunas |
|---|---|---|
| EMG (ombro, cotovelo, punho, joelho, tornozelo, cervical) | .txt | tempo (s), canal 1, canal 2… (dá para escolher a coluna de cada músculo) |
| Planos e eixos | .csv do app (separado por `;`) | tempo (ms), X, Y, Z |
| Marcha | .txt | tempo (s), Fz (N) |
| Equilíbrio | .mat (`dataToExport` com ML, AP, Tempo) ou .txt/.csv | tempo, ML, AP |

## Estrutura

```
index.html          página com as 9 abas
vendor/             biblioteca de gráficos uPlot (licença MIT)
data/               dados de exemplo de cada experimento
img/                logos LEMOH e UFPA
.nojekyll           faz o GitHub Pages servir os arquivos como estão
```

## Versão Streamlit (`app.py`)

O mesmo conteúdo também roda como app Python no Streamlit, com as 9 abas, upload de arquivos e escalas iguais para comparação.
Arquivos usados pelo Streamlit: `app.py`, `requirements.txt`, `.streamlit/config.toml`, `data/` e `img/`.

Publicar no Streamlit Community Cloud:
1. Suba este repositório no GitHub (público).
2. Entre em https://share.streamlit.io com a conta do GitHub.
3. Clique em **Create app** → **Deploy a public app from GitHub**.
4. Escolha o repositório `cinesiologia`, branch `main`, arquivo principal `app.py`.
5. Clique em **Deploy**. O endereço fica parecido com `https://cinesiologia.streamlit.app`.

Rodar no próprio computador: `pip install -r requirements.txt` e depois `streamlit run app.py`.
