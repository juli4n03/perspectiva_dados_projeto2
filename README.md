# KaBuM — Preço Justo

Projeto da disciplina **Perspectivas de Dados** que estima o preço justo de
componentes de hardware do KaBuM (RAM, CPU, GPU, SSD, fonte e placa-mãe),
com faixa de incerteza (conformal prediction), explicabilidade por
componente (SHAP) e um aplicativo Streamlit para consumir tudo isso: análise
de produto avulso, ranking de ofertas por categoria, compatibilidade de peças
a partir de uma CPU-âncora e montagem de build por orçamento.

O repositório está organizado por **etapa do pipeline**, não por entrega:

- **`01_coleta/`** — web scraper do KaBuM.
- **`02_features/`** — extração de specs via regex a partir do nome do produto.
- **`03_modelagem/`** — do scoring manual original até os modelos de preço
  aprendidos (RandomForest + conformal, RF pooled e TabICL).
- **`04_app/`** — aplicativo Streamlit que consome tudo isso.

O projeto evoluiu em duas fases: a primeira estabeleceu a coleta, a extração
de features e um recomendador com scoring manual; a segunda substituiu o
score manual por modelos de preço aprendidos dos dados, com faixa conformal,
SHAP e o app Streamlit. As duas fases são partes do mesmo trabalho — a
seção 6 (Jornada do projeto) documenta essa evolução em detalhe.

---

## Sumário

1. [Estrutura do repositório](#estrutura-do-repositório)
2. [Contexto e objetivo](#contexto-e-objetivo)
3. [Passo a passo — como rodar tudo](#passo-a-passo--como-rodar-tudo)
4. [Coleta dos dados](#1-coleta-dos-dados)
5. [Feature engineering](#2-feature-engineering)
6. [Modelagem — v2_02 (6 modelos especializados)](#3-modelagem--v2_02-6-modelos-especializados)
7. [Análise metodológica — v2_03 (modelo único e investigação do conformal)](#4-análise-metodológica--v2_03-modelo-único-e-investigação-do-conformal)
8. [Aplicativo Streamlit](#5-aplicativo-streamlit)
9. [Jornada do projeto — tentativas, achados e escolhas](#6-jornada-do-projeto--tentativas-achados-e-escolhas)
10. [Resultados finais e escolhas de arquitetura](#7-resultados-finais-e-escolhas-de-arquitetura)
11. [Limitações conhecidas](#8-limitações-conhecidas)
12. [Trabalho futuro](#9-trabalho-futuro)

---

## Estrutura do repositório

```
perspectiva_dados_projeto2/
├── README.md
├── baixar_dados.py (baixa 00_Dados/ do Drive automaticamente)
├── requirements.txt (dependências Python)
├── 00_Dados/ (coletas do scraper, não versionado — baixe com baixar_dados.py)
│   ├── 2026-06-26/
│   │   ├── kabum_ram_2026-06-26.csv
│   │   ├── kabum_ram_2026-06-26_features.csv
│   │   ├── ...
│   │   └── kabum_todas_pecas_2026-06-26.csv
│   ├── 2026-07-06/
│   └── ...
├── 01_coleta/
│   └── scraper_kabum_pecas.ipynb          ← percorre o KaBuM e grava 00_Dados/
├── 02_features/
│   ├── features.py                        ← módulo único de extração (fonte da verdade)
│   ├── feature_engineering_kabum.ipynb    ← versão original, por data/categoria
│   └── feature_engineering_kabum_todas_datas.ipynb  ← versão em loop (gera *_features.csv)
├── 03_modelagem/
│   ├── recomendacao_kabum.ipynb           ← histórico: recomendador com scoring manual
│   ├── salvar_catalogo.py                 ← gera modelos/catalogo.parquet
│   ├── v2_01_modelo_preco_ram.ipynb       ← histórico: protótipo (só RAM)
│   ├── v2_02_modelo_preco_todas_categorias.ipynb  ← treino dos 6 modelos (o que o app usa)
│   └── v2_03_modelo_tabicl_unico.ipynb    ← análise comparativa (pooled + TabICL + conformal)
├── modelos/                               ← artefatos gerados (compartilhado entre 03 e 04)
│   ├── modelo_preco_ram.joblib
│   ├── modelo_preco_cpu.joblib
│   ├── ...
│   ├── catalogo.parquet
│   └── resumo_metricas.csv
└── 04_app/
    └── app.py                             ← Streamlit
```

---

## Contexto e objetivo

A **primeira fase** do projeto (pastas `01_coleta/`, `02_features/` e o
notebook histórico em `03_modelagem/recomendacao_kabum.ipynb`) implementou:

- Um web scraper (`01_coleta/`) que coleta produtos das seis categorias
  listadas no KaBuM.
- Um pipeline de feature engineering baseado em regex (`02_features/`), para
  extrair atributos técnicos (capacidade, frequência, socket, etc.) do campo
  `nome`.
- Um sistema de recomendação com **scoring manual** — uma fórmula heurística
  que combinava R$/GB, R$/W, avaliações e afins para tentar identificar
  produtos com bom custo-benefício.

A **segunda fase** (o restante de `03_modelagem/` e o app em `04_app/`)
substitui a heurística manual por **modelos de regressão supervisionada**,
um por categoria, treinados sobre os dados coletados em
múltiplas datas. Cada modelo prevê o preço esperado de um produto dado
suas specs, junto com uma **faixa de incerteza de 90%** via conformal
prediction e explicações locais via SHAP. Esses modelos alimentam um app
Streamlit com três abas: análise de produto, ranking de ofertas e
montagem de build.

O objetivo do trabalho é duplo:

1. **Prático**: um app que ajude alguém a decidir se um preço no KaBuM é
   oferta, justo ou caro.
2. **Metodológico**: comparar três arquiteturas de modelagem (seis modelos
   especializados vs RandomForest pooled vs TabICL) e investigar
   empiricamente se conformal prediction "falha" em séries temporais como
   se costuma afirmar.

---

## Passo a passo — como rodar tudo

Esta seção é o **roteiro executável completo**, do zero até o app rodando
no navegador. Requisitos: Windows 10/11, Python 3.11, ~4 GB livres em disco.

### 0. Pré-requisitos
0 Pré requisitos e instalação do UV
Requisitos: Windows 10/11, Python 3.11, ~4 GB livres em disco.

Instalação do UV (se ainda não fez):


# instalar uv
```powershell
 -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Verificar Python e UV:

```powershell
python --version
uv --version
```

### 1. Clonar o repositório e criar ambiente virtual

Abra o PowerShell na pasta onde quer o projeto:

```powershell
# clone
git clone <URL_DO_REPO> perspectiva_dados_projeto2
cd perspectiva_dados_projeto2

# inicializar projeto uv (cria pyproject.toml e .venv do projeto)
uv init

```

Quando o venv estiver ativo, o prompt mostra `(.venv)` no início da linha.

### 2. Instalar dependências

```powershell
uv add -r .\requirements.txt
uv sync
```

Isso instala pandas, scikit-learn, streamlit, shap, joblib, pyarrow, gdown
e os demais pacotes usados. Dura ~2 minutos.


### 3. Baixar os dados do Google Drive

Os dados coletados estão numa pasta pública no Google Drive. Baixe
automaticamente:

```powershell
uv run python ./baixar_dados.py

```

O script vai criar `00_Dados/` e baixar 6 pastas de coleta (2026-06-26 a
2026-07-06), totalizando ~27.000 linhas. Dura ~3-5 minutos dependendo da
conexão.

**Alternativa manual**: se preferir, acesse
[esta pasta no Drive](https://drive.google.com/drive/folders/1ZSOr9PP7XvwfqOHyXkj7JwfS_0cX-uJU?usp=sharing),
baixe tudo e extraia para `00_Dados/` na raiz do projeto.

### 4. Gerar o catálogo consolidado

```powershell
cd 03_modelagem
# executar o script de geração do catálogo via uv
uv run python 03_modelagem/salvar_catalogo.py

```

Isso lê `00_Dados/`, aplica extração de features via `features.py`,
consolida em `modelos/catalogo.parquet`, aplica filtros de sanidade e
marca produtos não-genuínos. Dura ~30 segundos.

Output esperado (final):

```
✓ Catálogo salvo em: ...\modelos\catalogo.parquet
  27.208 linhas
  6 coletas: ['2026-06-26', ...]
  6 categorias:
    ram        : 7.066 produtos
    ...
```

### 5. Treinar os modelos (v2_02)

Abra o Jupyter (recomendo VS Code com a extensão Python):

```powershell
# abrir notebook via uv
uv run jupyter notebook v2_02_modelo_preco_todas_categorias.ipynb
```
e execute todo o arquivo (run all). E apos rodar, salve o arquivo.


Ou clique em "Run All" no VS Code. O notebook:

1. Carrega o `df` a partir de `00_Dados/` (com fallback pros `_features.csv`
   quando não tem `todas_pecas`).
2. Aplica re-extração das features (usando `features.py` atualizado).
3. Treina 6 RandomForest, um por categoria, com conformal normalizado.
4. Salva `modelos/modelo_preco_<cat>.joblib` (6 arquivos) e
   `modelos/resumo_metricas.csv`.

Ou mais rapido ainda, execute:

```powershell
uv run jupyter nbconvert --to notebook --execute --inplace 03_modelagem/v2_02_modelo_preco_todas_categorias.ipynb
```

Dura ~3 minutos numa máquina moderna sem GPU.

**Métricas esperadas** (ver célula final do notebook):

| categoria  | n | MAE (R$) | R² | cobertura | largura média |
|------------|--:|--------:|---:|----------:|--------------:|
| ram        | 1.274 | 269 | 0,87 | 86% | 956 |
| cpu        | 559 | 529 | 0,62 | 96% | 3.384 |
| gpu        | 518 | 1.722 | 0,59 | 88% | 5.350 |
| ssd        | 879 | 683 | 0,37 | 93% | 2.282 |
| fonte      | 627 | 143 | 0,69 | 92% | 803 |
| placa_mae  | 935 | 599 | 0,38 | 89% | 2.223 |

### 6. Rodar o app Streamlit

O `app.py` mora em `04_app/`, uma pasta diferente da usada nos passos
anteriores (`03_modelagem/`). A partir de `03_modelagem/` (onde o passo
anterior deixou você), com o venv ativo:

```powershell
cd ..\04_app
uv run python -m streamlit run ./app.py
```

O navegador abre em `http://localhost:8501`. Se não abrir sozinho, copie
esse endereço no navegador.

O app tem 3 abas:

- **🔍 Analisar produto** — cole nome + preço → veredito com faixa e SHAP.
- **🧩 Peças compatíveis** — ranking de ofertas com filtros, ou compatibilidade
  a partir de uma CPU.
- **🛒 Build por orçamento** — perfil (Gamer / Trabalho / Básico) → build
  completa respeitando compatibilidade.

### 7. (Opcional) Rodar a análise metodológica com TabICL (v2_03)

O notebook `v2_03_modelo_tabicl_unico.ipynb` faz a comparação
metodológica descrita na seção 4. Ele usa o RF pooled como baseline
sempre. Se quiser incluir o TabICL, precisa de GPU NVIDIA com CUDA.

**Sem GPU** — abra e "Run All". A célula do TabICL cai em `try/except`
e imprime "TabICL não disponível" — o resto do notebook roda normalmente
com o RF pooled.

**Com GPU NVIDIA** — instale PyTorch com CUDA e TabICL:

```powershell
# use o Python específico do seu ambiente
# (descubra com: python -c "import sys; print(sys.executable)")
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install tabicl
```

Confirme:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
# Esperado: True
```

Aí sim "Run All" no notebook. Ele detecta a GPU automaticamente e treina
o TabICL com contexto de 3000 linhas. Dura ~5 minutos numa RTX 4060.

### Resumo dos comandos (referência rápida)

```powershell
# do zero
git clone <URL> && cd perspectiva_dados_projeto2
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python baixar_dados.py

# gerar catálogo e treinar
cd 03_modelagem
python salvar_catalogo.py
jupyter notebook v2_02_modelo_preco_todas_categorias.ipynb   # Run All

# rodar app
cd ..\04_app
streamlit run app.py
```

---

## 1. Coleta dos dados

O scraper (`01_coleta/scraper_kabum_pecas.ipynb`)
percorre as páginas de categoria do KaBuM e extrai, para cada produto:

- Identificador (`id`) e nome
- Preço original, preço atual e preço à vista (PIX)
- Desconto percentual
- Avaliação e número de avaliações
- Disponibilidade
- Fabricante (quando o site expõe)
- Categoria (chave interna: `ram`, `cpu`, `gpu`, `ssd`, `fonte`, `placa_mae`)

Cada execução do scraper gera uma nova pasta em `00_Dados/<data>/`
contendo um CSV por categoria (`kabum_<cat>_<data>.csv`) e um CSV
consolidado (`kabum_todas_pecas_<data>.csv`).

Para o v2, o dataset final tem **6 coletas** entre 2026-06-26 e 2026-07-06,
totalizando ~27 mil linhas antes dos filtros.

### Estabilidade temporal do dataset

Diagnóstico rodado no notebook v2_03 (célula 7):

| Coleta → Coleta          | Produtos em ambas | \|Δ%\| mediano | \|Δ%\| médio |
|--------------------------|------------------:|---------------:|-------------:|
| 2026-06-26 → 2026-06-28  |             3.860 |          0,00% |        0,47% |
| 2026-06-28 → 2026-06-30  |             3.835 |          0,00% |        0,65% |
| 2026-06-30 → 2026-07-01  |             3.780 |          0,00% |        1,73% |
| 2026-07-01 → 2026-07-02  |             3.825 |          0,00% |        0,79% |
| 2026-07-02 → 2026-07-06  |             3.653 |          0,00% |        1,33% |

**A série é praticamente estacionária no período coletado.** A mediana da
variação absoluta de preço entre coletas consecutivas é sempre 0%. Esse
diagnóstico é importante para a interpretação da análise do conformal
(seção 4).

---

## 2. Feature engineering

Existem **dois lugares** onde a extração acontece, e é intencional que os
dois convivam:

- `02_features/feature_engineering_kabum.ipynb`: notebook original que gera
  os arquivos `_features.csv` por categoria dentro de cada pasta de coleta.
- `02_features/features.py`: módulo Python importável que replica e estende
  a mesma lógica. É a **fonte da verdade** para o app e para o script
  `salvar_catalogo.py`, e é reaplicado antes do treino em `v2_02`.

### Por que dois lugares?

O notebook original escrevia specs em CSVs (`*_features.csv`). Com o tempo,
algumas dessas regex ficaram desatualizadas — por exemplo, o regex de
`mobo_chipset` não capturava "B550M-A" corretamente, e a tabela `GPU_TDP`
cobria apenas ~25 modelos. Manter o `features.py` como um módulo Python permite:

1. Testar a extração em isolamento (`extrair_features_produto(nome, categoria)`).
2. Aplicar a extração no app Streamlit em tempo real (usuário cola um nome
   e o app extrai as specs).
3. **Reaplicar sobre os `_features.csv` antigos** — quando uma regex é
   melhorada, o notebook v2_02 reprocessa antes de treinar, sem precisar
   re-rodar o v1.

### O que `features.py` extrai

Uma tabela resumida das features extraídas por categoria (via regex sobre
`nome`):

- **RAM**: `ram_geracao` (DDR3/DDR4/DDR5), `ram_gb`, `ram_mhz`, `ram_cl`,
  `ram_notebook`.
- **CPU**: `cpu_marca`, `cpu_socket`, `cpu_serie` (i3/i5/i7/i9/Ryzen 3/5/7/9),
  `cpu_tdp_w`, `cpu_ddr_suportado` (via tabela `SOCKET_DDR`), `cpu_com_cooler`,
  `cpu_cores`, `cpu_threads`, `cpu_clock_ghz`.
- **GPU**: `gpu_marca_chip` (NVIDIA/AMD/Intel), `gpu_modelo`, `gpu_vram_gb`,
  `gpu_tdp_w` (via tabela `GPU_TDP` cobrindo ~70 modelos de consumo).
- **SSD**: `ssd_interface` (NVMe/SATA), `ssd_geracao_pcie` (3.0/4.0/5.0),
  `ssd_capacidade_gb`, `ssd_leitura_mbs`, `ssd_notebook`.
- **Fonte**: `fonte_wattagem`, `fonte_certificacao` (80 Plus Bronze/Gold/etc.),
  `fonte_modular`, `fonte_atx3`.
- **Placa-mãe**: `mobo_socket`, `mobo_chipset`, `mobo_ddr`, `mobo_form_factor`
  (ATX/mATX/ITX), `mobo_slots_m2`, `mobo_max_ram_gb`.

### Tabelas de fallback dentro do `features.py`

Três dicionários fazem trabalho pesado quando a regex não pega:

- **`SOCKET_DDR`**: mapeia socket → geração DDR suportada. Ex: `AM5 → DDR5`,
  `LGA1700 → DDR4/DDR5`. Usado para preencher `cpu_ddr_suportado` a partir
  do socket.
- **`MODELO_SOCKET_CPU`**: fallback para inferir o socket a partir do modelo
  da CPU. O KaBuM raramente coloca "LGA1700" no nome de um "Core i3-14100F"
  — a tabela cobre esse caso com regex `i[3579]-14\d{3} → LGA1700`.
  Cobre Intel 6ª–14ª geração, Ryzen 1000–9000, Threadripper e EPYC.
- **`GPU_TDP`**: modelo → TDP em Watts. Cobre RTX 20/30/40/50, GTX 10/16/900,
  GT low-end, RX 400/500/5000/6000/7000/9000 e Intel Arc A/B. Usado
  tanto para preencher `gpu_tdp_w` quanto para dimensionar fontes na
  aba de compatibilidade do app.

### Consolidação: `salvar_catalogo.py`

Este script lê todas as pastas de `00_Dados/`, faz merge das colunas de
specs com o consolidado `kabum_todas_pecas_<data>.csv` e escreve
`modelos/catalogo.parquet` — arquivo único que alimenta o app. Se a pasta
não tem o `todas_pecas` (por exemplo coletas mais antigas onde só sobrou
os `_features.csv`), o script cai num fluxo alternativo que concatena
os arquivos por categoria.

Além disso, aplica duas rodadas de limpeza que o app consome:

1. **Sanidade numérica**: valores absurdos viram NaN. Ex.: RAM ≤ 256 GB,
   VRAM de GPU ≤ 48 GB, capacidade de SSD entre 32 GB e 8 TB. Isso pega
   erros de parsing tipo "94208 GB" (que é um código de produto capturado
   pelo regex, não capacidade real).
2. **Produtos não-genuínos**: coluna `eh_genuino` marca `False` para itens
   que o KaBuM classifica junto mas não são a peça-alvo — adaptadores
   ("Adaptador de Memória Ctech DDR4" que não é uma RAM), suportes
   ("Suporte para Placa de Vídeo" que não é uma GPU), GPUs profissionais
   (Nvidia Quadro/H100/A100, AMD Instinct), SSDs enterprise (DC600M,
   PM893), etc. Cada categoria tem seu regex de exclusão em
   `salvar_catalogo.py`.

---

## 3. Modelagem — v2_02 (6 modelos especializados)

**Notebook**: `v2_02_modelo_preco_todas_categorias.ipynb`

### Arquitetura

Um `RandomForestRegressor` por categoria, cada um treinado com features
específicas daquela categoria. Cada bundle salvo em
`modelos/modelo_preco_<categoria>.joblib` contém:

- `preproc`: `ColumnTransformer` com `SimpleImputer` (mediana em
  numéricas, "desconhecido" em categóricas) e `OneHotEncoder`
  (`min_frequency=10`, `handle_unknown="ignore"`).
- `modelo_media`: RF que prevê o preço (n_estimators=300, min_samples_leaf=5).
- `modelo_variancia`: RF secundário que prevê o resíduo ao quadrado —
  usado para conformal normalizado.
- `q_norm`: quantil de conformidade calibrado no conjunto de calibração.
- `metricas`: MAE, R², cobertura, largura média.

### Split

60% treino / 20% calibração / 20% teste, com `random_state=42` fixo.
O conjunto de calibração é separado do treino explicitamente para o
conformal (não pode ser reutilizado, ou a garantia de cobertura marginal
se perde).

### Conformal prediction normalizado

Fluxo (para cada categoria):

1. Treina `modelo_media` sobre `(X_treino, y_treino)`.
2. Calcula resíduos no treino: `r = y_treino - modelo_media.predict(X_treino)`.
3. Treina `modelo_variancia` sobre `(X_treino, r²)`.
4. No conjunto de calibração:
   - Prevê média e variância.
   - Score não-conforme normalizado:
     `s_i = |y_cal_i - pred_i| / sqrt(var_i)`.
   - `q_norm` = quantil de `(n+1)(1-α)/n` de `s` (com `α = 0.1` → 90% de
     cobertura alvo).
5. Em produção, a faixa para uma nova amostra é:
   `[pred ± q_norm · sqrt(var)]`, clipada em zero (preço não pode ser
   negativo).

A **normalização por variância local** deixa a faixa mais estreita em
regiões do espaço de features onde o modelo é confiante (RAMs comuns) e
mais larga onde é incerto (GPUs raras/exóticas), sem sacrificar a
garantia marginal de cobertura.

### Filtro de qualidade no treino

Antes do split, o notebook aplica:

- Remoção de duplicatas por nome.
- Exclusão de produtos não-genuínos por regex (idêntico ao usado em
  `salvar_catalogo.py`).
- Sanidade numérica (ram_mhz ≤ 9000, gpu_vram_gb ≤ 48, etc.).

### Métricas finais (v2_02, sem log-transform, 6 categorias)

| categoria  | n produtos |    MAE |   R² | cobertura | largura média |
|------------|-----------:|-------:|-----:|----------:|--------------:|
| ram        |      1.274 |    269 | 0,87 |       86% |           956 |
| cpu        |        559 |    529 | 0,62 |       96% |         3.384 |
| gpu        |        518 |  1.722 | 0,59 |       88% |         5.350 |
| ssd        |        879 |    683 | 0,37 |       93% |         2.282 |
| fonte      |        627 |    143 | 0,69 |       92% |           803 |
| placa_mae  |        935 |    599 | 0,38 |       89% |         2.223 |

Cobertura empírica próxima do alvo de 90% em todas as categorias —
conformal calibrado. R² varia bastante entre categorias: RAM tem
distribuição bem-comportada e features de alto poder explicativo (`ram_gb`,
`ram_mhz`, `ram_geracao` explicam quase tudo). Placa-mãe e SSD são as
mais difíceis: nomes têm muito ruído e specs faltam frequentemente.

---

## 4. Análise metodológica — v2_03 (modelo único e investigação do conformal)

**Notebook**: `v2_03_modelo_tabicl_unico.ipynb`

O v2_02 encapsula uma escolha de arquitetura: **um modelo por categoria**.
Isso é ok, mas não a única alternativa. Este notebook complementar responde
duas perguntas metodológicas:

### 4.1. Modelo único vs modelos especializados

Em vez de seis RFs, treinamos:

- **RF pooled**: um único RandomForest sobre uma tabela larga esparsa
  (todas as categorias empilhadas, `categoria_key` como feature, alvo em
  `log(preço)` para lidar com a escala 22 → 83.400 R$).
- **TabICL**: transformer para dados tabulares (in-context learning) com
  quantis nativos — sem precisar de conformal por cima. Rodado com contexto
  de 3.000 linhas em GPU (RTX 4060).

### Comparação empírica (R² por categoria)

| categoria  | 6 especialistas | RF pooled  | TabICL  |
|------------|----------------:|-----------:|--------:|
| ram        |            0,87 |   **0,92** |   0,90  |
| cpu        |        **0,62** |       0,52 |   0,26  |
| gpu        |            0,59 |       0,64 | **0,90**|
| ssd        |            0,37 |       0,75 | **0,79**|
| fonte      |            0,69 |   **0,76** |   0,78  |
| placa_mae  |            0,38 |   **0,62** |   0,56  |

**Vencedores por categoria**: RF pooled 4× | 6 especialistas 1× (CPU) |
TabICL 2× (GPU e SSD, com fonte em empate técnico).

**Métricas gerais TabICL**: MAE R$ 390, R² 0,836, cobertura da faixa
nativa 89,7% (praticamente calibrada em 90% sem conformal por cima),
largura média R$ 1.525.

### O que essa comparação revela

O modelo **único com log-transform** melhora dramaticamente as categorias
mais fracas do v2_02: SSD sobe de R² 0,37 para 0,75, placa-mãe de 0,38
para 0,62. É uma diferença de duas categorias que valem a pena.

O **TabICL rivaliza com o RF pooled** no agregado (R² 0,84 vs 0,70),
com cobertura calibrada em 89,7% **sem precisar de conformal por cima**.
Impressiona especialmente em GPU (R² 0,90), onde é o líder por larga
margem. A subamostragem para 3.000 linhas de contexto (das 14.025
disponíveis) provavelmente ainda limita seu desempenho em CPU — só
480 exemplos ficam disponíveis para uma categoria com features
específicas que aparecem apenas ali.

### 4.2. Conformal em série temporal — realmente falha?

A hipótese clássica é que série temporal **viola exchangeability** e
portanto o conformal perde sua garantia teórica de cobertura. Testamos
empiricamente com dois splits:

| Split       | Cobertura | Largura média |
|-------------|----------:|--------------:|
| Aleatório   |     88,8% |         1.248 |
| Temporal *  |     89,7% |         1.294 |

\* Treino/calibração nas 5 coletas mais antigas, teste na mais recente.

**A diferença é de 0,9 pontos percentuais — dentro do ruído de amostragem.**
O conformal continuou válido no split temporal. Isso é coerente com o
diagnóstico da seção 1: a série é praticamente estacionária.

### Teste de estresse — quando o conformal quebra?

Aplicamos choques sintéticos no conjunto de teste:

| Choque                              | Cobertura |
|-------------------------------------|----------:|
| Baseline (sem choque)               |     88,8% |
| Uniforme +12% (todos os preços)     |     76,7% |
| Uniforme −12% (todos os preços)     |     75,5% |
| Heterogêneo (40% caem 20–50%, tipo BF) | 68,8%  |

Sob choques de magnitude relevante, o conformal degrada substancialmente
(12–20 pontos percentuais). O **choque heterogêneo (Black Friday)** é
o pior cenário.

### Conclusão metodológica precisa

> Conformal split não falha por **ordenação temporal per se**. Falha por
> **distribution shift substancial**. Em séries quase estacionárias
> (como a coletada aqui, |Δ%| mediano de 0% entre coletas consecutivas),
> exchangeability é aproximadamente preservada e o conformal continua
> calibrado. O que quebra a garantia é a magnitude do desvio de
> distribuição em relação à largura do intervalo — não o eixo temporal
> em si.

Essa é uma versão mais precisa da frase de manual "conformal falha em
série temporal", e é sustentada empiricamente pelos números acima.

---

## 5. Aplicativo Streamlit

**Rodar** (a partir da raiz do repositório):

```powershell
cd 04_app
streamlit run app.py
```

O app carrega:

- Os seis bundles `modelos/modelo_preco_<cat>.joblib` (v2_02).
- O catálogo consolidado `modelos/catalogo.parquet` (gerado por
  `salvar_catalogo.py`).

Se o catálogo não existir, as abas 2 e 3 exibem uma instrução para
gerá-lo. A aba 1 funciona só com os bundles.

### Aba 1 — Analisar produto

Fluxo: usuário cola nome do produto, informa preço e escolhe a categoria.
O app extrai as specs via regex (com botão "🪄 Extrair specs do nome"
ou automaticamente ao clicar em "Analisar"), permite ao usuário corrigir
manualmente, e retorna:

- **Preço justo previsto** (ponto central do modelo).
- **Faixa de 90%** (conformal normalizado).
- **Veredito colorido**: 🎉 Oferta (abaixo da faixa), ✅ Justo (dentro),
  ⚠️ Caro (acima).
- **Waterfall SHAP** explicando quais features empurraram o preço para
  cima ou para baixo.

Detalhe importante de UX: se o usuário deixar `fabricante` em branco,
aparece um aviso amarelo explicando que o modelo aprendeu que produtos
sem fabricante identificado tendem a ser kits enterprise/obscuros e mais
caros. Isso evita a interpretação errada de "oferta espetacular" quando
na verdade é só viés do imputer.

### Aba 2 — Peças compatíveis

Dois modos:

**Ranking de ofertas** — Escolhe categoria, aplica filtros (preço, veredito,
data de coleta, specs numéricas com slider, categóricas com multiselect),
e mostra o catálogo ordenado por `desvio_pct` (maiores ofertas primeiro).
Dois toggles ficam ativos por padrão: "Excluir não-genuínos" (usa `eh_genuino`)
e "Só coleta mais recente" (deduplicando por `id`, mantendo a última data).

**Compatibilidade a partir de uma CPU** — Usuário escolhe uma CPU-âncora
(via filtros de marca, socket, busca por nome). O app deriva:

- **Placas-mãe** com mesmo socket (usa `MODELO_SOCKET_CPU` como fallback
  quando o socket não vem no nome da CPU).
- **RAM** com DDR compatível (mapa `SOCKET_DDR` da CPU).
- **Fontes** com wattagem ≥ TDP(CPU) + TDP(GPU) + 150 W de margem.
- **GPUs** (livre — PCIe é universal).

Cada seção mostra top-10 por oferta e a conta da fonte é exibida no título
("⚡ Fontes ≥ 455W  (CPU 105W + GPU 200W + margem 150W)").

### Aba 3 — Build por orçamento

Usuário informa orçamento total, escolhe um perfil ("Gamer", "Trabalho",
"Básico") — que aloca percentuais fixos por categoria — e opcionalmente
edita os percentuais em um expander.

O app monta a build **em cascata respeitando compatibilidade**:

1. **CPU** primeiro (define socket e DDR).
2. **Placa-mãe** com mesmo socket.
3. **RAM** com DDR compatível.
4. **GPU** livre.
5. **SSD** livre.
6. **Fonte** com wattagem suficiente.

Dentro de cada teto orçamentário, o app usa a **estratégia "melhor
caro-mas-justo"**: se houver peças com veredito "✅ Justo" dentro do teto,
escolhe a mais cara (assume "mais caro = melhor especificação"). Se não
houver, cai para a maior oferta (`desvio_pct` mais negativo).

Um log expansível de decisões mostra passo a passo, incluindo avisos
quando socket ou DDR não são identificados (nesses casos o filtro é
suspenso e a compatibilidade não é garantida — o app **avisa
explicitamente em vez de esconder** o problema).

---

## 6. Jornada do projeto — tentativas, achados e escolhas

Esta seção documenta **o caminho real** que o projeto percorreu, não só
o estado final. As decisões finais só fazem sentido à luz das tentativas
que fracassaram no meio do caminho.

### Narrativa geral

Começamos com uma proposta simples: **substituir o score manual do v1
por um modelo aprendido**. A hipótese inicial era "um modelo por
categoria, RandomForest, com conformal por cima para dar faixa de
incerteza".

Isso funcionou como MVP — os R² iniciais foram razoáveis (RAM 0,90 no
começo, com apenas uma coleta). Mas conforme fomos coletando mais dados
e polindo, começamos a descobrir que:

1. **A extração de features estava frágil** em muitos lugares. O regex
   de chipset da placa-mãe não pegava "B550M-A", a tabela de TDP de GPU
   cobria só 25 modelos dos ~70 relevantes, o socket da CPU frequentemente
   não vinha no nome (o site presume que quem compra um "i3-14100F" sabe
   que é LGA1700).
2. **Produtos "não-alvo" contaminavam o modelo**. O KaBuM classifica
   como "GPU" tanto uma RTX 5090 quanto um suporte de placa de vídeo por
   R$ 30. Como o modelo achava que "GPU" custava em média R$ 4.000, o
   suporte virava "oferta de -99%" no ranking.
3. **A distribuição de preço tem cauda muito longa**. GPUs vão de R$ 200
   a R$ 25.000. Isso quebra tanto o R² absoluto quanto a calibração do
   conformal (a faixa fica larga demais para produtos baratos e apertada
   demais para caros).
4. **Não estávamos alinhados com a orientação metodológica original.**
   No planejamento inicial, considerou-se um **modelo único** com
   `categoria_key` como feature e alvo em `log(preço)`, usando TabICL
   e uma investigação empírica do conformal em série temporal. Fomos
   descobrindo aos poucos que essa era uma alternativa válida (e talvez
   melhor).

A partir desses achados, o projeto se desdobrou em duas linhas paralelas:

- **v2_02** — a arquitetura de 6 modelos especializados que foi
  polida iterativamente com filtros de qualidade, extração de features
  melhorada e sanidade. Este é o modelo que o app consome.
- **v2_03** — a análise metodológica que compara essa arquitetura com
  as alternativas do plano original (RF pooled e TabICL), e investiga
  empiricamente a validade do conformal no dataset temporal coletado.

### Tópicos detalhados

#### T1. Filtros de qualidade — o mais importante para R²

**Motivação**: rankings iniciais tinham "🎉 Oferta -99%" com adaptadores,
suportes e GPUs profissionais aparecendo no topo. Não eram ofertas —
eram categorização errada do KaBuM sendo interpretada literalmente pelo
modelo.

**Tentativas**:

1. Filtro conservador (só "adaptador", "cooler", "case") — resolveu ~30%
   dos casos.
2. Filtro estendido cobrindo RAMs de servidor (Dell PowerEdge, HP DL160,
   ECC, RDIMM, PC2/PC3L legados), GPUs profissionais (Quadro, Tesla,
   H100, A100, RTX A/Ada, Instinct), SSDs enterprise (DC600M, PM893,
   SAS) — resolveu ~90%.
3. Sanidade numérica (ram_gb ≤ 256, gpu_vram_gb ≤ 48, ssd_capacidade_gb
   ∈ [32, 8192]) — pegou os últimos casos, como uma "GPU com 94208 GB
   de VRAM" que era código de produto sendo confundido pelo regex.

**Achado**: essa etapa sozinha rendeu **mais melhoria de R² e MAE do que
qualquer mudança de modelo**. É contra-intuitivo mas real: no mundo real,
80% do valor vem de dados limpos.

#### T2. Extração de features — expandindo os fallbacks

**Motivação**: cobertura de `gpu_modelo` estava em 57%, `gpu_tdp_w` em
33%. Muitas GPUs eram vistas pelo modelo como "genéricas sem feature",
o que fazia o modelo virar quase que um baseline de fabricante.

**Tentativas**:

1. Expandir `GPU_TDP` de 25 para ~70 modelos, cobrindo RTX 20/30/40/50,
   GTX 10/16/900, GT low-end, RX 400/500/5000/6000/7000/9000 e Intel Arc.
2. Melhorar `normalizar_gpu_modelo` para colapsar "RTX 5060 Ti" e
   "RTX5060ti" no mesmo texto, separar sufixos compostos "TISUPER" em
   "TI SUPER".
3. Criar `MODELO_SOCKET_CPU` como fallback: quando o socket não aparece
   no nome da CPU, deriva pelo modelo. Ex: `i[3579]-14\d{3}` → LGA1700.

**Achado**: cobertura de `gpu_modelo` foi para 84%, `gpu_tdp_w` para 78%.
R² da GPU subiu de 0,46 para 0,59. Ganho concreto direto de expandir
tabelas.

#### T3. Log-transform — funcionou parcialmente

**Motivação**: preços com cauda pesada (R$ 22 a R$ 83.400) fazem o modelo
otimizar a MSE às custas dos preços baixos. Log-transform (`y = log1p(preco)`)
uniformiza as escalas.

**Tentativas**:

1. Aplicar log em **todas** as 6 categorias no v2_02.

**Achado — misto**:

| categoria  | Δ R² com log |
|------------|-------------:|
| cpu        |    **+0,10** |
| fonte      |    **+0,06** |
| ram        |        −0,06 |
| gpu        |        −0,27 |
| ssd        |        +0,03 |
| placa_mae  |        −0,12 |

Log **ajuda quando cada categoria tem sub-caudas próprias** (CPU tem
Ryzen 9 X3D a R$ 4.500 e Athlon a R$ 200; fonte tem 1200W Gold a
R$ 1.500 e 400W a R$ 150). **Prejudica quando a distribuição é
suficientemente uniforme** (RAM DDR4 gira entre R$ 100 e R$ 1.500;
GPU teve o pior efeito, provavelmente porque a cauda foi contaminada
por GPUs profissionais que só filtramos depois).

**Decisão**: reverter para escala linear no v2_02. Log só entra no
v2_03 (modelo pooled), onde a escala geral é maior (R$ 22 → R$ 83.400)
e o benefício aparece de verdade.

#### T4. Modelo único (RF pooled) — vitória com ressalvas

**Motivação**: seguir o plano metodológico original. Uma tabela larga
esparsa, `categoria_key` como feature, `log(preço)` como alvo.

**Tentativas**:

1. Empilhar todas as categorias em uma matriz única com 276 features
   após one-hot.
2. Treinar RandomForest com 300 árvores.

**Achado**:

| categoria  | R² v2_02 (6 esp) | R² pooled | Δ |
|------------|-----------------:|----------:|---:|
| ram        |             0,87 |      0,92 | +0,05 |
| cpu        |             0,62 |      0,52 | −0,10 |
| gpu        |             0,59 |      0,64 | +0,05 |
| ssd        |             0,37 |      0,75 | **+0,38** |
| fonte      |             0,69 |      0,76 | +0,07 |
| placa_mae  |             0,38 |      0,62 | **+0,24** |

**Vitória em 5 de 6 categorias.** As duas maiores melhorias
(SSD +0,38, placa-mãe +0,24) são justamente as categorias mais fracas
do v2_02. Só CPU regrediu (provavelmente porque tem features únicas
que ficam "diluídas" com features das outras categorias).

**Ressalva**: o app **não foi migrado** para o pooled. Ver seção 7.1
para o porquê.

#### T5. TabICL — funcionou depois de vencer obstáculos

**Motivação**: TabICL retorna quantis nativos (sem precisar de conformal
por cima). Se rivalizar com o RF pooled, seria a solução mais elegante.

**Obstáculos vencidos**:

1. **PyTorch CPU-only não roda TabICL em tempo aceitável** — primeira
   tentativa em CPU levou 12 minutos para 5 lotes de 100 (estimativa
   total: 2 horas).
2. **VRAM da RTX 4060 é 8 GB, apertado para TabICL**. Contexto de
   5000 estourou memória. Reduzimos para 3000 (que ficou bom) e
   depois testamos 4000 (também ok).
3. **Dois Pythons diferentes no sistema** — o `pip install torch --index-url
   .../cu121` foi instalado em um Python que não era o do notebook.
   Precisou instalar explicitamente no `sys.executable` correto.

**Achado**:

| categoria  | R² pooled | R² TabICL@3000 |
|------------|----------:|---------------:|
| ram        |      0,92 |           0,90 |
| cpu        |      0,52 |           0,26 |
| gpu        |      0,64 |       **0,90** |
| ssd        |      0,75 |       **0,79** |
| fonte      |      0,76 |       **0,78** |
| placa_mae  |      0,62 |           0,56 |

**Cobertura da faixa nativa do TabICL: 89,7%** (alvo 90%) — praticamente
calibrada, sem precisar de conformal.

**Ressalva**: performance ruim em CPU (0,26) provavelmente porque
subamostragem para contexto de 3000 deixou apenas ~480 CPUs no contexto,
e CPU tem muitas features específicas (`cpu_cores`, `cpu_socket`, etc.)
que ficam mal representadas com poucos exemplos.

#### T6. Investigação do conformal — a hipótese "clássica" não se sustentou

**Motivação**: era hipótese planejada que "conformal falha em série
temporal porque viola exchangeability". Testar empiricamente.

**Tentativas**:

1. Split aleatório (baseline). Cobertura: 88,8%.
2. Split temporal (treino/cal nas 5 coletas antigas, teste na mais
   recente). Cobertura: 89,7%.
3. Testes de estresse com choque sintético (uniforme +/-12%, heterogêneo
   estilo Black Friday).

**Achado**: a diferença aleatório vs temporal foi de 0,9 pp — dentro
do ruído. **Conformal NÃO falhou.** O motivo é que a série coletada é
**praticamente estacionária** (|Δ%| mediano de 0% entre coletas).

Sob choque real (heterogêneo Black Friday), a cobertura cai para 68,8%.
Ou seja: **o que quebra o conformal é magnitude do distribution shift,
não presença de tempo per se**.

Essa é uma **versão mais precisa** da frase de manual, e é sustentada
empiricamente. É o achado metodológico mais forte do trabalho.

#### T7. UX e tratamento de casos difíceis no app

**Motivação**: o app tinha comportamentos "silenciosos" que davam
respostas erradas com aparência de certas.

**Casos tratados**:

- **Auto-extração de features**: se o usuário digitou nome + preço mas
  não clicou em "Extrair specs", o botão "Analisar" agora **extrai
  automaticamente** e popula o formulário. Antes, ele passava specs
  vazias para o modelo, que dava um preço justo aleatório baseado na
  mediana do treino.
- **Aviso de fabricante ausente**: modelo aprendeu que produtos sem
  fabricante identificado são mais caros. Se o usuário deixa em branco,
  aparece aviso amarelo explicando o viés (senão, "oferta de -50%"
  aparecia por artefato de imputação).
- **Log de decisões na aba de build**: quando socket ou DDR da CPU não
  são identificados, o app avisa "compatibilidade não garantida" em
  vez de silenciosamente escolher qualquer placa-mãe. Aconteceu com
  o i3-14100F no primeiro teste — o socket LGA1700 não estava no nome
  e o app pareou com uma placa AM4 até adicionarmos o
  `MODELO_SOCKET_CPU`.

---

## 7. Resultados finais e escolhas de arquitetura

### 7.1. Por que o app usa 6 modelos especializados (e não RF pooled ou TabICL)?

O v2_03 mostra que **RF pooled com log-transform** é melhor que os 6
especialistas em 4 de 6 categorias, e que **TabICL** vence em GPU (0,90 vs
0,59) e SSD (0,79 vs 0,37). Mesmo assim, o app usa os 6 especialistas.
Razões:

1. **Cronograma**. A aplicação e sua infraestrutura (features, extração,
   filtros, cache, SHAP) foram construídas sobre os 6 modelos
   especializados. Trocar a arquitetura para pooled ou TabICL implicaria
   refactor do app (um único bundle, transformações em log destransformadas
   na inferência, TabICL exige GPU para inferência em tempo aceitável).
2. **SHAP**. O TreeExplainer do SHAP funciona bem em árvores, mal em
   transformers. Trocar para TabICL implicaria abrir mão do waterfall
   de explicação por produto — que é uma das partes mais valorizadas do
   app (aba 1).
3. **Portabilidade**. Um bundle `.joblib` por categoria é ~5 MB e
   inferência é milissegundos em CPU. TabICL exige PyTorch com CUDA
   para inferência rápida — implantar isso em qualquer máquina é caro.

**Portanto**: o v2_02 continua sendo o modelo em produção. O v2_03 é uma
**análise metodológica** documentada, e a recomendação de "trabalho
futuro" é migrar para pooled com log-transform (não TabICL, pelas razões
acima).

### 7.2. Filtros conservadores de não-genuínos

Documentado em `salvar_catalogo.py`. Foram descobertos empiricamente
inspecionando os rankings iniciais do modelo. Padrões que causaram
mais dano no treino inicial:

- **RAM**: SSDs de servidor Dell/HP (PowerEdge, Precision, DL160, PC2-,
  PC3L-), memórias ECC/registered/rdimm, kits em MB ao invés de GB.
- **GPU**: suportes/cabos, GPUs profissionais (Quadro, Tesla, H100/H200,
  RTX A/Ada), GPUs muito antigas (G210, GT710/730/1030, Radeon
  HD/R5/R7/R9).
- **SSD**: adaptadores, cases, gavetas, DC600M/PM893/Micron enterprise,
  drives SAS, SSDs "para servidor".

Isso remove ~10–15% do dataset bruto, mas melhora significativamente a
qualidade das previsões porque o modelo passa a aprender só sobre a
categoria "de consumidor" que o app se propõe a atender.

---

## 8. Limitações conhecidas

**Viés do fabricante ausente.** Quando `fabricante` vem vazio no dataset
(o site não classifica), o `SimpleImputer` substitui por "desconhecido".
No treino, produtos com fabricante desconhecido são majoritariamente
kits enterprise ou obscuros e mais caros. O modelo aprende essa
correlação. Consequência: no app, se o usuário não preencher fabricante,
a previsão vem sistematicamente inflacionada. Mitigação: aviso amarelo
explícito na aba 1 quando fabricante está vazio.

**R² baixo em placa-mãe e SSD no v2_02.** 0,38 e 0,37 respectivamente.
Placa-mãe é intrinsecamente difícil (nomes têm muito ruído, chipset é
opcional, features caras como VRM não estão no nome). SSD idem — muitos
sem `ssd_geracao_pcie` extraível. Modelo pooled resolve isso (0,62 e
0,75), mas por escolha de arquitetura o app usa os 6 especialistas.

**Poucas coletas (6 datas em ~10 dias).** O dataset é curto. A
investigação do conformal em série temporal é limitada pelo próprio
horizonte — não observamos Black Friday, lançamentos, choques de câmbio.
O teste de estresse sintético compensa parcialmente, mas seria melhor
com dados reais de mudança de regime.

**Extração via regex é frágil.** O `features.py` faz o melhor esforço,
mas nomes idiossincráticos escapam. Cobertura de `gpu_modelo` era 57%
antes de expandir a tabela `GPU_TDP` (foi para 84%). Ainda existem
casos edge: RAMs com fabricante embutido no nome, CPUs de socket
antigo (LGA1150, LGA1155), fontes ATX 3.1 vs ATX 3.0.

**GPU baseline (R² 0,59) mesmo após polimento.** As GPUs têm cauda
muito pesada (RTX 5090 a R$ 25.000, GT 710 a R$ 200). Mesmo com o
`GPU_TDP` estendido e filtros de GPU profissional, o RF especialista
ainda não modela isso bem em R$ absoluto. É outra motivação forte para
o pooled com log-transform como próximo passo.

**Modelo é retrato de um momento.** Preços de hardware mudam com
lançamentos, câmbio, promoções. O modelo treinado nas coletas de junho
e julho de 2026 vai gradualmente ficar desatualizado. Estratégia
saudável seria re-treinar periodicamente com coletas novas.

---

## 9. Trabalho futuro

**Migrar app para RF pooled com log-transform.** O v2_03 mostrou que é
melhor que os 6 especialistas em 4 de 6 categorias, especialmente nas
mais fracas do v2_02 (SSD e placa-mãe). A migração exige refatorar o
carregamento de bundles (de 6 para 1), a inferência (destransformar
log→R$), a faixa conformal (fica assimétrica em R$) e adaptar o SHAP
para operar em log-space. Estimativa: 4–8 horas de trabalho.

**Integrar TabICL como opção no app.** Adicionar um toggle "modelo
principal / modelo alternativo" na aba 1, permitindo comparar as
previsões dos dois. Requer PyTorch com CUDA no ambiente de execução —
mais realista em servidor com GPU do que localmente. TabICL brilha
especialmente em GPU (R² 0,90) e sua **faixa nativa dispensa conformal**
com cobertura calibrada em 89,7%.

**Coletar mais dados, especialmente através de eventos.** Black Friday,
lançamento de nova geração de GPU, choque cambial. Todos são
oportunidades de testar o conformal em condições reais de shift, não
sintéticas.

**Melhorar features estruturalmente.** Colunas explícitas para
"produto de servidor" (bool), "kit RGB/gamer" (bool), "geração" (numérica
ordinal), permitindo o modelo aprender relações que hoje ficam implícitas.
Também: extrair mais specs de placa-mãe (chipset extendido, quantidade
de slots PCIe, VRM) e SSD (velocidade de escrita, TBW, DRAM cache).

---

## Autoria

Juliano e Beatriz — disciplina Perspectivas de Dados do Departamento de Estatística da UFSCar
