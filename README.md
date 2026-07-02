# Projeto 1 — Sistema de Recomendação de Peças de PC (KaBuM)

Pipeline de **web scraping → NLP → análise** sobre o catálogo de hardware do
KaBuM, que coleta produtos de seis categorias, extrai as especificações técnicas
embutidas no nome de cada produto e recomenda montagens (*builds*) de PC
compatíveis e com bom custo-benefício.

> Disciplina: **Perspectivas de Dados** · Autor: **Juliano** · Ano: 2026

---

## Problema

Em e-commerces de hardware, especificações críticas (capacidade, frequência,
geração, latência, socket, TDP) ficam **embutidas no campo de texto livre do nome
do produto**, e não em campos estruturados. Isso impede comparações objetivas e,
principalmente, dificulta saber quais peças são compatíveis entre si. O projeto
ataca esse problema estruturando o texto e usando as specs recuperadas para montar
um recomendador de builds.

---

## Pipeline

O repositório espelha o pipeline pedido na disciplina, e a numeração das pastas
deixa a ordem explícita:

1. **Coleta** — scraping das 6 categorias a partir do bloco `__NEXT_DATA__` que o
   Next.js do KaBuM embute na página (ver detalhe abaixo).
2. **NLP** — estruturação do nome do produto em specs, em duas frentes
   (representação textual clássica + extração de informação via regex).
3. **Análise / Recomendação** — filtro de compatibilidade + ranking por
   custo-benefício, em dois modos de uso.

---

## 1. Coleta de dados

- **Fonte:** catálogo do KaBuM, 6 categorias — **RAM, CPU, GPU, placa-mãe, SSD e
  fonte** — totalizando **~4.636 produtos**.
- **Técnica-chave (`__NEXT_DATA__`):** o site é renderizado em Next.js, então o
  HTML não traz os produtos prontos. Eles estão num JSON dentro da tag
  `<script id="__NEXT_DATA__">`. O detalhe que diferencia a coleta é que esse JSON
  é **aninhado**: o campo `props.pageProps.data` volta a ser uma *string* JSON, o
  que exige **dois `json.loads` encadeados** até chegar em
  `catalogServer.data` (a lista de produtos). Feito com `requests` +
  `BeautifulSoup`, **sem Selenium**.
- **Campos coletados por produto:** `id`, `nome`, `preco_original`, `preco_atual`,
  `preco_pix`, `desconto_pct`, `avaliacao`, `num_avaliacoes`, `disponivel`,
  `fabricante`, `categoria`, `garantia`, `frete_gratis`, `url`, `data_coleta`.
- **Organização:** cada execução salva um CSV por categoria mais um consolidado,
  dentro de uma subpasta nomeada pela data da coleta (`00_Dados/AAAA-MM-DD/`),
  permitindo múltiplos *snapshots* ao longo do tempo.

---

## 2. NLP — do nome do produto às specs

O NLP foi feito em **duas abordagens**, e a transição entre elas é o principal arco
metodológico do trabalho.

### 2.1 Exploração textual clássica (em RAM)

Sobre os nomes dos produtos de RAM: limpeza de texto, remoção de *stopwords*,
vetorização com **Bag of Words** (`CountVectorizer`) e **TF-IDF**, e visualização
em 2D com **t-SNE** colorida por fabricante.

**Achado:** o t-SNE agrupou os produtos por **fabricante e estilo de escrita**
(Corsair "Vengeance", Kingston "Fury Beast"), **não** pelas specs técnicas. Uma
RAM DDR4 e uma DDR5 da mesma marca ficam próximas porque compartilham as palavras
de marketing do nome. Ou seja: a representação textual genérica **não resolve
compatibilidade**, que é o que o projeto precisa. Essa "falha útil" motivou a
mudança de abordagem.

### 2.2 Extração de informação via regex (todas as categorias)

Em vez de tratar o nome como saco de palavras, extraem-se diretamente as entidades
estruturadas (*Information Extraction*), com padrões por categoria:

| Categoria | Specs extraídas (exemplos) |
|---|---|
| RAM | geração (DDR4/DDR5), capacidade (GB), frequência (MHz), latência (CL) |
| CPU | socket, marca, geração, TDP |
| GPU | VRAM, TDP |
| Placa-mãe | socket, chipset, DDR suportado, form factor, slots M.2 |
| SSD | capacidade, interface (M.2/SATA) |
| Fonte | wattagem, modular |

**Tabelas de referência** complementam a regex quando o nome não é explícito:
`socket → DDR` (ex.: AM5→DDR5, LGA1700→DDR4/DDR5) e `modelo de CPU → socket`
(inferência pelo modelo quando o socket não aparece no texto).

**Cobertura (exemplos):** specs mais diretas ficaram acima de 90% (`ram_gb`,
`cpu_marca`, `fonte_wattagem`, `gpu_vram_gb`). O `cpu_socket`, campo crítico pra
compatibilidade, chegou a **84,4%** após tratar variações de escrita
(`LGA  1700`, `Sk1151`, `1151p`, sockets AMD/Xeon antigos). Campos raramente
citados no nome (chipset, latência CL) ficaram baixos por natureza do dado.

---

## 3. Sistema de recomendação

Duas camadas:

### 3.1 Filtro de compatibilidade (regras rígidas)

1. **CPU ↔ placa-mãe** — mesmo socket.
2. **RAM ↔ placa-mãe** — mesma geração DDR.
3. **RAM ↔ CPU** — geração DDR suportada.
4. **SSD ↔ placa-mãe** — interface disponível (M.2/SATA).
5. **Fonte ↔ CPU+GPU** — `wattagem ≥ (TDP_cpu + TDP_gpu) × 1,25`.

Um passo de **limpeza por categoria** (dicionário `FILTROS_CATEGORIA`) remove
produtos que caíram na categoria errada (ex.: "Suporte para placa de vídeo" na
categoria GPU, adaptadores na categoria SSD) e exclui RAM SODIMM (notebook) das
builds desktop.

### 3.2 Ranking por custo-benefício

Score normalizado combinando três fatores:

```
score = 0,5 × preço* + 0,3 × avaliação + 0,2 × popularidade
```

(*preço normalizado; `popularidade` derivada de `num_avaliacoes`.*)

### 3.3 Dois modos de uso

- **Modo 1 — Build por orçamento:** o usuário informa um valor; o sistema
  distribui o orçamento por categoria com pesos de mercado (GPU 35%, CPU 22%,
  placa-mãe 15%, fonte 10%, RAM 10%, SSD 8%), seleciona candidatos, aplica o filtro
  de compatibilidade e retorna a build compatível de maior score total.
- **Modo 2 — Peças compatíveis:** o usuário escolhe uma peça de referência (ex.:
  "Ryzen 5 5600") e o sistema retorna as peças compatíveis de outra categoria
  ranqueadas por score (ex.: placas-mãe AM4/DDR4).

---

## Resultados

- **Modo 1:** builds coerentes e compatíveis dentro do orçamento (ex.: teste de
  R$ 5.000 com CPU + placa-mãe + RAM mutuamente compatíveis e fonte adequada).
- **Modo 2:** para um Ryzen 5 5600, retornou apenas placas-mãe AM4/DDR4, corretas,
  ordenadas por score.

---

## Limitações conhecidas

- **Snapshot único:** a base é uma fotografia de uma data; sem série temporal de
  preços (mitigável coletando em várias datas — a estrutura de pastas já suporta).
- **Sem reviews textuais:** só metadados dos produtos, não o texto das avaliações.
- **Orçamento não é totalmente usado no Modo 1:** os pesos por categoria são fixos
  e o saldo que sobra em uma categoria **não é redistribuído** para as outras, então
  o total costuma ficar abaixo do orçamento informado. É a limitação mais relevante
  e uma candidata natural a melhoria.
- **Score com pesos fixos e arbitrários:** o custo-benefício é uma heurística feita
  à mão, não um modelo aprendido dos dados.

---

## Tecnologias

`Python` · `requests` · `BeautifulSoup` · `pandas` · `scikit-learn`
(`CountVectorizer`, `TfidfVectorizer`, `t-SNE`) · `matplotlib` · `Jupyter`.

---

## Estrutura do repositório

```
projeto1-kabum/
├── README.md
├── requirements.txt
├── 00_dados/
│   ├── AAAA-MM-DD/                      # snapshot por data de coleta
│   │   ├── kabum_ram_AAAA-MM-DD.csv
│   │   ├── kabum_cpu_AAAA-MM-DD.csv
│   │   ├── kabum_gpu_AAAA-MM-DD.csv
│   │   ├── kabum_ssd_AAAA-MM-DD.csv
│   │   ├── kabum_fonte_AAAA-MM-DD.csv
│   │   └── kabum_placa_mae_AAAA-MM-DD.csv
│   └── DICIONARIO.md                    # descrição das colunas
├── 01_scraping/
│   └── 01_scraping_kabum.ipynb          # coleta multi-categoria (__NEXT_DATA__)
├── 02_nlp/
│   └── 02_extracao_specs.ipynb          # regex/IE → features estruturadas
├── 03_analise/
│   ├── 03_eda.ipynb
│   └── 04_score_custo_beneficio.ipynb   # score + compatibilidade + 2 modos
├── 04_resultados/
│   ├── figuras/
│   └── tabelas/
└── slides/
    └── apresentacao.pdf                 # Beamer (Overleaf)
```

## Como executar

```bash
pip install -r requirements.txt
jupyter notebook
```

Rodar os notebooks na ordem numérica (`01_` → `02_` → `03_`).

---

## Artefatos gerados

- Documento de **regras de compatibilidade e atributos por categoria** (.docx)
- Notebooks: **scraping**, **feature engineering** e **recomendação**
- **Relatório acadêmico** completo (.docx)
- **Slides** em Beamer/LaTeX (tema Madrid)
- **Diagrama Excalidraw** do fluxo de parsing `__NEXT_DATA__`

---

## Próximo passo (v2)

Substituir o score de custo-benefício feito à mão por um **modelo de preço justo
aprendido dos dados**, com **conformal prediction** (intervalo de preço →
detector de oferta/sobrepreço), **SHAP** (explicar o preço) e comparação com um
**modelo fundacional tabular (TabICL)**; e levar o recomendador do notebook para
um **aplicativo interativo (Streamlit)** com demonstração ao vivo.
