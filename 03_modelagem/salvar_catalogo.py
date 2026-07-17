"""
Gera modelos/catalogo.parquet: dataset completo (todas as datas, todas as
categorias) com specs extraídas via `features.py` — usado pelo app Streamlit
nas abas "Peças compatíveis" e "Build por orçamento".

Estratégia de leitura por pasta de coleta:

  1. Se existir `kabum_todas_pecas_<data>.csv` na pasta → carrega e faz merge
     com os `kabum_<cat>_<data>_features.csv` para trazer as specs.

  2. Se NÃO existir o `todas_pecas` mas existirem os 6 `_features.csv` →
     concatena eles diretamente (esses arquivos já são autocontidos:
     têm `data_coleta`, `categoria_key`, `id`, `preco_pix`, `fabricante`,
     etc. + as specs extraídas).

Rodar:
    python salvar_catalogo.py
"""

from __future__ import annotations

import glob
import os
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "02_features"))

from features import (
    features_cpu,
    features_fonte,
    features_gpu,
    features_placa_mae,
    features_ram,
    features_ssd,
    normalizar_gpu_modelo,
)

# Força stdout/stderr para UTF-8: o console padrão do Windows (cp1252/cp437)
# não sabe imprimir caracteres como "→" ou "✓" usados nos prints abaixo,
# e quebraria com UnicodeEncodeError em máquinas sem essa configuração.
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

# 00_Dados/ mora na raiz do repositório (um nível acima deste script).
# Defina a variável de ambiente KABUM_DATA_ROOT para apontar para outro lugar
# sem editar o código.
DATA_ROOT = os.environ.get(
    "KABUM_DATA_ROOT",
    str(pathlib.Path(__file__).resolve().parents[1] / "00_Dados"),
)
MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "modelos"

FEATURES_POR_CAT = {
    "ram":       features_ram,
    "cpu":       features_cpu,
    "gpu":       features_gpu,
    "ssd":       features_ssd,
    "fonte":     features_fonte,
    "placa_mae": features_placa_mae,
}

FEATURES_COLS = {
    "ram":       ["ram_geracao", "ram_gb", "ram_mhz", "ram_cl", "ram_notebook"],
    "cpu":       ["cpu_marca", "cpu_socket", "cpu_serie", "cpu_tdp_w",
                  "cpu_ddr_suportado", "cpu_com_cooler",
                  "cpu_cores", "cpu_threads", "cpu_clock_ghz"],
    "gpu":       ["gpu_marca_chip", "gpu_modelo", "gpu_vram_gb", "gpu_tdp_w"],
    "ssd":       ["ssd_interface", "ssd_geracao_pcie", "ssd_capacidade_gb",
                  "ssd_notebook", "ssd_leitura_mbs"],
    "fonte":     ["fonte_wattagem", "fonte_certificacao", "fonte_modular", "fonte_atx3"],
    "placa_mae": ["mobo_socket", "mobo_chipset", "mobo_ddr", "mobo_form_factor",
                  "mobo_slots_m2", "mobo_max_ram_gb"],
}


def carregar_pasta_via_todas_pecas(pasta: pathlib.Path, data_coleta: str) -> pd.DataFrame:
    """Fluxo original: usa kabum_todas_pecas_<data>.csv + aplica extração de specs."""
    arquivo = pasta / f"kabum_todas_pecas_{data_coleta}.csv"
    base = pd.read_csv(arquivo)

    for cat, fn in FEATURES_POR_CAT.items():
        mask = base["categoria_key"] == cat
        if not mask.any():
            continue
        sub = fn(base.loc[mask, ["nome"]].copy())
        cols_novas = [c for c in FEATURES_COLS[cat] if c in sub.columns]
        base.loc[mask, cols_novas] = sub[cols_novas].values

    return base


def carregar_pasta_via_features(pasta: pathlib.Path, data_coleta: str) -> pd.DataFrame:
    """Fluxo alternativo: concatena os 6 kabum_<cat>_<data>_features.csv.

    Esses arquivos já são autocontidos (têm data_coleta, categoria_key, id,
    preco_pix, fabricante, etc.) — basta concatenar.
    """
    dfs = []
    for cat in FEATURES_POR_CAT:
        arq = pasta / f"kabum_{cat}_{data_coleta}_features.csv"
        if arq.exists():
            dfs.append(pd.read_csv(arq, encoding="utf-8-sig"))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, sort=False)


def carregar_e_extrair():
    """Percorre todas as pastas de coleta e retorna o df consolidado."""
    root = pathlib.Path(DATA_ROOT)
    pastas = sorted(p for p in root.iterdir() if p.is_dir())
    if not pastas:
        raise FileNotFoundError(f"Nenhuma pasta de coleta em {DATA_ROOT}")

    print(f"{len(pastas)} pasta(s) de coleta encontrada(s).\n")

    dfs = []
    for pasta in pastas:
        data_coleta = pasta.name
        arq_todas = pasta / f"kabum_todas_pecas_{data_coleta}.csv"

        if arq_todas.exists():
            base = carregar_pasta_via_todas_pecas(pasta, data_coleta)
            fonte_txt = "via todas_pecas"
        else:
            base = carregar_pasta_via_features(pasta, data_coleta)
            fonte_txt = "via _features.csv"

        if base.empty:
            print(f"  [pulei] {data_coleta}: nenhum CSV lido")
            continue

        # aplica normalização de gpu_modelo (idempotente — pode rodar em cima
        # de valores já normalizados)
        if "gpu_modelo" in base.columns:
            base["gpu_modelo"] = base["gpu_modelo"].apply(normalizar_gpu_modelo)

        print(f"  {data_coleta}: {len(base):>5} produtos  ({fonte_txt})")
        dfs.append(base)

    return pd.concat(dfs, ignore_index=True, sort=False)


def main():
    df = carregar_e_extrair()

    # limpezas mínimas
    df = df.dropna(subset=["preco_pix"])
    df = df[df["preco_pix"] >= 1]

    antes = len(df)
    df = df.drop_duplicates(subset=["id", "data_coleta"])
    print(f"\n{antes} → {len(df)} linhas após deduplicação por (id, data_coleta)")

    # sanidade
    if "ram_mhz" in df.columns:
        fora = df["ram_mhz"] > 9000
        if fora.any():
            print(f"[sanidade] {fora.sum()} valores de ram_mhz > 9000 -> NaN")
            df.loc[fora, "ram_mhz"] = None

    if "ram_gb" in df.columns:
        fora = df["ram_gb"] > 256
        if fora.any():
            print(f"[sanidade] {fora.sum()} valores de ram_gb > 256 -> NaN")
            df.loc[fora, "ram_gb"] = None

    if "gpu_vram_gb" in df.columns:
        fora = df["gpu_vram_gb"] > 48
        if fora.any():
            print(f"[sanidade] {fora.sum()} valores de gpu_vram_gb > 48 -> NaN")
            df.loc[fora, "gpu_vram_gb"] = None

    if "ssd_capacidade_gb" in df.columns:
        fora = (df["ssd_capacidade_gb"] < 32) | (df["ssd_capacidade_gb"] > 8192)
        if fora.any():
            print(f"[sanidade] {fora.sum()} valores de ssd_capacidade_gb fora de [32, 8192] -> NaN")
            df.loc[fora, "ssd_capacidade_gb"] = None

    # marca produtos "não-genuínos"
    # (?:...) em vez de (...) para evitar UserWarning do pandas
    PADROES_NAO_GENUINOS = {
        "ram":       r"\b(?:adaptador|dissipador|cooler|case|carcaça|carcaca|"
                     r"raid card|riser|controladora|extensor"
                     r"|poweredge|proliant|precision|macpro|mac pro|"
                     r"ml\d{3}|dl\d{3}|bl\d{3}|c\d{4}\b|"
                     r"ecc|registered|rdimm|udimm ecc|"
                     r"512\s*mb|1024\s*mb|"
                     r"pc2-|pc3l-"
                     r")\b",
        "cpu":       r"\b(?:cooler|water cooler|arctic|noctua|dissipador|"
                     r"pasta térmica|pasta termica|thermal|adaptador socket)\b",
        "gpu":       r"\b(?:suporte de gpu|suporte para placa|suporte placa|"
                     r"suporte ajustável|suporte ajustavel|"
                     r"riser|extensor|cabo|adaptador|"
                     r"case|carcaça|carcaca|cooler|dissipador|water block|"
                     r"g210|gt\s?210|gt\s?710|gt\s?730|gt\s?1030|"
                     r"radeon hd|radeon r5|radeon r7\s|radeon r9|"
                     r"radeon x\d|"
                     r"quadro|tesla|"
                     r"rtx\s?a\d{4}|rtx\s?ada|"
                     r"h100|h200|a100|a800|a40|a30|a10\b|"
                     r"l40|l4\b|"
                     r"mi\d{3}|instinct"
                     r")\b",
        "ssd":       r"\b(?:adaptador|gaveta|enclosure|case externo|case|"
                     r"caixa externa|cabo|dock|"
                     r"dc600m|dc1500|dc500|pm[0-9]{3,}|pm893|"
                     r"micron\s?\d{4}|sas\b|"
                     r"para\s+servidor|para servidor|"
                     r"poweredge|proliant|r\d{3}\s|r\d{4}\s"
                     r")\b",
        "fonte":     r"\b(?:cabo|extensor|adaptador|comb|comb power|"
                     r"protetor|filtro de linha)\b",
        "placa_mae": r"\b(?:adaptador|riser|extensor|espa[cç]ador|"
                     r"cabo|painel)\b",
    }

    df["eh_genuino"] = True
    for cat, padrao in PADROES_NAO_GENUINOS.items():
        mask = (df["categoria_key"] == cat) & df["nome"].str.contains(
            padrao, case=False, na=False, regex=True
        )
        if mask.any():
            print(f"[não-genuíno/{cat}] {mask.sum()} produtos marcados")
            df.loc[mask, "eh_genuino"] = False

    # Uniformiza tipos de colunas categóricas que podem vir misturadas
    # (float/str) das diferentes fontes (_features.csv vs merge). O parquet
    # rejeita mistura float/str na mesma coluna.
    colunas_uniformizar = [
        "ssd_geracao_pcie", "ram_geracao", "ram_cl", "mobo_chipset",
        "cpu_ddr_suportado", "gpu_modelo",
    ]
    for col in colunas_uniformizar:
        if col in df.columns:
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    caminho = MODEL_DIR / "catalogo.parquet"
    df.to_parquet(caminho, index=False)
    print(f"\n✓ Catálogo salvo em: {caminho}")
    print(f"  {len(df):,} produtos totais")
    print(f"  {df['data_coleta'].nunique()} coletas: {sorted(df['data_coleta'].unique())}")
    print(f"  {df['categoria_key'].nunique()} categorias:")
    for cat, n in df["categoria_key"].value_counts().items():
        print(f"    {cat:10s}: {n:,} produtos")


if __name__ == "__main__":
    main()
