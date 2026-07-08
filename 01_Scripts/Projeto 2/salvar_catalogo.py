"""
Gera modelos/catalogo.parquet: dataset completo (todas as datas, todas as
categorias) com specs extraídas via `features.py` — usado pelo app Streamlit
nas abas "Peças compatíveis" e "Build por orçamento".

Rodar após treinar os modelos (v2_02) ou após uma nova coleta:

    python salvar_catalogo.py

Se você mudar DATA_ROOT no notebook v2_02, mude aqui também (ou passe via
variável de ambiente KABUM_DATA_ROOT).
"""

from __future__ import annotations

import glob
import os
import pathlib

import pandas as pd

from features import (
    features_cpu,
    features_fonte,
    features_gpu,
    features_placa_mae,
    features_ram,
    features_ssd,
    normalizar_gpu_modelo,
)

# ---------------------------------------------------------------------------
# Configuração — ajuste para o seu ambiente
# ---------------------------------------------------------------------------

DATA_ROOT = os.environ.get(
    "KABUM_DATA_ROOT",
    r"C:\Users\julia\OneDrive\Área de Trabalho\Projetos\Perspectivas de Dados\perspectiva_dados_projeto2\00_Dados",
)
MODEL_DIR = pathlib.Path(__file__).parent / "modelos"

FEATURES_POR_CAT = {
    "ram":       features_ram,
    "cpu":       features_cpu,
    "gpu":       features_gpu,
    "ssd":       features_ssd,
    "fonte":     features_fonte,
    "placa_mae": features_placa_mae,
}

# Mesmas colunas de spec por categoria usadas no v2_02 (merge por id)
FEATURES_COLS = {
    "ram":       ["ram_geracao", "ram_gb", "ram_mhz", "ram_cl", "ram_notebook"],
    "cpu":       ["cpu_marca", "cpu_socket", "cpu_serie", "cpu_tdp_w",
                  "cpu_ddr_suportado", "cpu_com_cooler"],
    "gpu":       ["gpu_marca_chip", "gpu_modelo", "gpu_vram_gb", "gpu_tdp_w"],
    "ssd":       ["ssd_interface", "ssd_geracao_pcie", "ssd_capacidade_gb", "ssd_notebook"],
    "fonte":     ["fonte_wattagem", "fonte_certificacao", "fonte_modular", "fonte_atx3"],
    "placa_mae": ["mobo_socket", "mobo_chipset", "mobo_ddr", "mobo_form_factor",
                  "mobo_slots_m2", "mobo_max_ram_gb"],
}


def carregar_e_extrair():
    """Junta todos os kabum_todas_pecas_<data>.csv com specs extraídas."""
    arquivos = sorted(glob.glob(f"{DATA_ROOT}/*/kabum_todas_pecas_*.csv"))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum kabum_todas_pecas_*.csv encontrado em {DATA_ROOT}"
        )
    print(f"{len(arquivos)} arquivo(s) encontrado(s).")

    dfs = []
    for arquivo in arquivos:
        pasta = pathlib.Path(arquivo).parent
        data_coleta = pasta.name
        base = pd.read_csv(arquivo)

        # extrai as specs por categoria (mesmo procedimento do notebook v2_02
        # de feature engineering, mas usando features.py como fonte única)
        for cat, fn in FEATURES_POR_CAT.items():
            mask = base["categoria_key"] == cat
            if not mask.any():
                continue
            sub = fn(base.loc[mask, ["nome"]].copy())
            # anota as colunas geradas
            cols_novas = [c for c in FEATURES_COLS[cat] if c in sub.columns]
            base.loc[mask, cols_novas] = sub[cols_novas].values

        # normaliza gpu_modelo (mesmo tratamento do notebook v2_02)
        if "gpu_modelo" in base.columns:
            base["gpu_modelo"] = base["gpu_modelo"].apply(normalizar_gpu_modelo)

        print(f"  {data_coleta}: {len(base)} produtos")
        dfs.append(base)

    df = pd.concat(dfs, ignore_index=True)
    return df


def main():
    df = carregar_e_extrair()

    # limpezas mínimas (mesmas do v2_02)
    df = df.dropna(subset=["preco_pix"])
    df = df[df["preco_pix"] >= 1]

    # dedup por (id, data_coleta) — mesmo id numa coleta é duplicata verdadeira
    antes = len(df)
    df = df.drop_duplicates(subset=["id", "data_coleta"])
    print(f"\n{antes} → {len(df)} linhas após deduplicação por (id, data_coleta)")

    # sanidade: ram_mhz não passa de 9000 (evita "56600 MHz" do dataset)
    if "ram_mhz" in df.columns:
        fora = df["ram_mhz"] > 9000
        if fora.any():
            print(f"[sanidade] {fora.sum()} valores de ram_mhz > 9000 -> NaN")
            df.loc[fora, "ram_mhz"] = None

    # sanidade: ram_gb até 256 (kits enterprise chegam a 128-256; acima é ruído)
    if "ram_gb" in df.columns:
        fora = df["ram_gb"] > 256
        if fora.any():
            print(f"[sanidade] {fora.sum()} valores de ram_gb > 256 -> NaN")
            df.loc[fora, "ram_gb"] = None

    # sanidade: gpu_vram_gb até 48 (RTX 6000 Ada e similares; acima é ruído,
    # geralmente número de código de produto capturado pelo regex)
    if "gpu_vram_gb" in df.columns:
        fora = df["gpu_vram_gb"] > 48
        if fora.any():
            print(f"[sanidade] {fora.sum()} valores de gpu_vram_gb > 48 -> NaN")
            df.loc[fora, "gpu_vram_gb"] = None

    # ---- marca produtos "não-genuínos" da categoria ------------------------
    # Cada categoria tem itens que o KaBuM classifica junto mas não são o
    # produto principal (adaptadores, dissipadores, coolers, carcaças, kits
    # sem GPU, etc.). Isso contamina o ranking de ofertas e a extração de
    # specs.
    #
    # A lista abaixo é conservadora: só palavras que **só** aparecem em
    # não-produto (nunca no produto genuíno). Se precisar ajustar, mexa aqui
    # e re-rode este script.

    PADROES_NAO_GENUINOS = {
        "ram":       r"\b(adaptador|dissipador|cooler|case|carcaça|carcaca|"
                     r"raid card|riser|controladora|extensor"
                     # RAMs de servidor/legacy — o modelo foi treinado em RAM
                     # desktop moderna e superestima o preço dessas.
                     r"|poweredge|proliant|precision|macpro|mac pro|"
                     r"ml\d{3}|dl\d{3}|bl\d{3}|c\d{4}\b|"
                     r"ecc|registered|rdimm|udimm ecc|"
                     r"512\s*mb|1024\s*mb|"      # DDR/DDR2 antigos
                     r"pc2-|pc3l-"                # PC2-xxxx (DDR2) e PC3L (DDR3L)
                     r")\b",
        "cpu":       r"\b(cooler|water cooler|arctic|noctua|dissipador|"
                     r"pasta térmica|pasta termica|thermal|adaptador socket)\b",
        "gpu":       r"\b(suporte de gpu|suporte para placa|suporte placa|"
                     r"suporte ajustável|suporte ajustavel|"
                     r"riser|extensor|cabo|adaptador|"
                     r"case|carcaça|carcaca|cooler|dissipador|water block|"
                     # GPUs muito antigas/entrada (ruído no ranking):
                     r"g210|gt\s?210|gt\s?710|gt\s?730|gt\s?1030|"
                     r"radeon hd|radeon r5|radeon r7\s|radeon r9|"
                     r"radeon x\d)\b",
        "ssd":       r"\b(adaptador|gaveta|enclosure|case externo|"
                     r"caixa externa|cabo|dock)\b",
        "fonte":     r"\b(cabo|extensor|adaptador|comb|comb power|"
                     r"protetor|filtro de linha)\b",
        "placa_mae": r"\b(adaptador|riser|extensor|espa[cç]ador|"
                     r"cabo|painel)\b",
    }

    df["eh_genuino"] = True
    for cat, padrao in PADROES_NAO_GENUINOS.items():
        mask = (df["categoria_key"] == cat) & df["nome"].str.contains(
            padrao, case=False, na=False, regex=True
        )
        if mask.any():
            print(f"[não-genuíno/{cat}] {mask.sum()} produtos marcados "
                  f"como não-genuínos (adaptadores, coolers, cabos etc.)")
            df.loc[mask, "eh_genuino"] = False

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    caminho = MODEL_DIR / "catalogo.parquet"
    df.to_parquet(caminho, index=False)
    print(f"\n✓ Catálogo salvo em: {caminho}")
    print(f"  {len(df)} produtos totais")
    print(f"  {df['categoria_key'].nunique()} categorias:")
    for cat, n in df["categoria_key"].value_counts().items():
        print(f"    {cat:10s}: {n} produtos")


if __name__ == "__main__":
    main()
