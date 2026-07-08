"""
KaBuM — Extração de features a partir do nome do produto.

Este módulo é a fonte única da verdade para as funções `features_<categoria>`
usadas pelo notebook de feature engineering e pelo app Streamlit.

Cada função aceita um DataFrame com a coluna `nome` e devolve o mesmo
DataFrame com colunas adicionais preenchidas via regex. Também exportamos
uma função `extrair_features_produto(nome, categoria)` que trabalha sobre um
único nome (para uso no app).

Se você melhorar um regex, mude aqui e re-importe no notebook e no app.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers de regex
# ---------------------------------------------------------------------------

def extrair(nome, padrao, tipo=str, grupo=1):
    """Aplica um regex ao nome e retorna o grupo capturado (ou None)."""
    if pd.isna(nome):
        return None
    match = re.search(padrao, nome, re.IGNORECASE)
    if match:
        try:
            return tipo(match.group(grupo))
        except (ValueError, IndexError):
            return None
    return None


def contem(nome, padrao):
    """Retorna True se o padrão for encontrado no nome."""
    if pd.isna(nome):
        return False
    return bool(re.search(padrao, nome, re.IGNORECASE))


# ---------------------------------------------------------------------------
# 1. RAM
# ---------------------------------------------------------------------------

def features_ram(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    n = d["nome"]

    d["ram_geracao"]  = n.apply(lambda x: extrair(x, r"(DDR[2-5])"))
    d["ram_gb"]       = n.apply(lambda x: extrair(x, r"(\d+)\s*GB", int))
    d["ram_mhz"]      = n.apply(lambda x: extrair(x, r"(\d{3,5})\s*MHz", int))
    d["ram_cl"]       = n.apply(lambda x: extrair(x, r"CL(\d+)", int))
    d["ram_notebook"] = n.apply(lambda x: contem(x, r"Notebook|SODIMM|SO-DIMM"))

    return d


# ---------------------------------------------------------------------------
# 2. CPU
# ---------------------------------------------------------------------------

# socket → gerações DDR suportadas (usada quando o nome não informa DDR)
SOCKET_DDR = {
    "AM5":     "DDR5",
    "AM4":     "DDR4",
    "AM3":     "DDR3",
    "AM2":     "DDR2",
    "FM2":     "DDR3",
    "STR5":    "DDR5",
    "SP3":     "DDR4",
    "LGA1851": "DDR5",
    "LGA1700": "DDR4/DDR5",
    "LGA1200": "DDR4",
    "LGA1151": "DDR4",
    "LGA1150": "DDR3",
    "LGA1155": "DDR3",
    "LGA2011": "DDR3/DDR4",
}


# Fallback: quando o nome não menciona o socket explicitamente (ex.:
# "Core i3-14100F" sem "LGA1700"), inferir pelo modelo da CPU.
# Padrões em regex — o primeiro que casar define o socket.
MODELO_SOCKET_CPU = [
    # Intel — Core Ultra série 2 (Arrow Lake)
    (r"Core\s*Ultra\s*[579]\s*2\d{2}",         "LGA1851"),
    # Intel — Core Ultra série 1 e 12/13/14th gen usam LGA1700
    (r"Core\s*Ultra\s*[579]\s*1\d{2}",         "LGA1700"),
    (r"i[3579]-14\d{3}",                        "LGA1700"),
    (r"i[3579]-13\d{3}",                        "LGA1700"),
    (r"i[3579]-12\d{3}",                        "LGA1700"),
    (r"i[3579]-11\d{3}",                        "LGA1200"),
    (r"i[3579]-10\d{3}",                        "LGA1200"),
    (r"i[3579]-9\d{3}",                         "LGA1151"),
    (r"i[3579]-8\d{3}",                         "LGA1151"),
    (r"i[3579]-7\d{3}",                         "LGA1151"),
    (r"i[3579]-6\d{3}",                         "LGA1151"),
    # AMD — Ryzen série 9000/8000/7000 -> AM5
    (r"Ryzen\s*[3579]\s*9\d{3}",                "AM5"),
    (r"Ryzen\s*[3579]\s*8\d{3}",                "AM5"),
    (r"Ryzen\s*[3579]\s*7\d{3}",                "AM5"),
    # AMD — Ryzen série 5000/4000/3000/2000/1000 -> AM4
    (r"Ryzen\s*[3579]\s*5\d{3}",                "AM4"),
    (r"Ryzen\s*[3579]\s*4\d{3}",                "AM4"),
    (r"Ryzen\s*[3579]\s*3\d{3}",                "AM4"),
    (r"Ryzen\s*[3579]\s*2\d{3}",                "AM4"),
    (r"Ryzen\s*[3579]\s*1\d{3}",                "AM4"),
    # Threadripper série 7000 -> sTR5
    (r"Threadripper.*7\d{3}",                   "STR5"),
    # EPYC -> SP3/SP5
    (r"EPYC\s*9\d{3}",                          "SP5"),
    (r"EPYC\s*7\d{3}",                          "SP3"),
]


def _inferir_socket_por_modelo(nome):
    """Fallback: procura o modelo da CPU no nome e retorna o socket."""
    if pd.isna(nome):
        return None
    for padrao, socket in MODELO_SOCKET_CPU:
        if re.search(padrao, nome, re.IGNORECASE):
            return socket
    return None


def _normalizar_socket_cpu(s):
    if pd.isna(s):
        return None
    s = re.sub(r"\s+", "", s).upper()
    num = re.search(r"\d{3,4}", s)
    if not num:
        return s
    n = num.group()
    if s in ("AM2", "AM3", "AM4", "AM5", "FM1", "FM2", "STR5", "SP3"):
        return s
    return f"LGA{n}"


def features_cpu(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    n = d["nome"]

    d["cpu_socket"] = n.apply(lambda x: extrair(x,
        r"(AM[2-5]|FM[12]|STR5|SP3"
        r"|LGA\s*\d{3,4}"
        r"|Sk(\d{3,4})"
        r"|(\d{3,4})[pP]\b"
        r"|L(\d{4})\b"
        r"|\((\d{4})\)"
        r"|\b(1[0-9]\d{2}|20[0-9]{2})\s+(?:Intel|Core|i[3579]|Xeon)"
        r")"
    ))
    d["cpu_socket"] = d["cpu_socket"].apply(_normalizar_socket_cpu)

    # Fallback: se o regex principal não achou socket, tenta inferir pelo modelo
    # (ex.: "Core i3-14100F" → LGA1700 via tabela MODELO_SOCKET_CPU)
    faltando = d["cpu_socket"].isna()
    if faltando.any():
        d.loc[faltando, "cpu_socket"] = n[faltando].apply(_inferir_socket_por_modelo)

    # TDP: exige "W" isolado (\s+W ou pontuação antes), evita capturar SKU
    # tipo "100-100000926BOX" onde o número precede um W literal.
    # (Mantido igual ao notebook original — sabemos que ainda gera outliers em
    # alguns SKUs; ver `SANIDADE` no notebook v2_02.)
    d["cpu_tdp_w"] = n.apply(lambda x: extrair(x, r"(\d+)\s*W(?!h)", int))

    d["cpu_com_cooler"] = n.apply(lambda x: contem(x, r"Box|Wraith|Cooler|Fan"))
    d["cpu_marca"] = n.apply(lambda x:
        "Intel" if contem(x, r"Intel|Core\s*i[3579]|Core\s*Ultra") else
        "AMD"   if contem(x, r"AMD|Ryzen|Athlon") else None
    )
    d["cpu_serie"] = n.apply(lambda x: extrair(x,
        r"(Ryzen\s*[3579]|Core\s*i[3579]|Core\s*Ultra\s*\d)"
    ))
    d["cpu_ddr_suportado"] = d["cpu_socket"].map(SOCKET_DDR)

    return d


# ---------------------------------------------------------------------------
# 3. Placa-mãe
# ---------------------------------------------------------------------------

def features_placa_mae(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    n = d["nome"]

    d["mobo_socket"]      = n.apply(lambda x: extrair(x, r"(AM[45]|LGA\s?\d{3,4})"))
    d["mobo_socket"]      = d["mobo_socket"].str.replace(" ", "").str.upper()
    d["mobo_chipset"]     = n.apply(lambda x: extrair(x, r"\b([A-Z]\d{3,4})(?!\s*MHz)\b"))
    d["mobo_ddr"]         = n.apply(lambda x: extrair(x, r"(DDR[45])"))
    d["mobo_form_factor"] = n.apply(lambda x: extrair(x, r"\b(ATX|mATX|m-ATX|Micro-ATX|Mini-ATX|ITX|Mini-ITX)\b"))
    d["mobo_slots_m2"]    = n.apply(lambda x: extrair(x, r"(\d+)\s*x?\s*M\.2", int))
    d["mobo_max_ram_gb"]  = n.apply(lambda x: extrair(x, r"(\d+)\s*GB(?=.*RAM)", int))

    return d


# ---------------------------------------------------------------------------
# 4. GPU
# ---------------------------------------------------------------------------

# modelo GPU → TDP aproximado em Watts
GPU_TDP = {
    "RTX 5090": 575, "RTX 5080": 360, "RTX 5070 Ti": 300, "RTX 5070": 250,
    "RTX 5060 Ti": 180, "RTX 5060": 150,
    "RTX 4090": 450, "RTX 4080": 320, "RTX 4070 Ti": 285, "RTX 4070": 200,
    "RTX 4060 Ti": 165, "RTX 4060": 115,
    "RTX 3090": 350, "RTX 3080": 320, "RTX 3070": 220, "RTX 3060": 170,
    "RX 9070 XT": 304, "RX 9070": 220,
    "RX 7900 XTX": 355, "RX 7900 XT": 315, "RX 7800 XT": 263, "RX 7700 XT": 245,
    "RX 7600": 165, "RX 6700 XT": 230, "RX 6600": 132,
}


def normalizar_gpu_modelo(s):
    """Colapsa variações tipo 'RTX 5060 Ti' e 'RTX5060ti' no mesmo texto."""
    if pd.isna(s):
        return s
    s = str(s).upper()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"(RTX|GTX|RX)(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)(TI|XT|SUPER)", r"\1 \2", s)
    return s.strip()


def _extrair_modelo_gpu(nome):
    """Retorna o modelo mais longo da tabela GPU_TDP encontrado no nome."""
    if pd.isna(nome):
        return None
    for modelo in sorted(GPU_TDP.keys(), key=len, reverse=True):
        if re.search(re.escape(modelo), nome, re.IGNORECASE):
            return modelo
    # fallback: padrão genérico (pega modelos não catalogados em GPU_TDP)
    m = re.search(r"(RTX\s?\d{4}(?:\s?Ti)?|RX\s?\d{4}(?:\s?XT)?)", nome, re.IGNORECASE)
    return m.group(1).strip() if m else None


# mapa TDP com as chaves já normalizadas — evita bug em que "RTX 5060 Ti"
# (chave original) não casa com "RTX 5060 TI" (após normalizar_gpu_modelo).
GPU_TDP_NORMALIZADO = {normalizar_gpu_modelo(k): v for k, v in GPU_TDP.items()}


def features_gpu(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    n = d["nome"]

    d["gpu_marca_chip"] = n.apply(lambda x:
        "NVIDIA" if contem(x, r"RTX|GTX|NVIDIA") else
        "AMD"    if contem(x, r"\bRX\b|Radeon|AMD") else
        "Intel"  if contem(x, r"Arc|Intel") else None
    )
    d["gpu_modelo"]  = n.apply(_extrair_modelo_gpu).apply(normalizar_gpu_modelo)
    d["gpu_vram_gb"] = n.apply(lambda x: extrair(x, r"(\d+)\s*GB", int))
    d["gpu_tdp_w"]   = d["gpu_modelo"].map(GPU_TDP_NORMALIZADO)

    return d


# ---------------------------------------------------------------------------
# 5. SSD
# ---------------------------------------------------------------------------

def features_ssd(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    n = d["nome"]

    d["ssd_interface"] = n.apply(lambda x:
        "NVMe" if contem(x, r"NVMe|M\.2") else
        "SATA" if contem(x, r"SATA") else None
    )
    d["ssd_geracao_pcie"] = n.apply(lambda x: extrair(x, r"PCIe\s?(\d\.\d)|Gen\s?(\d)", grupo=1))
    d["ssd_capacidade_gb"] = n.apply(lambda x: (
        extrair(x, r"(\d+)\s*TB", float) * 1024
        if contem(x, r"\d+\s*TB") else
        extrair(x, r"(\d+)\s*GB", int)
    ))
    d["ssd_notebook"] = n.apply(lambda x: contem(x, r"Notebook|2230|2242"))

    return d


# ---------------------------------------------------------------------------
# 6. Fonte
# ---------------------------------------------------------------------------

def features_fonte(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    n = d["nome"]

    d["fonte_wattagem"]     = n.apply(lambda x: extrair(x, r"(\d{3,4})\s*W(?!h)", int))
    d["fonte_certificacao"] = n.apply(lambda x: extrair(x, r"(Titanium|Platinum|Gold|Silver|Bronze)"))
    d["fonte_modular"]      = n.apply(lambda x:
        "Full"  if contem(x, r"Full.?Modular|Modular\s*Full") else
        "Semi"  if contem(x, r"Semi.?Modular|Modular\s*Semi") else
        "Não"   if contem(x, r"Não.?Modular|Non.?Modular") else None
    )
    d["fonte_atx3"] = n.apply(lambda x: contem(x, r"ATX\s*3\.0|ATX3"))

    return d


# ---------------------------------------------------------------------------
# Dispatch: extração para uma única string (usado pelo app)
# ---------------------------------------------------------------------------

FEATURES_POR_CATEGORIA = {
    "ram":       features_ram,
    "cpu":       features_cpu,
    "gpu":       features_gpu,
    "ssd":       features_ssd,
    "fonte":     features_fonte,
    "placa_mae": features_placa_mae,
}


def extrair_features_produto(nome: str, categoria: str) -> dict:
    """Extrai features de um único nome de produto.

    Uso no app:
        specs = extrair_features_produto("Memória Kingston Fury 16GB DDR5 6000MHz CL30", "ram")
        # -> {"ram_gb": 16, "ram_mhz": 6000, "ram_cl": 30, "ram_geracao": "DDR5", ...}

    Retorna apenas as colunas geradas pela extração (não inclui `nome` nem
    campos externos como `fabricante`, que vêm do site).
    """
    if categoria not in FEATURES_POR_CATEGORIA:
        raise ValueError(
            f"categoria '{categoria}' desconhecida; use uma de "
            f"{list(FEATURES_POR_CATEGORIA)}"
        )

    fn = FEATURES_POR_CATEGORIA[categoria]
    df_um = pd.DataFrame({"nome": [nome]})
    df_um = fn(df_um)
    row = df_um.iloc[0].to_dict()
    row.pop("nome", None)
    return row
