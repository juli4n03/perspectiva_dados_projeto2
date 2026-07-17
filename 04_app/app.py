"""
KaBuM — Preço Justo (v0.2)

App Streamlit para os modelos gerados pelo notebook v2_02.

Novidades desta versão:
  - Aba "Analisar produto" implementada: formulário por categoria, auto-
    preencher a partir do nome (regex do módulo `features`), previsão de
    preço justo, faixa de 90%, veredito e waterfall SHAP.

Rodar:
    streamlit run app.py
"""

from __future__ import annotations

import pathlib
import sys
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "02_features"))

from features import FEATURES_POR_CATEGORIA, extrair_features_produto

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "modelos"
CATEGORIAS = ["ram", "cpu", "gpu", "ssd", "fonte", "placa_mae"]

LABEL_CATEGORIA = {
    "ram":       "RAM",
    "cpu":       "CPU",
    "gpu":       "GPU",
    "ssd":       "SSD",
    "fonte":     "Fonte",
    "placa_mae": "Placa-mãe",
}

# Rótulos amigáveis para os campos de spec (código -> exibição na UI)
LABEL_FEATURE = {
    # RAM
    "ram_gb":              "Capacidade (GB)",
    "ram_mhz":             "Frequência (MHz)",
    "ram_cl":              "Latência CAS",
    "ram_geracao":         "Geração (DDR)",
    # CPU
    "cpu_tdp_w":           "TDP (W)",
    "cpu_socket":          "Socket",
    "cpu_marca":           "Marca",
    "cpu_ddr_suportado":   "DDR suportado",
    "cpu_serie":           "Série",
    # GPU
    "gpu_vram_gb":         "VRAM (GB)",
    "gpu_tdp_w":           "TDP (W)",
    "gpu_marca_chip":      "Marca do chip",
    "gpu_modelo":          "Modelo",
    # SSD
    "ssd_capacidade_gb":   "Capacidade (GB)",
    "ssd_interface":       "Interface",
    "ssd_geracao_pcie":    "Geração PCIe",
    # Fonte
    "fonte_wattagem":      "Wattagem (W)",
    "fonte_modular":       "Modular",
    "fonte_certificacao":  "Certificação 80 Plus",
    # Placa-mãe
    "mobo_slots_m2":       "Slots M.2",
    "mobo_socket":         "Socket",
    "mobo_ddr":            "DDR",
    "mobo_form_factor":    "Form factor",
    "mobo_chipset":        "Chipset",
    # Comum
    "fabricante":          "Fabricante",
}

def label_de(campo: str) -> str:
    return LABEL_FEATURE.get(campo, campo)

st.set_page_config(
    page_title="KaBuM — Preço Justo",
    page_icon="💾",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Carregamento dos bundles
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Carregando modelos treinados...")
def carregar_bundles(model_dir: pathlib.Path) -> dict:
    bundles = {}
    for cat in CATEGORIAS:
        caminho = model_dir / f"modelo_preco_{cat}.joblib"
        if caminho.exists():
            try:
                bundles[cat] = joblib.load(caminho)
            except Exception as exc:
                st.warning(f"Falha ao carregar {caminho.name}: {exc}")
    return bundles


@st.cache_data(show_spinner="Carregando catálogo...")
def carregar_catalogo(model_dir: pathlib.Path) -> Optional[pd.DataFrame]:
    """Carrega modelos/catalogo.parquet. Retorna None se não existir.

    Usado pelas abas "Peças compatíveis" e "Build por orçamento". A aba
    "Analisar produto" NÃO depende do catálogo.
    """
    caminho = model_dir / "catalogo.parquet"
    if not caminho.exists():
        return None
    try:
        return pd.read_parquet(caminho)
    except Exception as exc:
        st.warning(f"Falha ao carregar catalogo.parquet: {exc}")
        return None


# ---------------------------------------------------------------------------
# Predição — usa o bundle para prever preço justo + faixa de 90%
# ---------------------------------------------------------------------------

def prever_preco_e_faixa(specs: dict, bundle: dict) -> dict:
    """Retorna {'preco_justo', 'faixa_baixo', 'faixa_alto', 'X_transformado'}.

    Se `bundle["usa_log"]` for True, o modelo interno prevê log1p(preço) e
    esta função destransforma para R$ (faixa fica assimétrica, o que é o
    comportamento correto para preços).
    """
    X1 = pd.DataFrame([specs])
    for c in bundle["num"] + bundle["cat"]:
        if c not in X1.columns:
            X1[c] = np.nan
    X1 = X1[bundle["num"] + bundle["cat"]]

    for c in bundle["cat"]:
        X1[c] = X1[c].where(X1[c].isna(), X1[c].astype(str))

    Xt = pd.DataFrame(
        bundle["preproc"].transform(X1),
        columns=bundle["preproc"].get_feature_names_out(),
    )
    pred_raw = float(bundle["modelo_media"].predict(Xt)[0])
    escala = float(np.sqrt(max(bundle["modelo_variancia"].predict(Xt)[0], 1e-6)))

    usa_log = bundle.get("usa_log", False)
    if usa_log:
        # faixa em log-space, depois destransformada -> assimétrica em R$
        preco = float(np.expm1(pred_raw))
        baixo = float(np.expm1(pred_raw - bundle["q_norm"] * escala))
        alto  = float(np.expm1(pred_raw + bundle["q_norm"] * escala))
    else:
        # modelo antigo: linear em R$
        preco = pred_raw
        baixo = pred_raw - bundle["q_norm"] * escala
        alto  = pred_raw + bundle["q_norm"] * escala

    baixo = max(baixo, 0.0)
    return {
        "preco_justo": preco,
        "faixa_baixo": baixo,
        "faixa_alto": alto,
        "X_transformado": Xt,
    }


def classificar(preco_real: float, lo: float, hi: float) -> tuple[str, str]:
    """Retorna (veredito, cor_hex) para uso na UI."""
    if preco_real < lo:
        return "🎉 Oferta", "#22c55e"      # verde
    if preco_real > hi:
        return "⚠️ Caro", "#ef4444"        # vermelho
    return "✅ Justo", "#3b82f6"           # azul


# ---------------------------------------------------------------------------
# Sidebar — status dos modelos
# ---------------------------------------------------------------------------

def render_sidebar(bundles: dict, catalogo: Optional[pd.DataFrame]) -> None:
    st.sidebar.header("Status dos modelos")
    if not bundles:
        st.sidebar.error(
            f"Nenhum bundle encontrado em `{MODEL_DIR}`.\n\n"
            "Rode o notebook `v2_02_modelo_preco_todas_categorias.ipynb`."
        )
        return

    linhas = []
    for cat in CATEGORIAS:
        if cat in bundles:
            m = bundles[cat]["metricas"]
            linhas.append({
                "categoria": LABEL_CATEGORIA[cat],
                "MAE": f"R$ {m['mae']:.0f}",
                "R²": f"{m['r2']:.2f}",
                "cobertura": f"{m['cobertura']:.0%}",
            })
        else:
            linhas.append({
                "categoria": LABEL_CATEGORIA[cat],
                "MAE": "—", "R²": "—", "cobertura": "ausente",
            })
    st.sidebar.dataframe(pd.DataFrame(linhas), hide_index=True, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Catálogo")
    if catalogo is None:
        st.sidebar.warning(
            "`catalogo.parquet` não encontrado. As abas "
            "**Peças compatíveis** e **Build por orçamento** ficam "
            "indisponíveis.\n\nGere com:\n```\npython salvar_catalogo.py\n```"
        )
    else:
        st.sidebar.success(
            f"{len(catalogo):,} produtos • "
            f"{catalogo['data_coleta'].nunique()} coletas"
        )
    st.sidebar.caption(f"Diretório: `{MODEL_DIR}`")


# ---------------------------------------------------------------------------
# Aba 1 — Analisar produto
# ---------------------------------------------------------------------------

def _render_formulario_specs(bundle: dict) -> dict:
    """Renderiza campos de spec com base no bundle. Retorna dict de specs.

    Usa `session_state` diretamente via `key=` — para popular via extração
    automática, escreva em `st.session_state[<key>]` ANTES de chamar essa
    função (e faça `st.rerun()`).
    """
    specs = {}
    campos = bundle["num"] + bundle["cat"]
    cat_key = bundle["categoria"]

    colunas = st.columns(2)
    for i, campo in enumerate(campos):
        col = colunas[i % 2]
        with col:
            key = f"{cat_key}_{'num' if campo in bundle['num'] else 'cat'}_{campo}"
            v = st.text_input(label_de(campo), key=key)
            if campo in bundle["num"]:
                if v.strip() == "":
                    specs[campo] = None
                else:
                    try:
                        specs[campo] = float(v)
                    except ValueError:
                        st.warning(f"'{label_de(campo)}' precisa ser numérico — valor '{v}' ignorado.")
                        specs[campo] = None
            else:
                specs[campo] = v.strip() if v.strip() else None

    return specs


def _preencher_session_state(bundle: dict, valores: dict) -> None:
    """Escreve as specs extraídas nas keys que os text_inputs vão ler."""
    cat_key = bundle["categoria"]
    for campo in bundle["num"] + bundle["cat"]:
        key = f"{cat_key}_{'num' if campo in bundle['num'] else 'cat'}_{campo}"
        val = valores.get(campo)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            st.session_state[key] = ""
        else:
            # números com float inteiro (16.0) mostrados como "16"
            if isinstance(val, float) and val.is_integer():
                st.session_state[key] = str(int(val))
            else:
                st.session_state[key] = str(val)


def _render_resultado(preco_real: float, resultado: dict, bundle: dict,
                      specs_usadas: Optional[dict] = None) -> None:
    preco_justo = resultado["preco_justo"]
    lo, hi = resultado["faixa_baixo"], resultado["faixa_alto"]
    veredito, cor = classificar(preco_real, lo, hi)
    desvio_pct = (preco_real - preco_justo) / preco_justo * 100 if preco_justo else 0.0

    st.subheader("Resultado")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Preço observado", f"R$ {preco_real:,.2f}")
    c2.metric("Preço justo (previsto)", f"R$ {preco_justo:,.2f}",
              delta=f"{desvio_pct:+.1f}%", delta_color="inverse")
    c3.metric("Faixa de 90%", f"R$ {lo:,.0f} – {hi:,.0f}")
    c4.markdown(
        f"<div style='background:{cor}22;border-left:4px solid {cor};"
        f"padding:0.75rem 1rem;border-radius:0.35rem;height:100%'>"
        f"<div style='font-size:0.85rem;opacity:0.7'>Veredito</div>"
        f"<div style='font-size:1.5rem;font-weight:600;color:{cor}'>{veredito}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Specs efetivamente usadas na previsão (importante para auditoria: mostra
    # ao usuário o que o modelo "viu" — evita a falsa sensação de que o app
    # entendeu specs que na verdade estavam em branco).
    if specs_usadas:
        preenchidas = {label_de(k): v for k, v in specs_usadas.items() if v is not None}
        if preenchidas:
            with st.expander("Specs usadas na previsão", expanded=False):
                st.write(preenchidas)
        else:
            st.warning(
                "⚠️ Nenhuma spec foi identificada — a previsão usou a mediana do "
                "treino para todos os campos. Verifique o nome do produto ou "
                "preencha manualmente."
            )

    st.markdown("---")
    st.subheader("Explicação (SHAP)")
    if bundle.get("usa_log", False):
        st.caption(
            "Cada barra mostra o quanto uma feature empurrou o **log-preço** "
            "para cima (vermelho) ou para baixo (azul). Uma barra de +0.10 "
            "equivale a aproximadamente +10% no preço final."
        )
    else:
        st.caption(
            "Cada barra mostra o quanto uma feature empurrou o preço para cima "
            "(vermelho) ou para baixo (azul) em relação à média dos produtos."
        )

    try:
        explainer = shap.TreeExplainer(bundle["modelo_media"])
        shap_values = explainer(resultado["X_transformado"])

        fig = plt.figure(figsize=(8, 4.5))
        shap.plots.waterfall(shap_values[0], show=False, max_display=10)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as exc:
        st.warning(f"Não foi possível gerar o gráfico SHAP: {exc}")


def aba_analisar_produto(bundles: dict) -> None:
    st.header("Analisar produto")
    st.markdown(
        "Cole o **nome do produto** e o **preço observado**, escolha a "
        "categoria — o app extrai as specs automaticamente (você pode "
        "corrigir manualmente) e responde: **oferta, justo ou caro?**."
    )

    if not bundles:
        st.info("Aguardando modelos. Ver painel lateral.")
        return

    cats_disponiveis = [c for c in CATEGORIAS if c in bundles]
    cat = st.selectbox(
        "Categoria",
        cats_disponiveis,
        format_func=lambda c: LABEL_CATEGORIA[c],
        key="cat_analise",
    )
    bundle = bundles[cat]

    col_nome, col_preco = st.columns([3, 1])
    with col_nome:
        nome = st.text_input(
            "Nome do produto",
            placeholder="Ex.: Memória RAM Kingston Fury Beast, 16GB, 3200MHz, DDR4, CL16",
            key=f"nome_{cat}",
        )
    with col_preco:
        preco_real = st.number_input(
            "Preço observado (R$)",
            min_value=0.0, value=0.0, step=10.0, format="%.2f",
            key=f"preco_{cat}",
        )

    if st.button("🪄 Extrair specs do nome", disabled=not nome.strip()):
        try:
            extraido = extrair_features_produto(nome, cat)
            _preencher_session_state(bundle, extraido)
            st.rerun()
        except Exception as exc:
            st.error(f"Erro na extração: {exc}")

    st.markdown("**Specs** (preenchidas pela extração; ajuste se algo estiver errado)")
    specs = _render_formulario_specs(bundle)

    # Alerta específico: sem fabricante o modelo tende a superestimar preço.
    # (Não bloqueia a análise; só avisa.)
    if "fabricante" in bundle["cat"] and not specs.get("fabricante"):
        st.warning(
            "⚠️ **Fabricante em branco.** O modelo aprendeu que produtos sem "
            "fabricante identificado tendem a ser kits enterprise/obscuros e mais "
            "caros. Preencher o fabricante (ex.: `Kingston`, `Corsair`) melhora "
            "muito a precisão da previsão."
        )

    st.markdown("")
    if st.button("Analisar", type="primary", disabled=(preco_real <= 0 or not nome.strip())):
        specs_modelo = {k: v for k, v in specs.items()
                        if k in bundle["num"] + bundle["cat"]}

        # Se o usuário não clicou em "Extrair specs" e não preencheu nada
        # manualmente, faz a extração agora e re-renderiza com o formulário
        # populado (evita silenciosamente prever com a mediana do treino).
        tudo_vazio = all(v is None for v in specs_modelo.values())
        if tudo_vazio:
            try:
                extraido = extrair_features_produto(nome, cat)
                tem_algo = any(
                    v is not None and not (isinstance(v, float) and pd.isna(v))
                    for k, v in extraido.items()
                    if k in bundle["num"] + bundle["cat"]
                )
                if tem_algo:
                    _preencher_session_state(bundle, extraido)
                    st.session_state[f"analisar_pendente_{cat}"] = True
                    st.rerun()
                else:
                    st.warning(
                        "⚠️ A extração automática não encontrou nenhuma spec "
                        "no nome. Preencha manualmente ao menos uma feature "
                        "para uma previsão significativa."
                    )
                    return
            except Exception as exc:
                st.error(f"Erro ao extrair specs do nome: {exc}")
                return

        with st.spinner("Calculando..."):
            resultado = prever_preco_e_faixa(specs_modelo, bundle)
        _render_resultado(preco_real, resultado, bundle, specs_usadas=specs_modelo)

    # Segunda passada após o rerun da auto-extração: o formulário já está
    # populado, mostra o resultado.
    elif st.session_state.pop(f"analisar_pendente_{cat}", False):
        specs_modelo = {k: v for k, v in specs.items()
                        if k in bundle["num"] + bundle["cat"]}
        st.info("🪄 Specs extraídas automaticamente do nome do produto.")
        with st.spinner("Calculando..."):
            resultado = prever_preco_e_faixa(specs_modelo, bundle)
        _render_resultado(preco_real, resultado, bundle, specs_usadas=specs_modelo)


# ---------------------------------------------------------------------------
# Placeholders das outras abas
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _computar_precos_justos_cache(catalogo_hash: int, cat: str, _bundle: dict,
                                   df_cat: pd.DataFrame) -> pd.DataFrame:
    """Wrapper cacheado — chave é (hash do catálogo, categoria).

    Se `_bundle['usa_log']`, destransforma log→R$ (faixa fica assimétrica).
    """
    df = df_cat.copy()
    campos = _bundle["num"] + _bundle["cat"]
    for c in campos:
        if c not in df.columns:
            df[c] = np.nan
    X = df[campos].copy()
    for c in _bundle["cat"]:
        if c in X.columns:
            X[c] = X[c].where(X[c].isna(), X[c].astype(str))

    Xt = _bundle["preproc"].transform(X)
    pred_raw = _bundle["modelo_media"].predict(Xt)
    variancia = np.maximum(_bundle["modelo_variancia"].predict(Xt), 1e-6)
    escala = np.sqrt(variancia)

    usa_log = _bundle.get("usa_log", False)
    if usa_log:
        preco_justo = np.expm1(pred_raw)
        df["faixa_baixo"] = np.maximum(np.expm1(pred_raw - _bundle["q_norm"] * escala), 0.0)
        df["faixa_alto"]  = np.expm1(pred_raw + _bundle["q_norm"] * escala)
    else:
        preco_justo = pred_raw
        df["faixa_baixo"] = np.maximum(pred_raw - _bundle["q_norm"] * escala, 0.0)
        df["faixa_alto"]  = pred_raw + _bundle["q_norm"] * escala

    df["preco_justo"] = preco_justo
    df["desvio_pct"] = (df["preco_pix"] - df["preco_justo"]) / df["preco_justo"] * 100
    df["veredito"] = df.apply(
        lambda row: classificar(row["preco_pix"], row["faixa_baixo"], row["faixa_alto"])[0],
        axis=1,
    )
    return df

def _computar_precos_justos(df_cat: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    # id estável do catálogo por categoria (usado como parte da chave do cache)
    catalogo_hash = hash((len(df_cat), tuple(df_cat["id"].head(20)) if "id" in df_cat.columns else ()))
    cat = bundle["categoria"]
    return _computar_precos_justos_cache(catalogo_hash, cat, bundle, df_cat)


def _filtros_por_categoria(df_cat: pd.DataFrame, cat: str) -> pd.DataFrame:
    """Renderiza controles de filtro específicos da categoria e devolve
    o dataframe filtrado.
    """
    st.markdown("**Filtros** — deixe vazio para não filtrar")

    # Filtros comuns
    with st.expander("Preço e coleta", expanded=True):
        c1, c2, c3 = st.columns(3)
        preco_max = c1.number_input(
            "Preço máximo (R$)", min_value=0.0, value=0.0, step=100.0,
            help="0 = sem limite",
        )
        veredito_sel = c2.multiselect(
            "Veredito",
            options=["🎉 Oferta", "✅ Justo", "⚠️ Caro"],
            default=["🎉 Oferta"],
        )
        coletas_disp = sorted(df_cat["data_coleta"].dropna().unique(), reverse=True)
        coletas_sel = c3.multiselect(
            "Coletas",
            options=coletas_disp,
            default=coletas_disp,
        )

    # Filtros específicos da categoria (numéricos com min/max, categóricos com multiselect)
    specs_filtros = _filtros_specs_por_cat(df_cat, cat)

    # Aplica filtros
    df = df_cat.copy()
    if preco_max > 0:
        df = df[df["preco_pix"] <= preco_max]
    if veredito_sel:
        df = df[df["veredito"].isin(veredito_sel)]
    if coletas_sel:
        df = df[df["data_coleta"].isin(coletas_sel)]
    for coluna, valores in specs_filtros.items():
        if valores is None:
            continue
        if isinstance(valores, tuple):
            lo, hi, ativo = valores
            if ativo:
                # slider foi mexido -> filtro rigoroso (exclui NaN)
                df = df[df[coluna].between(lo, hi)]
            # slider intocado -> não filtra nada (deixa NaN passar)
        else:
            df = df[df[coluna].isin(valores)]

    return df


# Filtros específicos por categoria — (feature, tipo) onde tipo é 'num' ou 'cat'
FILTROS_POR_CAT = {
    "ram": [("ram_gb", "num"), ("ram_mhz", "num"), ("ram_geracao", "cat"), ("fabricante", "cat")],
    "cpu": [("cpu_socket", "cat"), ("cpu_marca", "cat"), ("cpu_serie", "cat"), ("fabricante", "cat")],
    "gpu": [("gpu_vram_gb", "num"), ("gpu_modelo", "cat"), ("gpu_marca_chip", "cat"), ("fabricante", "cat")],
    "ssd": [("ssd_capacidade_gb", "num"), ("ssd_interface", "cat"), ("ssd_geracao_pcie", "cat"), ("fabricante", "cat")],
    "fonte": [("fonte_wattagem", "num"), ("fonte_certificacao", "cat"), ("fonte_modular", "cat"), ("fabricante", "cat")],
    "placa_mae": [("mobo_socket", "cat"), ("mobo_ddr", "cat"), ("mobo_form_factor", "cat"), ("fabricante", "cat")],
}


def _filtros_specs_por_cat(df_cat: pd.DataFrame, cat: str) -> dict:
    """Renderiza filtros de spec e retorna dict {coluna: valores_selecionados}."""
    filtros = FILTROS_POR_CAT.get(cat, [])
    resultado = {}
    if not filtros:
        return resultado

    with st.expander("Specs", expanded=True):
        cols = st.columns(2)
        for i, (coluna, tipo) in enumerate(filtros):
            if coluna not in df_cat.columns:
                continue
            with cols[i % 2]:
                serie = df_cat[coluna].dropna()
                if serie.empty:
                    continue
                if tipo == "num":
                    lo, hi = float(serie.min()), float(serie.max())
                    if lo == hi:
                        st.caption(f"{label_de(coluna)}: {lo:.0f} (único valor)")
                        continue
                    valores = st.slider(
                        label_de(coluna),
                        min_value=lo, max_value=hi, value=(lo, hi),
                        key=f"filtro_{cat}_{coluna}",
                    )
                    # ativo = usuário mexeu nos extremos (foi diferente do default)
                    ativo = valores[0] > lo or valores[1] < hi
                    resultado[coluna] = (valores[0], valores[1], ativo)
                else:
                    opcoes = sorted(serie.astype(str).unique())
                    sel = st.multiselect(
                        label_de(coluna),
                        options=opcoes,
                        default=[],
                        key=f"filtro_{cat}_{coluna}",
                    )
                    resultado[coluna] = sel if sel else None
    return resultado


def _modo_ranking_ofertas(bundles: dict, catalogo: pd.DataFrame) -> None:
    """Modo A: escolhe categoria, filtra, mostra ranking por desvio_pct."""
    cats_disponiveis = [c for c in CATEGORIAS if c in bundles]

    col_cat, col_op1, col_op2 = st.columns([2, 1, 1])
    with col_cat:
        cat = st.selectbox(
            "Categoria",
            cats_disponiveis,
            format_func=lambda c: LABEL_CATEGORIA[c],
            key="rank_cat",
        )
    with col_op1:
        st.markdown("")  # espaçamento
        excluir_ngen = st.checkbox(
            "Excluir não-genuínos",
            value=True,
            help="Remove adaptadores, coolers, cabos e outros produtos que o "
                 "site classifica junto mas não são o item principal da categoria.",
            key=f"ngen_{cat if 'cat' in dir() else ''}",
        )
    with col_op2:
        st.markdown("")
        so_recente = st.checkbox(
            "Só coleta mais recente",
            value=True,
            help="Mantém uma linha por produto (id). Desmarque para ver "
                 "histórico de preço em todas as coletas.",
            key=f"recente_{cat if 'cat' in dir() else ''}",
        )
    bundle = bundles[cat]

    # Filtra o catálogo por categoria
    df_cat = catalogo[catalogo["categoria_key"] == cat].copy()
    if df_cat.empty:
        st.warning(f"Nenhum produto de {LABEL_CATEGORIA[cat]} no catálogo.")
        return

    # Aplica filtros de qualidade (antes do compute — sinal mais limpo)
    total_antes = len(df_cat)
    if excluir_ngen and "eh_genuino" in df_cat.columns:
        df_cat = df_cat[df_cat["eh_genuino"]]
    if "eh_genuino" not in df_cat.columns and excluir_ngen:
        st.caption(
            "⚠️ Coluna `eh_genuino` não está no catálogo — re-rode "
            "`python salvar_catalogo.py` para habilitar esse filtro."
        )
    if so_recente and "id" in df_cat.columns and "data_coleta" in df_cat.columns:
        df_cat = (df_cat.sort_values("data_coleta")
                        .drop_duplicates(subset=["id"], keep="last"))

    excluidos = total_antes - len(df_cat)
    if excluidos > 0:
        st.caption(f"↳ {excluidos:,} linhas removidas pelos filtros de qualidade "
                   f"({total_antes:,} → {len(df_cat):,}).")

    with st.spinner("Calculando preços justos..."):
        df_cat = _computar_precos_justos(df_cat, bundle)

    df_filtrado = _filtros_por_categoria(df_cat, cat)

    # Ordenação: menor desvio_pct primeiro (maiores ofertas)
    df_filtrado = df_filtrado.sort_values("desvio_pct", ascending=True)

    st.markdown(f"**{len(df_filtrado):,} produtos** após filtros (de {len(df_cat):,} totais)")

    if df_filtrado.empty:
        st.info("Nenhum produto passou pelos filtros. Afrouxe alguma restrição.")
        return

    # Tabela: nome, veredito, preço observado, preço justo, desvio, faixa, data
    cols_exibir = {
        "nome": "Produto",
        "preco_pix": "Preço observado",
        "preco_justo": "Preço justo",
        "desvio_pct": "Desvio (%)",
        "faixa_baixo": "Faixa baixo",
        "faixa_alto": "Faixa alto",
        "veredito": "Veredito",
        "data_coleta": "Coleta",
    }
    cols_disponiveis = [c for c in cols_exibir if c in df_filtrado.columns]

    tabela = df_filtrado[cols_disponiveis].head(200).rename(columns=cols_exibir)

    st.dataframe(
        tabela,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Preço observado": st.column_config.NumberColumn(format="R$ %.2f"),
            "Preço justo":     st.column_config.NumberColumn(format="R$ %.2f"),
            "Faixa baixo":     st.column_config.NumberColumn(format="R$ %.2f"),
            "Faixa alto":      st.column_config.NumberColumn(format="R$ %.2f"),
            "Desvio (%)":      st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )
    if len(df_filtrado) > 200:
        st.caption(f"Mostrando os 200 primeiros (de {len(df_filtrado):,}).")


def _picker_cpu_ancora(bundles: dict, catalogo: pd.DataFrame) -> Optional[pd.Series]:
    """Renderiza o seletor de CPU-âncora e retorna a linha selecionada."""
    df_cpu = catalogo[catalogo["categoria_key"] == "cpu"].copy()
    if "eh_genuino" in df_cpu.columns:
        df_cpu = df_cpu[df_cpu["eh_genuino"]]
    # só coleta mais recente por CPU (âncora única)
    if "id" in df_cpu.columns and "data_coleta" in df_cpu.columns:
        df_cpu = (df_cpu.sort_values("data_coleta")
                        .drop_duplicates(subset=["id"], keep="last"))

    if df_cpu.empty:
        st.warning("Nenhuma CPU no catálogo.")
        return None

    # Filtros para achar a CPU mais fácil
    c1, c2, c3 = st.columns(3)
    marcas = c1.multiselect(
        "Marca da CPU",
        options=sorted(df_cpu["cpu_marca"].dropna().astype(str).unique()),
        default=[],
        key="anc_cpu_marca",
    )
    if marcas:
        df_cpu = df_cpu[df_cpu["cpu_marca"].isin(marcas)]

    sockets = c2.multiselect(
        "Socket",
        options=sorted(df_cpu["cpu_socket"].dropna().astype(str).unique()),
        default=[],
        key="anc_cpu_socket",
    )
    if sockets:
        df_cpu = df_cpu[df_cpu["cpu_socket"].isin(sockets)]

    busca = c3.text_input("Buscar no nome", key="anc_cpu_busca",
                          placeholder="ex.: 7600X, i5-13400, Ryzen 7")
    if busca.strip():
        df_cpu = df_cpu[df_cpu["nome"].str.contains(busca.strip(), case=False, na=False)]

    if df_cpu.empty:
        st.info("Nenhuma CPU passou pelos filtros — afrouxe alguma restrição.")
        return None

    # Mostra as CPUs candidatas
    opcoes = df_cpu[["nome", "preco_pix", "cpu_socket", "cpu_ddr_suportado"]].copy()
    opcoes["rotulo"] = opcoes.apply(
        lambda r: f"{r['nome']}  •  {r['cpu_socket'] or '?'}  •  R$ {r['preco_pix']:,.2f}",
        axis=1,
    )
    escolha_rotulo = st.selectbox(
        f"CPU-âncora ({len(df_cpu):,} candidatas)",
        options=opcoes["rotulo"].tolist(),
        key="anc_cpu_escolha",
    )
    idx = opcoes[opcoes["rotulo"] == escolha_rotulo].index[0]
    return df_cpu.loc[idx]


# ---------------------------------------------------------------------------
# Regras de compatibilidade
# ---------------------------------------------------------------------------

MARGEM_FONTE_W = 150   # margem para HDDs, coolers, ventoinhas


def _ddrs_suportados_pela_cpu(cpu_row: pd.Series) -> list[str]:
    """Retorna a lista de gerações DDR suportadas pela CPU (ex.: ['DDR4', 'DDR5'])."""
    raw = cpu_row.get("cpu_ddr_suportado")
    if pd.isna(raw) or not raw:
        return []
    # SOCKET_DDR retorna coisas tipo "DDR4/DDR5"
    return [x.strip() for x in str(raw).split("/") if x.strip()]


def _tdp_estimado(cpu_row: pd.Series, tdp_gpu: float = 0.0) -> float:
    """TDP total estimado (CPU + GPU) + margem. Se CPU TDP faltar, assume 105W."""
    tdp_cpu = cpu_row.get("cpu_tdp_w")
    if pd.isna(tdp_cpu):
        tdp_cpu = 105  # default conservador para CPU moderna sem TDP identificado
    return float(tdp_cpu) + float(tdp_gpu) + MARGEM_FONTE_W


def _filtrar_placas_compativeis(cpu_row: pd.Series, catalogo: pd.DataFrame,
                                  form_factors: list[str]) -> pd.DataFrame:
    """Placas-mãe com mesmo socket + form factor selecionado."""
    df = catalogo[catalogo["categoria_key"] == "placa_mae"].copy()
    if "eh_genuino" in df.columns:
        df = df[df["eh_genuino"]]
    socket = cpu_row.get("cpu_socket")
    if socket:
        df = df[df["mobo_socket"].astype(str).str.upper() == str(socket).upper()]
    if form_factors:
        df = df[df["mobo_form_factor"].isin(form_factors)]
    return df


def _filtrar_ram_compativel(cpu_row: pd.Series, catalogo: pd.DataFrame) -> pd.DataFrame:
    df = catalogo[catalogo["categoria_key"] == "ram"].copy()
    if "eh_genuino" in df.columns:
        df = df[df["eh_genuino"]]
    ddrs = _ddrs_suportados_pela_cpu(cpu_row)
    if ddrs:
        # normaliza para comparar ("Ddr4" vs "DDR4" etc.)
        df = df[df["ram_geracao"].astype(str).str.upper().isin([d.upper() for d in ddrs])]
    return df


def _filtrar_fonte_suficiente(cpu_row: pd.Series, catalogo: pd.DataFrame,
                                tdp_gpu: float) -> pd.DataFrame:
    df = catalogo[catalogo["categoria_key"] == "fonte"].copy()
    if "eh_genuino" in df.columns:
        df = df[df["eh_genuino"]]
    exigido = _tdp_estimado(cpu_row, tdp_gpu)
    if "fonte_wattagem" in df.columns:
        df = df[df["fonte_wattagem"] >= exigido]
    return df, exigido


def _mostrar_ranking_compat(nome_secao: str, df_cat: pd.DataFrame,
                             bundle: dict, catalogo_total: int,
                             cols_extras: list[str] = None,
                             topo: int = 10) -> None:
    """Mostra ranking de ofertas de uma categoria com preço justo/faixa/desvio."""
    st.markdown(f"### {nome_secao}")
    if df_cat.empty:
        st.info(f"Nenhum produto compatível encontrado.")
        return

    # só coleta mais recente por id
    if "id" in df_cat.columns and "data_coleta" in df_cat.columns:
        df_cat = (df_cat.sort_values("data_coleta")
                        .drop_duplicates(subset=["id"], keep="last"))

    df_cat = _computar_precos_justos(df_cat, bundle)
    df_cat = df_cat.sort_values("desvio_pct", ascending=True)

    cols_extras = cols_extras or []
    cols_exibir = {"nome": "Produto", "preco_pix": "Preço",
                    "preco_justo": "Preço justo", "desvio_pct": "Desvio (%)",
                    "veredito": "Veredito"}
    for c in cols_extras:
        if c in df_cat.columns:
            cols_exibir[c] = label_de(c)

    tabela = df_cat[list(cols_exibir.keys())].head(topo).rename(columns=cols_exibir)
    st.dataframe(
        tabela,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Preço":       st.column_config.NumberColumn(format="R$ %.2f"),
            "Preço justo": st.column_config.NumberColumn(format="R$ %.2f"),
            "Desvio (%)":  st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )
    st.caption(f"{len(df_cat):,} compatíveis (mostrando {min(topo, len(df_cat))}).")


def _modo_compatibilidade(bundles: dict, catalogo: pd.DataFrame) -> None:
    """Modo B: escolhe uma CPU-âncora e mostra placa-mãe, RAM, GPU e fonte compatíveis."""
    st.markdown(
        "Escolha uma **CPU-âncora** — o app filtra placas-mãe, RAM, GPUs e "
        "fontes compatíveis (socket, DDR, wattagem, form factor) e ranqueia "
        "cada categoria por **oferta**."
    )

    faltando = [c for c in ["cpu", "placa_mae", "ram", "gpu", "fonte"] if c not in bundles]
    if faltando:
        st.warning(f"Modelos ausentes: {faltando}. Rode o notebook v2_02.")
        return

    # 1. Escolha da âncora
    st.markdown("#### 1. Escolha a CPU")
    cpu_row = _picker_cpu_ancora(bundles, catalogo)
    if cpu_row is None:
        return

    ddrs = _ddrs_suportados_pela_cpu(cpu_row)
    socket = cpu_row.get("cpu_socket") or "?"

    c1, c2, c3 = st.columns(3)
    c1.metric("Socket", str(socket))
    c2.metric("DDR suportado", " / ".join(ddrs) if ddrs else "?")
    c3.metric("TDP CPU (W)", f"{cpu_row.get('cpu_tdp_w', '?'):.0f}"
              if pd.notna(cpu_row.get("cpu_tdp_w")) else "não identificado")

    # 2. Preferências opcionais
    st.markdown("#### 2. Preferências (opcionais)")
    pc1, pc2 = st.columns(2)
    with pc1:
        form_factors = st.multiselect(
            "Form factor da placa-mãe",
            options=sorted(catalogo["mobo_form_factor"].dropna().astype(str).unique()),
            default=[],
            key="compat_ff",
        )
    with pc2:
        gpu_tdp_manual = st.number_input(
            "TDP da GPU (W) para dimensionar fonte",
            min_value=0, max_value=800, value=200, step=10,
            help="Se você não sabe, deixe 200W (RTX 4070 / RX 7800 XT são ~200W).",
            key="compat_gpu_tdp",
        )

    st.markdown("---")

    # 3. Placas-mãe compatíveis
    df_placas = _filtrar_placas_compativeis(cpu_row, catalogo, form_factors)
    _mostrar_ranking_compat(
        f"🧩 Placas-mãe compatíveis (socket {socket})",
        df_placas, bundles["placa_mae"], len(catalogo),
        cols_extras=["mobo_chipset", "mobo_ddr", "mobo_form_factor"],
    )

    # 4. RAM compatível
    df_ram = _filtrar_ram_compativel(cpu_row, catalogo)
    ddrs_txt = " / ".join(ddrs) if ddrs else "qualquer"
    _mostrar_ranking_compat(
        f"💾 RAM compatível ({ddrs_txt})",
        df_ram, bundles["ram"], len(catalogo),
        cols_extras=["ram_gb", "ram_mhz", "ram_geracao"],
    )

    # 5. Fonte suficiente
    df_fonte, exigido = _filtrar_fonte_suficiente(cpu_row, catalogo, gpu_tdp_manual)
    tdp_cpu_int = int(cpu_row.get("cpu_tdp_w")) if pd.notna(cpu_row.get("cpu_tdp_w")) else 105
    _mostrar_ranking_compat(
        f"⚡ Fontes ≥ {exigido:.0f}W  (CPU {tdp_cpu_int}W + "
        f"GPU {gpu_tdp_manual}W + margem {MARGEM_FONTE_W}W)",
        df_fonte, bundles["fonte"], len(catalogo),
        cols_extras=["fonte_wattagem", "fonte_certificacao", "fonte_modular"],
    )

    # 6. GPU (qualquer)
    df_gpu = catalogo[catalogo["categoria_key"] == "gpu"].copy()
    if "eh_genuino" in df_gpu.columns:
        df_gpu = df_gpu[df_gpu["eh_genuino"]]
    _mostrar_ranking_compat(
        "🎮 GPUs (compatibilidade PCIe é universal — todas listadas)",
        df_gpu, bundles["gpu"], len(catalogo),
        cols_extras=["gpu_modelo", "gpu_vram_gb", "gpu_tdp_w"],
    )


def aba_pecas_compativeis(bundles: dict, catalogo: Optional[pd.DataFrame]) -> None:
    st.header("Peças compatíveis")
    if catalogo is None:
        st.warning(
            "Esta aba precisa do `modelos/catalogo.parquet`, que ainda não "
            "foi gerado. No terminal, rode:\n\n"
            "```\npython salvar_catalogo.py\n```"
        )
        return

    st.markdown(
        "Explore o catálogo em dois modos: **Ranking de ofertas** para "
        "encontrar as melhores oportunidades dentro de uma categoria, "
        "**Compatibilidade** para achar peças que trabalham juntas."
    )

    modo = st.radio(
        "Modo",
        options=["Ranking de ofertas", "Compatibilidade a partir de uma CPU"],
        horizontal=True,
        key="modo_pecas",
    )

    st.markdown("---")

    if modo == "Ranking de ofertas":
        _modo_ranking_ofertas(bundles, catalogo)
    else:
        _modo_compatibilidade(bundles, catalogo)


# ---------------------------------------------------------------------------
# Aba 3 — Build por orçamento
# ---------------------------------------------------------------------------

# Presets: percentuais do orçamento por categoria.
# Somam 100%. Refletem uma alocação típica de mercado.
PRESETS_BUILD = {
    "Gamer":  {"cpu": 18, "placa_mae": 12, "ram": 8,  "gpu": 40, "ssd": 8,  "fonte": 14},
    "Trabalho": {"cpu": 30, "placa_mae": 15, "ram": 15, "gpu": 15, "ssd": 15, "fonte": 10},
    "Básico": {"cpu": 22, "placa_mae": 14, "ram": 14, "gpu": 20, "ssd": 14, "fonte": 16},
}

ORDEM_MONTAGEM = ["cpu", "placa_mae", "ram", "gpu", "ssd", "fonte"]


def _escolher_peca(df_cat: pd.DataFrame, bundle: dict, teto: float) -> Optional[pd.Series]:
    """Escolhe UMA peça: 'melhor caro-mas-justo' com fallback para 'melhor barato'.

    Estratégia C:
      1. Filtra por preço ≤ teto.
      2. Se houver peças com veredito 'Justo', escolhe a mais cara delas.
      3. Se não houver, escolhe a com melhor desvio_pct (mais negativo = maior oferta).
    """
    if df_cat.empty:
        return None
    df = df_cat[df_cat["preco_pix"] <= teto].copy()
    if df.empty:
        return None

    # 'Justo' = preço dentro da faixa conformal
    justos = df[df["veredito"] == "✅ Justo"]
    if not justos.empty:
        # dentro do orçamento, o mais caro tende a ser o melhor
        escolha = justos.nlargest(1, "preco_pix").iloc[0]
    else:
        # nada 'justo' — pega o menor desvio_pct (maior oferta)
        escolha = df.nsmallest(1, "desvio_pct").iloc[0]
    return escolha


def _montar_build(orcamento_total: float, percentuais: dict,
                   bundles: dict, catalogo: pd.DataFrame,
                   form_factors: list[str]) -> dict:
    """Monta a build peça por peça, respeitando compatibilidade em cascata.

    Ordem: CPU primeiro (define socket/DDR). Depois placa-mãe (socket), RAM
    (DDR), GPU (livre), SSD (livre), fonte (após saber TDP CPU+GPU).
    """
    build = {}     # cat -> Series com a peça escolhida
    logs = []      # mensagens para o usuário

    # Prepara catálogo por categoria com preço justo/veredito
    catalogos_com_preco = {}
    for cat in ORDEM_MONTAGEM:
        if cat not in bundles:
            continue
        df = catalogo[catalogo["categoria_key"] == cat].copy()
        if "eh_genuino" in df.columns:
            df = df[df["eh_genuino"]]
        # só coleta mais recente por id
        if "id" in df.columns and "data_coleta" in df.columns:
            df = df.sort_values("data_coleta").drop_duplicates(subset=["id"], keep="last")
        df = _computar_precos_justos(df, bundles[cat])
        catalogos_com_preco[cat] = df

    # --- CPU ---
    df_cpu = catalogos_com_preco["cpu"]
    teto_cpu = orcamento_total * percentuais["cpu"] / 100
    cpu = _escolher_peca(df_cpu, bundles["cpu"], teto_cpu)
    if cpu is None:
        logs.append(f"❌ Não foi possível achar CPU até R$ {teto_cpu:,.0f}. Aumente o orçamento.")
        return {"build": {}, "logs": logs}
    build["cpu"] = cpu
    socket_cpu = cpu.get("cpu_socket")
    ddrs_cpu = _ddrs_suportados_pela_cpu(cpu)
    logs.append(
        f"✅ CPU escolhida: {cpu['nome'][:60]}... "
        f"(socket **{socket_cpu or 'não identificado'}**, "
        f"DDR **{'/'.join(ddrs_cpu) if ddrs_cpu else 'não identificado'}**)"
    )

    # --- Placa-mãe (compatível com socket da CPU) ---
    df_mobo = catalogos_com_preco["placa_mae"]
    socket = cpu.get("cpu_socket")
    if pd.isna(socket) or not socket:
        logs.append(
            "⚠️ Socket da CPU não identificado no nome — pulando o filtro "
            "de socket da placa-mãe. **A compatibilidade não está garantida.** "
            "Verifique manualmente."
        )
        socket = None
    else:
        df_mobo = df_mobo[df_mobo["mobo_socket"].astype(str).str.upper() == str(socket).upper()]
    if form_factors:
        df_mobo = df_mobo[df_mobo["mobo_form_factor"].isin(form_factors)]
    teto_mobo = orcamento_total * percentuais["placa_mae"] / 100
    mobo = _escolher_peca(df_mobo, bundles["placa_mae"], teto_mobo)
    if mobo is None:
        if socket:
            logs.append(f"❌ Nenhuma placa-mãe socket {socket} até R$ {teto_mobo:,.0f}.")
        else:
            logs.append(f"❌ Nenhuma placa-mãe até R$ {teto_mobo:,.0f}.")
    else:
        build["placa_mae"] = mobo
        socket_mobo = mobo.get("mobo_socket", "?")
        aviso = ""
        if socket and str(socket_mobo).upper() != str(socket).upper():
            aviso = f" ⚠️ ATENÇÃO: socket {socket_mobo} ≠ {socket}"
        logs.append(f"✅ Placa-mãe: {mobo['nome'][:60]}... (socket {socket_mobo}){aviso}")

    # --- RAM (DDR compatível) ---
    df_ram = catalogos_com_preco["ram"]
    ddrs = _ddrs_suportados_pela_cpu(cpu)
    if ddrs:
        df_ram = df_ram[df_ram["ram_geracao"].astype(str).str.upper().isin([d.upper() for d in ddrs])]
    else:
        logs.append(
            "⚠️ DDR suportado pela CPU não identificado — pulando filtro. "
            "Verifique manualmente se a RAM é compatível."
        )
    teto_ram = orcamento_total * percentuais["ram"] / 100
    ram = _escolher_peca(df_ram, bundles["ram"], teto_ram)
    if ram is None:
        logs.append(f"❌ Nenhuma RAM até R$ {teto_ram:,.0f}.")
    else:
        build["ram"] = ram
        ram_ddr = ram.get("ram_geracao", "?")
        aviso = ""
        if ddrs and str(ram_ddr).upper() not in [d.upper() for d in ddrs]:
            aviso = f" ⚠️ ATENÇÃO: RAM {ram_ddr} pode não ser compatível"
        logs.append(f"✅ RAM: {ram['nome'][:60]}... ({ram_ddr}){aviso}")

    # --- GPU (livre) ---
    df_gpu = catalogos_com_preco["gpu"]
    teto_gpu = orcamento_total * percentuais["gpu"] / 100
    gpu = _escolher_peca(df_gpu, bundles["gpu"], teto_gpu)
    if gpu is None:
        logs.append(f"❌ Nenhuma GPU até R$ {teto_gpu:,.0f}.")
    else:
        build["gpu"] = gpu
        logs.append(f"✅ GPU: {gpu['nome'][:60]}...")

    # --- SSD (livre) ---
    df_ssd = catalogos_com_preco["ssd"]
    teto_ssd = orcamento_total * percentuais["ssd"] / 100
    ssd = _escolher_peca(df_ssd, bundles["ssd"], teto_ssd)
    if ssd is None:
        logs.append(f"❌ Nenhum SSD até R$ {teto_ssd:,.0f}.")
    else:
        build["ssd"] = ssd
        logs.append(f"✅ SSD: {ssd['nome'][:60]}...")

    # --- Fonte (wattagem >= TDP CPU + TDP GPU + margem) ---
    df_fonte = catalogos_com_preco["fonte"]
    tdp_cpu = float(cpu.get("cpu_tdp_w")) if pd.notna(cpu.get("cpu_tdp_w")) else 105
    tdp_gpu = float(gpu.get("gpu_tdp_w")) if (gpu is not None and pd.notna(gpu.get("gpu_tdp_w"))) else 200
    exigido = tdp_cpu + tdp_gpu + MARGEM_FONTE_W
    if "fonte_wattagem" in df_fonte.columns:
        df_fonte = df_fonte[df_fonte["fonte_wattagem"] >= exigido]
    teto_fonte = orcamento_total * percentuais["fonte"] / 100
    fonte = _escolher_peca(df_fonte, bundles["fonte"], teto_fonte)
    if fonte is None:
        logs.append(f"❌ Nenhuma fonte ≥ {exigido:.0f}W até R$ {teto_fonte:,.0f}. "
                    f"Considere aumentar o orçamento ou baixar a GPU.")
    else:
        build["fonte"] = fonte
        logs.append(f"✅ Fonte: {fonte['nome'][:60]}... ({int(fonte.get('fonte_wattagem', 0))}W)")

    return {"build": build, "logs": logs, "exigido_fonte_w": exigido}


def aba_build_orcamento(bundles: dict, catalogo: Optional[pd.DataFrame]) -> None:
    st.header("Build por orçamento")

    if catalogo is None:
        st.warning(
            "Esta aba precisa do `modelos/catalogo.parquet`. Rode "
            "`python salvar_catalogo.py`."
        )
        return
    faltando = [c for c in ORDEM_MONTAGEM if c not in bundles]
    if faltando:
        st.warning(f"Modelos ausentes: {faltando}. Rode o notebook v2_02.")
        return

    st.markdown(
        "Informe seu orçamento total. O app aloca por categoria (CPU, placa-mãe, "
        "RAM, GPU, SSD, fonte), escolhe a **melhor peça 'justa'** dentro do "
        "teto de cada uma (ou a maior oferta se nenhuma for 'justa'), e "
        "respeita compatibilidade (socket, DDR, wattagem)."
    )

    # Controles
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        orcamento = st.number_input(
            "Orçamento total (R$)",
            min_value=1500.0, max_value=50000.0, value=5000.0, step=500.0,
        )
    with c2:
        preset = st.selectbox("Perfil", options=list(PRESETS_BUILD.keys()))
    with c3:
        form_factors = st.multiselect(
            "Form factor da placa-mãe (opcional)",
            options=sorted(catalogo["mobo_form_factor"].dropna().astype(str).unique()),
            default=[],
        )

    percentuais = PRESETS_BUILD[preset].copy()

    # Editor avançado dos percentuais
    with st.expander("Ajustar alocação (avançado)", expanded=False):
        pcols = st.columns(6)
        for i, cat in enumerate(ORDEM_MONTAGEM):
            with pcols[i]:
                percentuais[cat] = st.slider(
                    LABEL_CATEGORIA[cat], min_value=0, max_value=60,
                    value=percentuais[cat], step=1, key=f"pct_{cat}",
                )
        soma = sum(percentuais.values())
        if soma != 100:
            st.warning(f"Percentuais somam {soma}%, deveriam somar 100%. "
                       "O orçamento efetivo pode ficar acima ou abaixo do total.")

    if not st.button("🛒 Montar build", type="primary"):
        return

    with st.spinner("Montando build..."):
        resultado = _montar_build(orcamento, percentuais, bundles, catalogo, form_factors)

    build = resultado["build"]
    logs = resultado["logs"]

    # Log das decisões
    with st.expander("Decisões da montagem", expanded=False):
        for msg in logs:
            st.markdown(f"- {msg}")

    if not build:
        st.error("Não foi possível montar a build com esses parâmetros.")
        return

    # Tabela final
    st.subheader("Build final")
    linhas = []
    total = 0.0
    total_justo = 0.0
    for cat in ORDEM_MONTAGEM:
        if cat not in build:
            linhas.append({
                "Categoria": LABEL_CATEGORIA[cat],
                "Produto": "❌ Não encontrado",
                "Preço": None, "Preço justo": None, "Desvio (%)": None, "Veredito": "—",
            })
            continue
        row = build[cat]
        linhas.append({
            "Categoria": LABEL_CATEGORIA[cat],
            "Produto": row["nome"],
            "Preço": float(row["preco_pix"]),
            "Preço justo": float(row["preco_justo"]),
            "Desvio (%)": float(row["desvio_pct"]),
            "Veredito": row["veredito"],
        })
        total += float(row["preco_pix"])
        total_justo += float(row["preco_justo"])

    st.dataframe(
        pd.DataFrame(linhas),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Preço":       st.column_config.NumberColumn(format="R$ %.2f"),
            "Preço justo": st.column_config.NumberColumn(format="R$ %.2f"),
            "Desvio (%)":  st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )

    # KPIs finais
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total da build", f"R$ {total:,.2f}",
               delta=f"R$ {total - orcamento:+,.0f} vs orçamento")
    mc2.metric("Total 'justo' previsto", f"R$ {total_justo:,.2f}")
    economia = total_justo - total
    mc3.metric("Economia estimada", f"R$ {economia:,.2f}",
               delta=f"{economia / total_justo * 100:+.1f}%" if total_justo else None,
               delta_color="normal")
    mc4.metric("Peças montadas", f"{len(build)}/{len(ORDEM_MONTAGEM)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("💾 KaBuM — Preço Justo")
    st.caption(
        "Modelos de preço justo por categoria (RAM, CPU, GPU, SSD, Fonte, Placa-mãe) "
        "com faixa de 90% e detector de ofertas."
    )

    bundles = carregar_bundles(MODEL_DIR)
    catalogo = carregar_catalogo(MODEL_DIR)
    render_sidebar(bundles, catalogo)

    aba1, aba2, aba3 = st.tabs([
        "🔍 Analisar produto",
        "🧩 Peças compatíveis",
        "🛒 Build por orçamento",
    ])
    with aba1: aba_analisar_produto(bundles)
    with aba2: aba_pecas_compativeis(bundles, catalogo)
    with aba3: aba_build_orcamento(bundles, catalogo)


if __name__ == "__main__":
    main()
