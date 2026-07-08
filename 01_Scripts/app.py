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
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from features import FEATURES_POR_CATEGORIA, extrair_features_produto

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

MODEL_DIR = pathlib.Path(__file__).parent / "modelos"
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

    A mesma lógica de inferência do notebook v2_02: transforma o input com o
    pré-processador do bundle, prevê média (`modelo_media`) e variância
    (`modelo_variancia`), aplica `q_norm` para a faixa e aplica o clip em
    zero (preço não pode ser negativo).
    """
    X1 = pd.DataFrame([specs])
    # garante todas as colunas esperadas pelo pré-processador
    for c in bundle["num"] + bundle["cat"]:
        if c not in X1.columns:
            X1[c] = np.nan
    X1 = X1[bundle["num"] + bundle["cat"]]

    # normaliza tipos das categóricas (mesmo tratamento do treino)
    for c in bundle["cat"]:
        X1[c] = X1[c].where(X1[c].isna(), X1[c].astype(str))

    Xt = pd.DataFrame(
        bundle["preproc"].transform(X1),
        columns=bundle["preproc"].get_feature_names_out(),
    )
    preco = float(bundle["modelo_media"].predict(Xt)[0])
    escala = float(np.sqrt(max(bundle["modelo_variancia"].predict(Xt)[0], 1e-6)))
    baixo = max(preco - bundle["q_norm"] * escala, 0.0)
    alto  = preco + bundle["q_norm"] * escala
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

    `_bundle` prefixado com _ para o cache ignorar (bundle não hashea).
    O hash do catálogo garante invalidação se ele for recarregado.
    """
    df = df_cat.copy()
    campos = _bundle["num"] + _bundle["cat"]
    X = df[campos].copy()
    for c in _bundle["cat"]:
        if c in X.columns:
            X[c] = X[c].where(X[c].isna(), X[c].astype(str))

    Xt = _bundle["preproc"].transform(X)
    preco_justo = _bundle["modelo_media"].predict(Xt)
    variancia = np.maximum(_bundle["modelo_variancia"].predict(Xt), 1e-6)
    escala = np.sqrt(variancia)

    df["preco_justo"] = preco_justo
    df["faixa_baixo"] = np.maximum(preco_justo - _bundle["q_norm"] * escala, 0.0)
    df["faixa_alto"] = preco_justo + _bundle["q_norm"] * escala
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
            lo, hi = valores
            df = df[(df[coluna].isna()) | ((df[coluna] >= lo) & (df[coluna] <= hi))]
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
                    resultado[coluna] = valores
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
    cat = st.selectbox(
        "Categoria",
        cats_disponiveis,
        format_func=lambda c: LABEL_CATEGORIA[c],
        key="rank_cat",
    )
    bundle = bundles[cat]

    # Filtra o catálogo por categoria + computa preço justo
    df_cat = catalogo[catalogo["categoria_key"] == cat].copy()
    if df_cat.empty:
        st.warning(f"Nenhum produto de {LABEL_CATEGORIA[cat]} no catálogo.")
        return

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
        options=["Ranking de ofertas", "Compatibilidade a partir de uma âncora"],
        horizontal=True,
        key="modo_pecas",
    )

    st.markdown("---")

    if modo == "Ranking de ofertas":
        _modo_ranking_ofertas(bundles, catalogo)
    else:
        st.info("🚧 Compatibilidade — próximo passo (Passo C).")


def aba_build_orcamento(bundles: dict, catalogo: Optional[pd.DataFrame]) -> None:
    st.header("Build por orçamento")
    if catalogo is None:
        st.warning(
            "Esta aba precisa do `modelos/catalogo.parquet`. Rode "
            "`python salvar_catalogo.py`."
        )
        return
    st.info("🚧 Em construção.")


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
