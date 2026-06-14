import streamlit as st

from ui.views import (
    renderizar_controles_avaliacao,
    secao_avaliacao,
    secao_seleciona_tipos,
    secao_transformacao,
)
from utils.arquivos import carregar_dataframe
from utils.avaliacao import (
    DEFAULT_EVALUATION_THRESHOLDS,
    ajustar_para_k_anonimato,
    atualizar_resultado_avaliacao,
    identificar_upload,
    inicializar_thresholds_avaliacao,
    sincronizar_estado_dataset,
)

st.set_page_config(page_title="Avaliador PPML", layout="wide")
st.title("Avaliador de Trade-off: Privacidade vs Utilidade")

arquivo_carregado = st.file_uploader("Faca upload da base (CSV)", type=["csv"])

if arquivo_carregado:
    df = carregar_dataframe(arquivo_carregado)
    if df is not None:
        upload_id = identificar_upload(arquivo_carregado)
        sincronizar_estado_dataset(st.session_state, df, upload_id)
        thresholds = inicializar_thresholds_avaliacao(st.session_state)

        classificacoes = secao_seleciona_tipos(st.session_state.profiler.df)
        st.session_state.profiler.atualizar_classificacao(classificacoes)

        st.divider()
        st.write("Resumo atual das classificacoes:")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Identificadores (DI)",
            len(st.session_state.profiler.obter_colunas_por_tipo("DI")),
        )
        col2.metric(
            "Quase-Identificadores (QI)",
            len(st.session_state.profiler.obter_colunas_por_tipo("QI")),
        )
        col3.metric(
            "Atributos Sensiveis (SA)",
            len(st.session_state.profiler.obter_colunas_por_tipo("SA")),
        )
        col4.metric(
            "Nao Sensiveis (NSA)",
            len(st.session_state.profiler.obter_colunas_por_tipo("NSA")),
        )

        secao_transformacao(st.session_state.profiler, st.session_state.anonymizer)
        thresholds, executar_ajuste_k = renderizar_controles_avaliacao(
            st.session_state,
            thresholds,
        )

        if executar_ajuste_k:
            ajustar_para_k_anonimato(st.session_state, thresholds["k_alvo"])

        thresholds_mudaram = (
            st.session_state.get("last_evaluated_thresholds") != thresholds
        )
        if st.session_state.get("evaluation_result") is None or thresholds_mudaram:
            atualizar_resultado_avaliacao(st.session_state, thresholds)
            st.session_state["last_evaluated_thresholds"] = thresholds.copy()

        secao_avaliacao(
            st.session_state["evaluation_result"],
            thresholds,
            st.session_state.get("last_auto_k_adjustment"),
        )
else:
    st.info("Aguardando upload do arquivo CSV para iniciar o processo.")
