import streamlit as st

from ui.views import (
    renderizar_controles_avaliacao,
    secao_avaliacao,
    secao_exportacao,
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

arquivo_carregado = st.file_uploader("Faça upload da base (CSV)", type=["csv"])

if arquivo_carregado:
    df = carregar_dataframe(arquivo_carregado)
    # No app.py, após carregar_dataframe:
    if len(df) > 5000:
        st.warning("Bases com mais de 5.000 registros podem causar lentidão no cálculo do t-closeness no Streamlit. Considere usar uma amostra.")
    if df is not None:
        upload_id = identificar_upload(arquivo_carregado)
        sincronizar_estado_dataset(st.session_state, df, upload_id)
        thresholds = inicializar_thresholds_avaliacao(st.session_state)

        classificacoes = secao_seleciona_tipos(st.session_state.profiler)
        st.session_state.profiler.atualizar_classificacao(classificacoes)

        st.divider()
        st.write("Resumo atual das classificações:")
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
            "Atributos Sensíveis (SA)",
            len(st.session_state.profiler.obter_colunas_por_tipo("SA")),
        )
        col4.metric(
            "Não Sensíveis (NSA)",
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
            baseline_metrics=st.session_state.get("baseline_metrics"),
        )

        st.divider()
        secao_exportacao(
            st.session_state.anonymizer,
            st.session_state.profiler,
            st.session_state.get("evaluation_result"),
            thresholds,
            nome_arquivo_original=getattr(arquivo_carregado, "name", "dataset"),
        )
else:
    st.info("Aguardando upload do arquivo CSV para iniciar o processo.")
