import streamlit as st
from utils.arquivos import carregar_dataframe
from ui.views import secao_seleciona_tipos, secao_transformacao
from core.profiler import DataProfiler
from core.anonymizer import Anonymizer

st.set_page_config(page_title="Avaliador PPML", layout="wide")

st.title("Avaliador de Trade-off: Privacidade vs Utilidade")

# Menu lateral para entrada do arquivo
with st.sidebar:
    st.header("Entrada de Dados")
    arquivo_carregado = st.file_uploader("Faça upload da base (CSV)", type=["csv"])

if arquivo_carregado:
    df = carregar_dataframe(arquivo_carregado)
    
    if df is not None:
        # Armazena a instância do DataProfiler no session_state
        if 'profiler' not in st.session_state:
            st.session_state.profiler = DataProfiler(df)
            
        # Armazena a instância do Anonymizer no session_state
        if 'anonymizer' not in st.session_state:
            st.session_state.anonymizer = Anonymizer(st.session_state.profiler)
            
        # Renderiza a UI e captura o input do usuário
        classificacoes = secao_seleciona_tipos(df)
        
        # Atualiza a classe com as novas definições
        st.session_state.profiler.atualizar_classificacao(classificacoes)
        
        # Métricas de resumo
        st.divider()
        st.write("Resumo atual das classificações:")
        col1, col2, col3 = st.columns(3)
        col1.metric("Identificadores (DI)", len(st.session_state.profiler.obter_colunas_por_tipo("DI")))
        col2.metric("Quase-Identificadores (QI)", len(st.session_state.profiler.obter_colunas_por_tipo("QI")))
        col3.metric("Sensíveis (SA)", len(st.session_state.profiler.obter_colunas_por_tipo("SA")))
        
        st.divider()
        
        # Chamada para renderizar as abas da Etapa 2
        secao_transformacao(st.session_state.profiler, st.session_state.anonymizer)
        
else:
    st.info("Aguardando upload do arquivo CSV na barra lateral para iniciar o processo.")