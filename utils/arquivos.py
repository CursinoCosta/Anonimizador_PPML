import pandas as pd
import streamlit as st

@st.cache_data
def carregar_dataframe(arquivo_carregado):
    """Carrega o arquivo CSV num DataFrame do Pandas de forma eficiente."""
    if arquivo_carregado is not None:
        try:
            df = pd.read_csv(arquivo_carregado)
            return df
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {e}")
            return None
    return None