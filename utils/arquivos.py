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


def exportar_csv(df: pd.DataFrame) -> bytes:
    """Serializa o DataFrame transformado para bytes CSV com encoding UTF-8."""
    return df.to_csv(index=False).encode("utf-8")