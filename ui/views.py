import streamlit as st

def seleciona_tipos(df):
    st.header("1. Data Profiling e Classificação")
    
    st.subheader("Amostra dos Dados")
    st.dataframe(df.head(5), use_container_width=True)
    
    st.subheader("Classificação de Atributos da Base")
    st.markdown("""
    Classifique cada coluna de acordo com a taxonomia de privacidade:
    * **DI (Identificador Direto):** CPF, Nome, Email (Serão suprimidos).
    * **QI (Quase-Identificador):** CEP, Idade, Gênero (Serão generalizados).
    * **SA (Atributo Sensível):** Doença, Renda, Religião (Alvo de proteção).
    * **NSA (Não Sensível):** Dados complementares genéricos.
    """)
    
    classificacoes = {}
    
    # Cria um grid de 3 ou 4 colunas para organizar os seletores visualmente
    n_colunas_layout = 4
    grid_colunas = st.columns(n_colunas_layout)
    
    # Itera sobre as colunas do DataFrame e cria um selectbox para cada uma
    for i, col in enumerate(df.columns):
        with grid_colunas[i % n_colunas_layout]:
            classificacoes[col] = st.selectbox(
                f"**{col}**",
                options=["NSA", "DI", "QI", "SA"],
                index=0,  # Define "NSA" como padrão para não forçar classificações acidentais
                help=f"Defina a categoria de privacidade para {col}",
                key=f"sel_{col}"
            )
    
    # Retorna o dicionário no mesmo formato esperado pelo core/profiler.py
    return classificacoes