import streamlit as st
import pandas as pd

def secao_seleciona_tipos(df):
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

def secao_transformacao(profiler, anonymizer):
    st.header("2. Transformação: Supressão e Generalização")
    st.markdown("Configure as distorções para proteger os dados. As alterações são aplicadas na memória.")

    abas = st.tabs(["Supressão (Remoção)", "Generalização", "Perturbação (Ruído/Permutação)", "Visualizar Resultado"])

    # --- ABA 1: SUPRESSÃO ---
    with abas[0]:
        st.subheader("Supressão de Colunas (Identificadores Diretos)")
        cols_di = profiler.obter_colunas_por_tipo("DI")
        
        colunas_suprimir = st.multiselect(
            "Selecione as colunas para remover completamente:",
            options=profiler.df.columns.tolist(),
            default=cols_di
        )
        if st.button("Aplicar Supressão de Colunas"):
            anonymizer.suprimir_colunas(colunas_suprimir)
            st.success("Colunas removidas!")

        st.divider()
        
        st.subheader("Supressão de Células (Granular)")
        coluna_alvo = st.selectbox("Selecione a coluna para supressão celular:", options=profiler.df.columns.tolist(), key="sup_cel_col")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Supressão Manual por Linha**")
            indices_disp = profiler.df.index.tolist()
            indices_selecionados = st.multiselect("IDs das linhas:", options=indices_disp, key="sup_cel_ind")
            if st.button("Suprimir Células Selecionadas"):
                anonymizer.suprimir_celulas_manualmente(coluna_alvo, indices_selecionados)
                st.success("Células suprimidas!")
                
        with col2:
            st.markdown("**Supressão por Regras (Palavras)**")
            palavras_input = st.text_input("Palavras a censurar (separadas por vírgula):", key="sup_cel_pal")
            if st.button("Suprimir por Palavras"):
                palavras = [p.strip() for p in palavras_input.split(",") if p.strip()]
                anonymizer.suprimir_celulas_por_regra(coluna_alvo, palavras)
                st.success("Regras aplicadas!")

    # --- ABA 2: GENERALIZAÇÃO ---
    with abas[1]:
        st.subheader("Generalização de Quase-Identificadores (QI)")
        cols_qi = profiler.obter_colunas_por_tipo("QI")
        
        if not cols_qi:
            st.info("Nenhuma coluna classificada como QI na etapa anterior.")
        else:
            coluna_gen = st.selectbox("Selecione a coluna para generalizar:", options=cols_qi, key="gen_col")
            
            # Nova opção adicionada ao radio button
            tipo_gen = st.radio(
                "Método de Generalização:", 
                ["Censura de Caracteres (Máscara)", "Hierarquia de Classes (Texto Livre)", "Agrupamento em Faixas (Numérico)"], 
                horizontal=True
            )
            
            if tipo_gen == "Censura de Caracteres (Máscara)":
                num_chars = st.number_input("Número de caracteres a ocultar:", min_value=1, value=3)
                direcao = st.radio("Direção da omissão:", ["Da Direita para Esquerda (ex: 0000**)", "Da Esquerda para Direita (ex: **0000)"])
                direcao_param = "direita_para_esquerda" if "Direita" in direcao else "esquerda_para_direita"
                
                if st.button("Aplicar Máscara"):
                    anonymizer.generalizar_por_mascara(coluna_gen, num_chars, direcao_param)
                    st.success("Máscara aplicada!")

            elif tipo_gen == "Hierarquia de Classes (Texto Livre)":
                st.markdown("Mapeie os valores originais para um nível superior da hierarquia.")
                valores_unicos = profiler.df[coluna_gen].astype(str).unique()
                
                df_mapeamento = pd.DataFrame({
                    "Valor Original": valores_unicos,
                    "Valor Generalizado": [""] * len(valores_unicos)
                })
                
                df_editado = st.data_editor(df_mapeamento, use_container_width=True, hide_index=True)
                
                if st.button("Aplicar Hierarquia"):
                    df_editado["Valor Generalizado"] = df_editado["Valor Generalizado"].fillna("").astype(str)
                    mapeamento_valido = df_editado[df_editado["Valor Generalizado"].str.strip() != ""]
                    dicionario_hierarquia = dict(zip(mapeamento_valido["Valor Original"], mapeamento_valido["Valor Generalizado"]))
                    
                    anonymizer.generalizar_por_hierarquia(coluna_gen, dicionario_hierarquia)
                    st.success("Hierarquia aplicada!")
                    
            elif tipo_gen == "Agrupamento em Faixas (Numérico)":
                st.markdown("Agrupa valores numéricos contínuos em intervalos. Exemplo: agrupar idades de 3 em 3 anos.")
                tamanho_faixa = st.number_input("Tamanho do conjunto (ex: 3 para faixas de 3 anos):", min_value=1, value=3, step=1)
                
                if st.button("Aplicar Agrupamento"):
                    anonymizer.generalizar_por_faixas(coluna_gen, tamanho_faixa)
                    st.success(f"Valores agrupados em faixas de tamanho {tamanho_faixa}!")

    # --- ABA 3: PERTURBAÇÃO ---
    with abas[2]:
        st.subheader("Adição de Ruído (Atributos Numéricos)")
        colunas_numericas = profiler.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if colunas_numericas:
            col_ruido = st.selectbox("Selecione a coluna numérica para adicionar ruído:", options=colunas_numericas, key="ruido_col")
            
            # Divide em duas colunas para alinhar a distribuição e as casas decimais
            col_dist, col_dec = st.columns(2)
            with col_dist:
                distribuicao = st.radio("Distribuição do Ruído:", ["Normal", "Uniforme"], horizontal=True)
            with col_dec:
                casas_decimais = st.number_input("Quantidade casas decimais", min_value=0, max_value=10, value=2, step=1)
            
            col_r1, col_r2 = st.columns(2)
            if distribuicao == "Normal":
                usar_estatisticas = st.checkbox("Usar média e desvio padrão da própria coluna", value=False)
                
                serie_num = pd.to_numeric(profiler.df[col_ruido], errors='coerce').dropna()
                media_orig = float(serie_num.mean()) if not serie_num.empty else 0.0
                desvio_orig = float(serie_num.std()) if not serie_num.empty and len(serie_num) > 1 else 1.0
                
                with col_r1: 
                    media_ruido = st.number_input("Média (μ):", value=media_orig if usar_estatisticas else 0.0, disabled=usar_estatisticas)
                with col_r2: 
                    desvio_ruido = st.number_input("Desvio Padrão (σ):", value=desvio_orig if usar_estatisticas else 1.0, min_value=0.0001 if not usar_estatisticas else None, disabled=usar_estatisticas)
                
                if st.button("Aplicar Ruído Normal"):
                    anonymizer.adicionar_ruido(col_ruido, "Normal", media=media_ruido, desvio_padrao=desvio_ruido, casas_decimais=casas_decimais)
                    st.success(f"Ruído Normal aplicado em {col_ruido}!")
            else:
                with col_r1: limite_inf = st.number_input("Limite Inferior:", value=-5.0)
                with col_r2: limite_sup = st.number_input("Limite Superior:", value=5.0)
                
                if st.button("Aplicar Ruído Uniforme"):
                    anonymizer.adicionar_ruido(col_ruido, "Uniforme", limite_inf=limite_inf, limite_sup=limite_sup, casas_decimais=casas_decimais)
                    st.success(f"Ruído Uniforme aplicado em {col_ruido}!")
        else:
            st.info("O dataset não possui colunas numéricas contínuas para aplicar ruído.")

        st.divider()

        st.subheader("Permutação de Dados (Embaralhamento)")
        col_sa = st.selectbox("Atributo Sensível (SA) para embaralhar:", options=profiler.df.columns.tolist(), key="perm_sa")
        cols_particao = st.multiselect("Colunas de agrupamento (QIs):", options=[c for c in profiler.df.columns if c != col_sa], key="perm_qi")
        
        if st.button("Aplicar Permutação"):
            anonymizer.permutar_dados(col_sa, cols_particao)
            st.success("Permutação aplicada!")

    # --- ABA 4: VISUALIZAÇÃO ---
    with abas[3]:
        st.subheader("Dataset Transformado")
        st.dataframe(anonymizer.df_anonimizado.head(15), use_container_width=True)
        
        if st.button("Desfazer todas as transformações"):
            anonymizer.df_anonimizado = profiler.df.copy()
            st.rerun()