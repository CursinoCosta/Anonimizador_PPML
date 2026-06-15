import pandas as pd
import streamlit as st
from ui.charts import criar_grafico_tradeoff


def secao_seleciona_tipos(profiler):
    df = profiler.df
    st.header("1. Data Profiling e Classificacao")
    st.subheader("Amostra dos Dados")
    st.dataframe(df.head(5), use_container_width=True)

    st.subheader("Classificacao de Atributos da Base")
    st.markdown(
        """
        Classifique cada coluna de acordo com a taxonomia de privacidade.

        - `DI`: identificador direto, como nome, email ou CPF.
        - `QI`: quase-identificador, como idade, cidade ou CEP.
        - `SA`: atributo sensivel, como doenca, renda ou religiao.
        - `NSA`: atributo nao sensivel.
        """
    )

    classificacoes = {}
    opcoes_tipos = ["NSA", "DI", "QI", "SA"]
    colunas_ui = st.columns(2)
    for indice, coluna in enumerate(df.columns):
        with colunas_ui[indice % 2]:
            chave = f"classificacao_{coluna}"
            valor_atual = profiler.tipos_colunas.get(coluna, "NSA")
            if st.session_state.get(chave) not in opcoes_tipos:
                st.session_state[chave] = valor_atual
            classificacoes[coluna] = st.selectbox(
                f"Tipo da coluna `{coluna}`",
                options=opcoes_tipos,
                index=opcoes_tipos.index(st.session_state[chave]),
                key=chave,
            )

    return classificacoes


def secao_transformacao(profiler, anonymizer):
    st.header("2. Transformacoes")
    st.markdown("Configure as distorcoes para proteger os dados.")

    abas = st.tabs(
        ["Supressao", "Generalizacao", "Perturbacao", "Visualizacao"]
    )

    with abas[0]:
        _renderizar_supressao(profiler, anonymizer)

    with abas[1]:
        _renderizar_generalizacao(profiler, anonymizer)

    with abas[2]:
        _renderizar_perturbacao(profiler, anonymizer)

    with abas[3]:
        _renderizar_visualizacao(profiler, anonymizer)


def renderizar_controles_avaliacao(session_state, thresholds):
    st.header("3. Pipeline de Modelos Sintaticos de Privacidade")

    col1, col2, col3 = st.columns(3)
    with col1:
        k_alvo = st.slider(
            "Parametro k",
            min_value=2,
            max_value=10,
            value=int(thresholds["k_alvo"]),
            key="threshold_k_alvo",
        )
    with col2:
        l_alvo = st.number_input(
            "Parametro l",
            min_value=1,
            max_value=10,
            value=int(thresholds["l_alvo"]),
            step=1,
            key="threshold_l_alvo",
        )
    with col3:
        t_limite = st.number_input(
            "Parametro t",
            min_value=0.0,
            max_value=1.0,
            value=float(thresholds["t_limite"]),
            step=0.05,
            format="%.2f",
            key="threshold_t_limite",
        )

    session_state["evaluation_thresholds"] = {
        "k_alvo": int(k_alvo),
        "l_alvo": int(l_alvo),
        "t_limite": float(t_limite),
    }

    qis_disponiveis = session_state.profiler.obter_colunas_por_tipo("QI")
    executar_ajuste = st.button(
        "Ajustar automaticamente para k",
        disabled=not bool(qis_disponiveis),
        help="Aplica generalizacoes adicionais sobre os QIs ate tentar atingir o k configurado.",
    )

    if not qis_disponiveis:
        st.caption("Defina ao menos uma coluna como QI para habilitar o ajuste automatico.")

    return session_state["evaluation_thresholds"].copy(), executar_ajuste


def secao_avaliacao(evaluation_result, thresholds, auto_k_result=None):
    st.header("4. Avaliacao")

    metrics = evaluation_result["metrics"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("k-anonimato", _formatar_metrica(metrics["k_anonymity"]))
    col2.metric("l-diversidade", _formatar_metrica(metrics["l_diversity"]))
    col3.metric("t-closeness", _formatar_metrica(metrics["t_closeness"]))
    col4.metric("Utilidade", _formatar_percentual(metrics["utility_score"]))
    col5.metric("Risco", _formatar_percentual(metrics["reidentification_risk"]))

    status = evaluation_result["status"]
    mensagem_status = (
        f"Status atual: `{status}`. "
        f"Thresholds ativos: k={thresholds['k_alvo']}, "
        f"l={thresholds['l_alvo']}, t={thresholds['t_limite']:.2f}."
    )
    if status == "seguro":
        st.success(mensagem_status)
    elif status == "parcialmente_seguro":
        st.info(mensagem_status)
    elif status == "nao_seguro":
        st.error(mensagem_status)
    else:
        st.warning(mensagem_status)

    st.markdown(evaluation_result["summary"])

    for aviso in evaluation_result["warnings"]:
        st.warning(aviso)

    if auto_k_result:
        st.subheader("Ajuste automatico para k")
        ajuste_msg = (
            f"k inicial: {_formatar_metrica(auto_k_result['k_inicial'])} | "
            f"k final: {_formatar_metrica(auto_k_result['k_final'])} | "
            f"atingiu alvo: {'sim' if auto_k_result['atingiu_alvo'] else 'nao'}"
        )
        if auto_k_result["atingiu_alvo"]:
            st.success(ajuste_msg)
        else:
            st.warning(ajuste_msg)

        for aviso in auto_k_result.get("warnings", []):
            st.warning(aviso)

        if auto_k_result.get("steps_applied"):
            st.dataframe(pd.DataFrame(auto_k_result["steps_applied"]), use_container_width=True)

    detalhes = evaluation_result["details"]
    st.caption(
        "QIs avaliados: "
        f"{', '.join(detalhes['qi_columns']) or 'nenhum'} | "
        "SAs avaliados: "
        f"{', '.join(detalhes['sa_columns']) or 'nenhum'}"
    )

    st.divider()

    st.subheader("Dashboard de Trade-off")

    fig = criar_grafico_tradeoff(metrics)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.divider()

    st.subheader("Risco Adversarial")

    tabela = detalhes.get(
        "adversarial_table",
        [],
    )

    if tabela:

        df_risco = pd.DataFrame(tabela)

        def destacar_linha(linha):

            if linha["Status"] == "VULNERAVEL":
                return [
                    "background-color: #ffcccc"
                ] * len(linha)

            return [
                "background-color: #ccffcc"
            ] * len(linha)

        st.dataframe(
            df_risco.style.apply(
                destacar_linha,
                axis=1,
            ),
            use_container_width=True,
        )

    else:
        st.info(
            "Nao ha classes de equivalencia para avaliar."
        )


def _renderizar_supressao(profiler, anonymizer):
    st.subheader("Supressao de Identificadores")

    cols_di = profiler.obter_colunas_por_tipo("DI")
    colunas_sugeridas = [
        coluna for coluna in cols_di if coluna in anonymizer.df_anonimizado.columns
    ]

    colunas_para_suprimir = st.multiselect(
        "Colunas para remover completamente",
        options=anonymizer.df_anonimizado.columns.tolist(),
        default=colunas_sugeridas,
        key="supressao_colunas",
    )
    if st.button("Aplicar Supressao de Colunas"):
        anonymizer.suprimir_colunas(colunas_para_suprimir)
        _marcar_avaliacao_desatualizada()
        st.success("Supressao aplicada.")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Supressao Manual por Linha**")
        coluna_alvo = st.selectbox(
            "Coluna alvo",
            options=anonymizer.df_anonimizado.columns.tolist(),
            key="supressao_manual_coluna",
        )
        indices_selecionados = st.multiselect(
            "Indices das linhas",
            options=anonymizer.df_anonimizado.index.tolist(),
            key="supressao_manual_indices",
        )
        if st.button("Suprimir Celulas Selecionadas"):
            anonymizer.suprimir_celulas_manualmente(coluna_alvo, indices_selecionados)
            _marcar_avaliacao_desatualizada()
            st.success("Celulas suprimidas.")

    with col2:
        st.markdown("**Supressao por Regras (Palavras)**")
        coluna_regra = st.selectbox(
            "Coluna para aplicar regra",
            options=anonymizer.df_anonimizado.columns.tolist(),
            key="supressao_regra_coluna",
        )
        palavras_input = st.text_input(
            "Palavras proibidas separadas por virgula",
            key="supressao_regra_palavras",
        )
        if st.button("Suprimir Palavras"):
            palavras = [
                palavra.strip()
                for palavra in palavras_input.split(",")
                if palavra.strip()
            ]
            anonymizer.suprimir_celulas_por_regra(coluna_regra, palavras)
            _marcar_avaliacao_desatualizada()
            st.success("Regras aplicadas.")


def _renderizar_generalizacao(profiler, anonymizer):
    st.subheader("Generalizacao")

    cols_qi = [
        coluna
        for coluna in profiler.obter_colunas_por_tipo("QI")
        if coluna in anonymizer.df_anonimizado.columns
    ]
    if not cols_qi:
        st.info("Nao ha colunas QI disponiveis para generalizacao.")
        return

    coluna_gen = st.selectbox(
        "Quase-identificador (QI)",
        options=cols_qi,
        key="generalizacao_coluna",
    )
    tipo_gen = st.radio(
        "Metodo de generalizacao",
        [
            "Censura de Caracteres (Mascara)",
            "Hierarquia de Classes (Texto Livre)",
            "Agrupamento em Faixas (Numerico)",
        ],
        horizontal=True,
    )

    if tipo_gen == "Censura de Caracteres (Mascara)":
        num_chars = st.number_input(
            "Numero de caracteres a ocultar",
            min_value=1,
            value=3,
            step=1,
        )
        direcao = st.radio(
            "Direcao da omissao",
            [
                "Da direita para a esquerda",
                "Da esquerda para a direita",
            ],
            horizontal=True,
        )
        direcao_param = (
            "direita_para_esquerda"
            if "direita" in direcao.lower()
            else "esquerda_para_direita"
        )
        if st.button("Aplicar Mascara"):
            anonymizer.generalizar_por_mascara(coluna_gen, int(num_chars), direcao_param)
            _marcar_avaliacao_desatualizada()
            st.success("Mascara aplicada.")

    elif tipo_gen == "Hierarquia de Classes (Texto Livre)":
        valores_unicos = (
            anonymizer.df_anonimizado[coluna_gen].astype(str).drop_duplicates().tolist()
        )
        df_mapeamento = pd.DataFrame(
            {
                "Valor Original": valores_unicos,
                "Valor Generalizado": valores_unicos,
            }
        )
        editado = st.data_editor(
            df_mapeamento,
            num_rows="fixed",
            hide_index=True,
            key=f"mapeamento_{coluna_gen}",
        )
        if st.button("Aplicar Hierarquia"):
            dicionario = dict(
                zip(editado["Valor Original"], editado["Valor Generalizado"])
            )
            anonymizer.generalizar_por_hierarquia(coluna_gen, dicionario)
            _marcar_avaliacao_desatualizada()
            st.success("Hierarquia aplicada.")

    else:
        coluna_numerica = pd.to_numeric(
            anonymizer.df_anonimizado[coluna_gen], errors="coerce"
        )
        if coluna_numerica.dropna().empty:
            st.info("A coluna selecionada nao possui valores numericos validos.")
            return

        tamanho_faixa = st.number_input(
            "Tamanho da faixa",
            min_value=1,
            value=5,
            step=1,
        )
        st.caption("Exemplo: tamanho 5 transforma 18 em [15-19].")
        if st.button("Aplicar Faixas"):
            anonymizer.generalizar_por_faixas(coluna_gen, int(tamanho_faixa))
            _marcar_avaliacao_desatualizada()
            st.success("Agrupamento em faixas aplicado.")


def _renderizar_perturbacao(profiler, anonymizer):
    st.subheader("Perturbacao")

    colunas_numericas = anonymizer.df_anonimizado.select_dtypes(
        include=["number"]
    ).columns.tolist()
    if colunas_numericas:
        col_ruido = st.selectbox(
            "Coluna numerica para adicionar ruido",
            options=colunas_numericas,
            key="ruido_coluna",
        )
        distribuicao = st.radio(
            "Distribuicao do ruido",
            ["Normal", "Uniforme"],
            horizontal=True,
            key="ruido_distribuicao",
        )
        casas_decimais = st.number_input(
            "Casas decimais",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
        )

        if distribuicao == "Normal":
            col_a, col_b = st.columns(2)
            with col_a:
                media = st.number_input("Media (mu)", value=0.0)
            with col_b:
                desvio = st.number_input(
                    "Desvio padrao (sigma)",
                    min_value=0.0001,
                    value=1.0,
                )
            if st.button("Aplicar Ruido Normal"):
                anonymizer.adicionar_ruido(
                    col_ruido,
                    distribuicao="Normal",
                    media=media,
                    desvio_padrao=desvio,
                    casas_decimais=int(casas_decimais),
                )
                _marcar_avaliacao_desatualizada()
                st.success("Ruido normal aplicado.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                limite_inf = st.number_input("Limite inferior", value=-1.0)
            with col_b:
                limite_sup = st.number_input("Limite superior", value=1.0)
            if st.button("Aplicar Ruido Uniforme"):
                anonymizer.adicionar_ruido(
                    col_ruido,
                    distribuicao="Uniforme",
                    limite_inf=limite_inf,
                    limite_sup=limite_sup,
                    casas_decimais=int(casas_decimais),
                )
                _marcar_avaliacao_desatualizada()
                st.success("Ruido uniforme aplicado.")
    else:
        st.info("O dataset nao possui colunas numericas continuas para aplicar ruido.")

    st.divider()
    st.subheader("Permutacao de Dados")
    colunas_sa = [
        coluna
        for coluna in profiler.obter_colunas_por_tipo("SA")
        if coluna in anonymizer.df_anonimizado.columns
    ]
    col_sa = st.selectbox(
        "Atributo sensivel (SA) para embaralhar",
        options=colunas_sa or anonymizer.df_anonimizado.columns.tolist(),
        key="perm_coluna_sa",
    )
    cols_particao = st.multiselect(
        "Colunas de agrupamento (QIs)",
        options=[
            coluna
            for coluna in profiler.obter_colunas_por_tipo("QI")
            if coluna in anonymizer.df_anonimizado.columns and coluna != col_sa
        ],
        key="perm_colunas_particao",
    )
    if st.button("Aplicar Permutacao"):
        anonymizer.permutar_dados(col_sa, cols_particao)
        _marcar_avaliacao_desatualizada()
        st.success("Permutacao aplicada.")


def _renderizar_visualizacao(profiler, anonymizer):
    st.subheader("Dataset Transformado")
    st.dataframe(anonymizer.df_anonimizado.head(15), use_container_width=True)

    if st.button("Desfazer Todas as Transformacoes"):
        anonymizer.df_anonimizado = profiler.df.copy()
        _marcar_avaliacao_desatualizada()
        st.rerun()


def _formatar_metrica(valor):
    if valor is None:
        return "N/A"
    if isinstance(valor, float):
        return f"{valor:.2f}"
    return str(valor)


def _formatar_percentual(valor):
    if valor is None:
        return "N/A"
    return f"{valor * 100:.0f}%"


def _marcar_avaliacao_desatualizada():
    st.session_state["evaluation_result"] = None
    st.session_state["last_auto_k_adjustment"] = None
