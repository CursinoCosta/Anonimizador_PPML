import pandas as pd
import streamlit as st
from ui.charts import criar_grafico_tradeoff
from utils.arquivos import exportar_csv


def secao_seleciona_tipos(profiler):
    df = profiler.df
    st.header("1. Data Profiling e Classificação")
    st.subheader("Amostra dos Dados")
    st.dataframe(df.head(5), use_container_width=True)

    st.subheader("Classificação de Atributos da Base")
    st.markdown(
        """
        Classifique cada coluna de acordo com a taxonomia de privacidade.

        - `DI`: identificador direto, como nome, email ou CPF.
        - `QI`: quase-identificador, como idade, cidade ou CEP.
        - `SA`: atributo sensível, como doença, renda ou religião.
        - `NSA`: atributo não sensível.
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
    st.header("2. Transformações")
    st.markdown("Configure as distorções para proteger os dados.")

    abas = st.tabs(
        ["Supressão", "Generalização", "Perturbação", "Visualização"]
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
    st.header("3. Pipeline de Modelos Sintáticos de Privacidade")

    col1, col2, col3 = st.columns(3)
    with col1:
        k_alvo = st.slider(
            "Parâmetro k",
            min_value=2,
            max_value=10,
            value=int(thresholds["k_alvo"]),
            key="threshold_k_alvo",
        )
    with col2:
        l_alvo = st.number_input(
            "Parâmetro l",
            min_value=1,
            max_value=10,
            value=int(thresholds["l_alvo"]),
            step=1,
            key="threshold_l_alvo",
        )
    with col3:
        t_limite = st.number_input(
            "Parâmetro t",
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
        st.caption("Defina ao menos uma coluna como QI para habilitar o ajuste automático.")

    return session_state["evaluation_thresholds"].copy(), executar_ajuste


def secao_avaliacao(evaluation_result, thresholds, auto_k_result=None, baseline_metrics=None):
    st.header("4. Avaliação")

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
    _estilos = {
        "seguro":               ("rgba(34, 197, 94, 0.2)",   "#ffffff"),
        "parcialmente_seguro":  ("rgba(245, 158, 11, 0.2)",  "#ffffff"),
        "nao_seguro":           ("rgba(239, 68, 68, 0.2)",   "#ffffff"),
    }.get(status, ("rgba(100, 116, 139, 0.2)", "#ffffff"))
    _bg, _fg = _estilos
    st.markdown(
        f"<style>div[data-testid='stAlert']:last-of-type > div {{"
        f"background-color: {_bg} !important; color: {_fg} !important;}}</style>",
        unsafe_allow_html=True,
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
        st.subheader("Ajuste automático para k")
        ajuste_msg = (
            f"k inicial: {_formatar_metrica(auto_k_result['k_inicial'])} | "
            f"k final: {_formatar_metrica(auto_k_result['k_final'])} | "
            f"atingiu o alvo: {'sim' if auto_k_result['atingiu_alvo'] else 'não'}"
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

    fig = criar_grafico_tradeoff(metrics, baseline_metrics=baseline_metrics, status=status)

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
        
        if "Status" in df_risco.columns:
            df_risco["Status"] = df_risco["Status"].replace({
                "VULNERAVEL": "VULNERÁVEL"
            })

        def destacar_linha(linha):

            if linha["Status"] == "VULNERÁVEL":
                return [
                    "background-color: rgba(239, 68, 68, 0.2); color: #ffffff"
                ] * len(linha)

            return [
                "background-color: rgba(34, 197, 94, 0.2); color: #ffffff"
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
            "Não há classes de equivalência para avaliar."
        )


def _renderizar_supressao(profiler, anonymizer):
    st.subheader("Supressão de Identificadores")

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
    if st.button("Aplicar Supressão de Colunas"):
        anonymizer.suprimir_colunas(colunas_para_suprimir)
        _marcar_avaliacao_desatualizada()
        st.success("Supressão aplicada.")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Supressão Manual por Linha**")
        coluna_alvo = st.selectbox(
            "Coluna alvo",
            options=anonymizer.df_anonimizado.columns.tolist(),
            key="supressao_manual_coluna",
        )
        indices_selecionados = st.multiselect(
            "Índices das linhas",
            options=anonymizer.df_anonimizado.index.tolist(),
            key="supressao_manual_indices",
        )
        if st.button("Suprimir Células Selecionadas"):
            anonymizer.suprimir_celulas_manualmente(coluna_alvo, indices_selecionados)
            _marcar_avaliacao_desatualizada()
            st.success("Células suprimidas.")

    with col2:
        st.markdown("**Supressão por Regras (Palavras)**")
        coluna_regra = st.selectbox(
            "Coluna para aplicar regra",
            options=anonymizer.df_anonimizado.columns.tolist(),
            key="supressao_regra_coluna",
        )
        palavras_input = st.text_input(
            "Palavras proibidas separadas por vírgula",
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
    st.subheader("Generalização")

    cols_qi = [
        coluna
        for coluna in profiler.obter_colunas_por_tipo("QI")
        if coluna in anonymizer.df_anonimizado.columns
    ]
    if not cols_qi:
        st.info("Não há colunas QI disponíveis para generalização.")
        return

    coluna_gen = st.selectbox(
        "Quase-identificador (QI)",
        options=cols_qi,
        key="generalizacao_coluna",
    )
    tipo_gen = st.radio(
        "Método de generalização",
        [
            "Censura de Caracteres (Máscara)",
            "Hierarquia de Classes (Texto Livre)",
            "Agrupamento em Faixas (Numérico)",
        ],
        horizontal=True,
    )

    if tipo_gen == "Censura de Caracteres (Máscara)":
        num_chars = st.number_input(
            "Número de caracteres a ocultar",
            min_value=1,
            value=3,
            step=1,
        )
        direcao = st.radio(
            "Direção da omissão",
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
        if st.button("Aplicar Máscara"):
            anonymizer.generalizar_por_mascara(coluna_gen, int(num_chars), direcao_param)
            _marcar_avaliacao_desatualizada()
            st.success("Máscara aplicada.")

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
            st.info("A coluna selecionada não possui valores numéricos válidos.")
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
    st.subheader("Perturbação")

    colunas_numericas = anonymizer.df_anonimizado.select_dtypes(
        include=["number"]
    ).columns.tolist()
    if colunas_numericas:
        col_ruido = st.selectbox(
            "Coluna numérica para adicionar ruído",
            options=colunas_numericas,
            key="ruido_coluna",
        )
        distribuicao = st.radio(
            "Distribuição do ruído",
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
                media = st.number_input("Média (mu)", value=0.0)
            with col_b:
                desvio = st.number_input(
                    "Desvio padrão (sigma)",
                    min_value=0.0001,
                    value=1.0,
                )
            if st.button("Aplicar Ruído Normal"):
                anonymizer.adicionar_ruido(
                    col_ruido,
                    distribuicao="Normal",
                    media=media,
                    desvio_padrao=desvio,
                    casas_decimais=int(casas_decimais),
                )
                _marcar_avaliacao_desatualizada()
                st.success("Ruído normal aplicado.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                limite_inf = st.number_input("Limite inferior", value=-1.0)
            with col_b:
                limite_sup = st.number_input("Limite superior", value=1.0)
            if st.button("Aplicar Ruído Uniforme"):
                anonymizer.adicionar_ruido(
                    col_ruido,
                    distribuicao="Uniforme",
                    limite_inf=limite_inf,
                    limite_sup=limite_sup,
                    casas_decimais=int(casas_decimais),
                )
                _marcar_avaliacao_desatualizada()
                st.success("Ruído uniforme aplicado.")
    else:
        st.info("O dataset não possui colunas numéricas contínuas para aplicar ruído.")

    st.divider()
    st.subheader("Permutação de Dados")
    colunas_sa = [
        coluna
        for coluna in profiler.obter_colunas_por_tipo("SA")
        if coluna in anonymizer.df_anonimizado.columns
    ]
    col_sa = st.selectbox(
        "Atributo sensível (SA) para embaralhar",
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
    if st.button("Aplicar Permutação"):
        anonymizer.permutar_dados(col_sa, cols_particao)
        _marcar_avaliacao_desatualizada()
        st.success("Permutação aplicada.")


def _renderizar_visualizacao(profiler, anonymizer):
    st.subheader("Dataset Transformado")
    st.dataframe(anonymizer.df_anonimizado.head(15), use_container_width=True)

    if st.button("Desfazer Todas as Transformações"):
        anonymizer.df_anonimizado = profiler.df.copy()
        if hasattr(anonymizer, 'historico'):
            anonymizer.historico = []
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


def secao_exportacao(anonymizer, profiler, evaluation_result=None, thresholds=None, nome_arquivo_original="dataset"):

    dataset_modificado = not anonymizer.df_anonimizado.equals(profiler.df)

    nome_base = nome_arquivo_original.replace(".csv", "")
    nome_saida = f"{nome_base}_anonimizado.csv"

    csv_bytes = exportar_csv(anonymizer.df_anonimizado)

    if dataset_modificado and evaluation_result and thresholds:
        st.subheader("Resumo da Anonimização")
        
        metrics = evaluation_result.get("metrics", {})
        detalhes = evaluation_result.get("details", {})
        n_linhas = len(anonymizer.df_anonimizado)
        n_colunas = len(anonymizer.df_anonimizado.columns)

        cols_di = profiler.obter_colunas_por_tipo("DI")
        cols_qi = detalhes.get("qi_columns", profiler.obter_colunas_por_tipo("QI"))
        cols_sa = detalhes.get("sa_columns", profiler.obter_colunas_por_tipo("SA"))

        def _fmt(v):
            if v is None:
                return "N/A"
            return f"{v:.2f}" if isinstance(v, float) else str(v)

        linhas_resumo = [
            f"**Parâmetros configurados:** k = {thresholds['k_alvo']}, "
            f"l = {thresholds['l_alvo']}, t = {thresholds['t_limite']:.2f}",
            f"**Resultado atingido:** "
            f"k-anonimato = {_fmt(metrics.get('k_anonymity'))} | "
            f"l-diversidade = {_fmt(metrics.get('l_diversity'))} | "
            f"t-closeness = {_fmt(metrics.get('t_closeness'))}",
            f"**Utilidade estimada:** {metrics.get('utility_score', 0) * 100:.0f}%",
            f"**Risco de reidentificação:** {metrics.get('reidentification_risk', 0) * 100:.0f}%",
            f"**Identificadores suprimidos (DI):** {', '.join(cols_di) or 'nenhum'}",
            f"**Quase-identificadores (QI):** {', '.join(cols_qi) or 'nenhum'}",
            f"**Atributos sensíveis (SA):** {', '.join(cols_sa) or 'nenhum'}",
            f"**Dataset final:** {n_linhas} registros × {n_colunas} colunas",
        ]

        st.markdown("\n\n".join(f"- {linha}" for linha in linhas_resumo))

        historico = getattr(anonymizer, "historico", None)
        if historico:
            st.markdown("**Transformações aplicadas:**")
            st.dataframe(
                pd.DataFrame(historico),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Relatório de Conformidade")
        
        k_esperado = thresholds['k_alvo']
        l_esperado = thresholds['l_alvo']
        t_esperado = thresholds['t_limite']
        
        k_real = metrics.get('k_anonymity')
        l_real = metrics.get('l_diversity')
        t_real = metrics.get('t_closeness')
        
        k_ok = k_real is not None and k_real >= k_esperado
        l_ok = l_real is not None and l_real >= l_esperado
        t_ok = t_real is not None and t_real <= t_esperado
        
        df_conformidade = pd.DataFrame({
            "Parâmetro": ["k-anonimato", "l-diversidade", "t-closeness"],
            "Esperado": [f"≥ {k_esperado}", f"≥ {l_esperado}", f"≤ {t_esperado:.2f}"],
            "Real": [_fmt(k_real), _fmt(l_real), _fmt(t_real)],
            "Status": ["✔ Atende" if k_ok else "✘ Não atende",
                    "✔ Atende" if l_ok else "✘ Não atende",
                    "✔ Atende" if t_ok else "✘ Não atende"]
        })
        
        st.dataframe(df_conformidade, use_container_width=True, hide_index=True)

    st.header("5. Exportar Dataset Anonimizado")

    nome_base = nome_arquivo_original.replace(".csv", "")
    nome_saida = f"{nome_base}_anonimizado.csv"

    csv_bytes = exportar_csv(anonymizer.df_anonimizado)

    st.download_button(
        label="⬇ Baixar CSV anonimizado",
        data=csv_bytes,
        file_name=nome_saida,
        mime="text/csv",
        disabled=not dataset_modificado,
        help=(
            "Exporta o dataset com todas as transformações aplicadas."
            if dataset_modificado
            else "Aplique ao menos uma transformação antes de exportar."
        ),
    )

    if not dataset_modificado:
        st.caption("Nenhuma transformação foi aplicada ainda.")
    else:
        st.caption(
            f"O dataset transformado possui {len(anonymizer.df_anonimizado)} linhas "
            f"e {len(anonymizer.df_anonimizado.columns)} colunas."
        )
