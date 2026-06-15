import hashlib

import pandas as pd

from core.anonymizer import Anonymizer
from core.evaluator import Evaluator
from core.profiler import DataProfiler


DEFAULT_EVALUATION_THRESHOLDS = {
    "k_alvo": 2,
    "l_alvo": 2,
    "t_limite": 0.2,
}

NUMERIC_K_WIDTHS = [5, 10, 20, 50]
TEXT_MASK_STEPS = [1, 2, 3, None]


def identificar_upload(arquivo_carregado):
    if arquivo_carregado is None:
        return None

    conteudo = arquivo_carregado.getvalue()
    nome = getattr(arquivo_carregado, "name", "sem_nome")
    resumo = hashlib.md5(conteudo).hexdigest()
    return f"{nome}:{resumo}"


def inicializar_thresholds_avaliacao(session_state):
    if "evaluation_thresholds" not in session_state:
        session_state["evaluation_thresholds"] = DEFAULT_EVALUATION_THRESHOLDS.copy()
    return session_state["evaluation_thresholds"].copy()


def sincronizar_estado_dataset(session_state, df, upload_id):
    dataset_trocou = session_state.get("dataset_upload_id") != upload_id

    if dataset_trocou or "profiler" not in session_state or "anonymizer" not in session_state:
        if dataset_trocou:
            for chave in list(session_state.keys()):
                if chave.startswith("classificacao_"):
                    del session_state[chave]
        session_state["dataset_upload_id"] = upload_id
        session_state["profiler"] = DataProfiler(df)
        session_state["anonymizer"] = Anonymizer(session_state["profiler"])
        session_state["evaluation_result"] = None
        session_state["last_auto_k_adjustment"] = None
        session_state["last_evaluated_thresholds"] = None
        session_state["evaluation_thresholds"] = DEFAULT_EVALUATION_THRESHOLDS.copy()
        session_state["baseline_metrics"] = None

    return dataset_trocou


def atualizar_resultado_avaliacao(session_state, thresholds=None):
    thresholds = _normalizar_thresholds(thresholds or DEFAULT_EVALUATION_THRESHOLDS)
    profiler = session_state["profiler"]
    anonymizer = session_state["anonymizer"]
    evaluator = Evaluator(profiler, profiler.df, anonymizer.df_anonimizado)
    resultado = evaluator.avaliar(**thresholds)
    session_state["evaluation_thresholds"] = thresholds.copy()
    session_state["evaluation_result"] = resultado

    if session_state.get("baseline_metrics") is None:
        session_state["baseline_metrics"] = resultado["metrics"].copy()

    return resultado


def ajustar_para_k_anonimato(session_state, k_alvo):
    profiler = session_state["profiler"]
    anonymizer = session_state["anonymizer"]
    qi_columns = [
        coluna
        for coluna in profiler.obter_colunas_por_tipo("QI")
        if coluna in anonymizer.df_anonimizado.columns
    ]

    if not qi_columns:
        resumo = {
            "k_inicial": None,
            "k_final": None,
            "atingiu_alvo": False,
            "steps_applied": [],
            "warnings": [
                "Não foi possível ajustar para k-anonimato porque não há colunas QI."
            ],
        }
        session_state["last_auto_k_adjustment"] = resumo
        return resumo

    thresholds = _normalizar_thresholds(
        session_state.get("evaluation_thresholds", DEFAULT_EVALUATION_THRESHOLDS)
    )
    thresholds["k_alvo"] = int(k_alvo)

    avaliacao_inicial = atualizar_resultado_avaliacao(session_state, thresholds)
    k_inicial = avaliacao_inicial["metrics"]["k_anonymity"]
    melhor_k = -1 if k_inicial is None else k_inicial
    melhor_df = anonymizer.df_anonimizado.copy(deep=True)
    steps_applied = []

    for rodada in range(4):
        for coluna in qi_columns:
            passo = _aplicar_generalizacao_progressiva(
                profiler,
                anonymizer,
                coluna,
                rodada,
            )
            if passo is not None:
                steps_applied.append(passo)

        avaliacao_atual = atualizar_resultado_avaliacao(session_state, thresholds)
        k_atual = avaliacao_atual["metrics"]["k_anonymity"]
        if k_atual is not None and k_atual >= melhor_k:
            melhor_k = k_atual
            melhor_df = anonymizer.df_anonimizado.copy(deep=True)
        if k_atual is not None and k_atual >= k_alvo:
            resumo = {
                "k_inicial": k_inicial,
                "k_final": k_atual,
                "atingiu_alvo": True,
                "steps_applied": steps_applied,
                "warnings": [],
            }
            session_state["last_auto_k_adjustment"] = resumo
            return resumo

    anonymizer.df_anonimizado = melhor_df
    avaliacao_final = atualizar_resultado_avaliacao(session_state, thresholds)
    resumo = {
        "k_inicial": k_inicial,
        "k_final": avaliacao_final["metrics"]["k_anonymity"],
        "atingiu_alvo": False,
        "steps_applied": steps_applied,
        "warnings": [
            "Não foi possível atingir k automaticamente com as generalizações disponíveis."
        ],
    }
    session_state["last_auto_k_adjustment"] = resumo
    return resumo


def _normalizar_thresholds(thresholds):
    return {
        "k_alvo": int(thresholds["k_alvo"]),
        "l_alvo": int(thresholds["l_alvo"]),
        "t_limite": float(thresholds["t_limite"]),
    }


def _aplicar_generalizacao_progressiva(profiler, anonymizer, coluna, rodada):
    serie_original = profiler.df[coluna]
    serie_numerica = pd.to_numeric(serie_original, errors="coerce")
    is_numeric = not serie_numerica.dropna().empty

    if is_numeric:
        largura = NUMERIC_K_WIDTHS[min(rodada, len(NUMERIC_K_WIDTHS) - 1)]
        anonymizer.df_anonimizado[coluna] = serie_original.copy()
        anonymizer.generalizar_por_faixas(coluna, largura)
        return {
            "column": coluna,
            "type": "numeric_range",
            "round": rodada + 1,
            "parameter": largura,
        }

    anonymizer.df_anonimizado[coluna] = anonymizer.df_anonimizado[coluna].astype(str)
    if rodada < len(TEXT_MASK_STEPS) - 1:
        num_chars = TEXT_MASK_STEPS[rodada]
    else:
        comprimento_maximo = (
            anonymizer.df_anonimizado[coluna]
            .astype(str)
            .map(len)
            .max()
        )
        num_chars = max(1, int(comprimento_maximo) - 1)

    anonymizer.generalizar_por_mascara(
        coluna,
        int(num_chars),
        "direita_para_esquerda",
    )
    return {
        "column": coluna,
        "type": "text_mask",
        "round": rodada + 1,
        "parameter": int(num_chars),
    }
