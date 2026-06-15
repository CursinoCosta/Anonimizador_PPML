import math

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


VALOR_NULO = "<NA>"


class Evaluator:
    """Calcula metricas de privacidade e utilidade para datasets anonimizados."""

    def __init__(self, profiler, df_original, df_anonimizado):
        self.profiler = profiler
        self.df_original = df_original.copy()
        self.df_anonimizado = df_anonimizado.copy()
        self.warnings = []

    def avaliar(self, k_alvo=2, l_alvo=2, t_limite=0.2):
        """Executa todas as metricas e retorna um resultado pronto para a UI."""
        self.warnings = []

        qi_columns = self._obter_qis()
        sa_columns = self._obter_sas()

        k_result = self.calcular_k_anonymity()
        l_result = self.calcular_l_diversity(l_alvo=l_alvo)
        t_result = self.calcular_t_closeness()
        utility_result = self.calcular_utilidade()

        metrics = {
            "k_anonymity": k_result["value"],
            "l_diversity": l_result["value"],
            "t_closeness": t_result["value"],
            "utility_score": utility_result["value"],
            "reidentification_risk": None,
        }
        metrics["reidentification_risk"] = self.calcular_risco_reidentificacao(
            metrics,
            k_alvo=k_alvo,
            l_alvo=l_alvo,
            t_limite=t_limite,
        )

        status = self._definir_status(metrics, k_alvo, l_alvo, t_limite)
        summary = self._gerar_resumo(metrics, status, qi_columns, sa_columns)

        adversarial_table = self.calcular_risco_adversarial(
            k_alvo,
            l_alvo,
        )

        details = {
            "qi_columns": qi_columns,
            "sa_columns": sa_columns,
            "k_anonymity": k_result,
            "l_diversity": l_result,
            "t_closeness": t_result,
            "utility": utility_result,
            "equivalence_classes": k_result.get("equivalence_classes", []),
            "risk_level": self._classificar_risco(metrics["reidentification_risk"]),
            "adversarial_table": adversarial_table,
        }

        return {
            "metrics": metrics,
            "status": status,
            "warnings": self._warnings_unicos(),
            "summary": summary,
            "details": details,
        }

    def _obter_qis(self):
        return self._obter_colunas_por_tipo("QI")

    def _obter_sas(self):
        return self._obter_colunas_por_tipo("SA")

    def _obter_colunas_por_tipo(self, tipo_coluna):
        colunas = []
        if hasattr(self.profiler, "obter_colunas_por_tipo"):
            colunas = self.profiler.obter_colunas_por_tipo(tipo_coluna)
        elif hasattr(self.profiler, "tipos_colunas"):
            colunas = [
                col
                for col, tipo in self.profiler.tipos_colunas.items()
                if tipo == tipo_coluna
            ]

        return [col for col in colunas if col in self.df_anonimizado.columns]

    def _normalizar_dataframe(self, df):
        return df.fillna(VALOR_NULO).astype(str)

    def _gerar_classes_equivalencia(self):
        qi_columns = self._obter_qis()
        if not qi_columns:
            return None

        df_norm = self._normalizar_dataframe(self.df_anonimizado)
        return df_norm.groupby(qi_columns, dropna=False, sort=False)

    def calcular_k_anonymity(self):
        qi_columns = self._obter_qis()
        if not qi_columns:
            self._adicionar_warning(
                "Não foi possível calcular k-anonimato porque não há colunas QI."
            )
            return {
                "value": None,
                "group_count": 0,
                "min_class_size": None,
                "mean_class_size": None,
                "max_class_size": None,
                "equivalence_classes": [],
            }

        grupos = self._gerar_classes_equivalencia()
        tamanhos = grupos.size()
        classes = [
            {"group": self._serializar_chave_grupo(chave), "size": int(tamanho)}
            for chave, tamanho in tamanhos.items()
        ]

        return {
            "value": int(tamanhos.min()),
            "group_count": int(len(tamanhos)),
            "min_class_size": int(tamanhos.min()),
            "mean_class_size": float(tamanhos.mean()),
            "max_class_size": int(tamanhos.max()),
            "equivalence_classes": classes,
        }

    def calcular_l_diversity(self, l_alvo=2):
        qi_columns = self._obter_qis()
        sa_columns = self._obter_sas()
        if not qi_columns:
            self._adicionar_warning(
                "Não foi possível calcular l-diversidade porque não há colunas QI."
            )
            return self._resultado_l_vazio()
        if not sa_columns:
            self._adicionar_warning(
                "Não foi possível calcular l-diversidade porque não há colunas SA."
            )
            return self._resultado_l_vazio()

        df_norm = self._normalizar_dataframe(self.df_anonimizado)
        grupos = df_norm.groupby(qi_columns, dropna=False, sort=False)
        diversidades = []
        details = []

        for chave, grupo in grupos:
            por_coluna = {
                coluna: int(grupo[coluna].nunique(dropna=False))
                for coluna in sa_columns
            }
            menor_grupo = min(por_coluna.values())
            diversidades.append(menor_grupo)
            details.append(
                {
                    "group": self._serializar_chave_grupo(chave),
                    "min_diversity": int(menor_grupo),
                    "by_sensitive_column": por_coluna,
                }
            )

        falhas = sum(1 for valor in diversidades if valor < l_alvo)

        return {
            "value": int(min(diversidades)),
            "min_diversity": int(min(diversidades)),
            "mean_diversity": float(np.mean(diversidades)),
            "failing_groups": int(falhas),
            "group_count": int(len(diversidades)),
            "by_group": details,
        }

    def calcular_t_closeness(self):
        qi_columns = self._obter_qis()
        sa_columns = self._obter_sas()
        if not qi_columns:
            self._adicionar_warning("Não foi possível calcular t-closeness porque não há colunas QI.")
            return self._resultado_t_vazio()
        if not sa_columns:
            self._adicionar_warning("Não foi possível calcular t-closeness porque não há colunas SA.")
            return self._resultado_t_vazio()

        df_norm = self._normalizar_dataframe(self.df_anonimizado)
        grupos = df_norm.groupby(qi_columns, dropna=False, sort=False)
        distancias = []
        details = []
        pior_caso = None

        for coluna_sa in sa_columns:
            # Tenta converter para numérico para verificar se é um dado contínuo
            serie_original = self.df_anonimizado[coluna_sa]
            serie_numerica = pd.to_numeric(serie_original, errors="coerce")
            is_numeric = not serie_numerica.dropna().empty

            if is_numeric:
                # =========================================================
                # TRATAMENTO PARA DADOS CONTÍNUOS (NORMALIZAÇÃO MIN-MAX)
                # =========================================================
                min_val = serie_numerica.min()
                max_val = serie_numerica.max()
                
                # Normaliza para a escala [0, 1]
                if max_val > min_val:
                    serie_proc = (serie_numerica - min_val) / (max_val - min_val)
                else:
                    serie_proc = serie_numerica.fillna(0)
                
                distribuicao_global = serie_proc.dropna().values
                
                for chave, grupo_idx in grupos.groups.items():
                    # Pega apenas os valores numéricos deste grupo específico
                    distribuicao_grupo = serie_proc.loc[grupo_idx].dropna().values
                    
                    if len(distribuicao_grupo) > 0 and len(distribuicao_global) > 0:
                        # Calcula a Earth Mover's Distance já contida entre 0 e 1
                        distancia = wasserstein_distance(distribuicao_global, distribuicao_grupo)
                    else:
                        distancia = 0.0
                        
                    item = {
                        "group": self._serializar_chave_grupo(chave),
                        "sensitive_column": coluna_sa,
                        "distance": float(distancia),
                    }
                    details.append(item)
                    distancias.append(distancia)
                    if pior_caso is None or distancia > pior_caso["distance"]:
                        pior_caso = item
            else:
                # =========================================================
                # TRATAMENTO PARA DADOS CATEGÓRICOS/TEXTO (TVD)
                # =========================================================
                distribuicao_global = self._distribuicao(df_norm[coluna_sa])
                categorias = distribuicao_global.index.union(df_norm[coluna_sa].unique())

                for chave, grupo in grupos:
                    distribuicao_grupo = self._distribuicao(grupo[coluna_sa])
                    distancia = self._distancia_variacao_total(
                        distribuicao_global,
                        distribuicao_grupo,
                        categorias,
                    )
                    item = {
                        "group": self._serializar_chave_grupo(chave),
                        "sensitive_column": coluna_sa,
                        "distance": float(distancia),
                    }
                    details.append(item)
                    distancias.append(distancia)
                    if pior_caso is None or distancia > pior_caso["distance"]:
                        pior_caso = item

        valor = max(distancias) if distancias else None
        media = float(np.mean(distancias)) if distancias else None

        return {
            "value": float(valor) if valor is not None else None,
            "max_distance": float(valor) if valor is not None else None,
            "mean_distance": media,
            "worst_case": pior_caso,
            "by_group": details,
        }

    def calcular_utilidade(self):
        colunas_originais = list(self.df_original.columns)
        colunas_anonimizadas = list(self.df_anonimizado.columns)
        colunas_compartilhadas = [
            coluna for coluna in colunas_originais if coluna in colunas_anonimizadas
        ]
        colunas_removidas = [
            coluna for coluna in colunas_originais if coluna not in colunas_anonimizadas
        ]

        total_celulas_original = len(self.df_original) * len(colunas_originais)
        if total_celulas_original == 0:
            self._adicionar_warning("Não foi possível calcular utilidade em dataset vazio.")
            return {
                "value": None,
                "preserved_ratio": None,
                "removed_columns": colunas_removidas,
                "changed_cells": 0,
                "total_original_cells": 0,
            }

        linhas_comparaveis = min(len(self.df_original), len(self.df_anonimizado))
        celulas_preservadas = 0
        celulas_alteradas = 0

        for coluna in colunas_compartilhadas:
            original = self.df_original[coluna].iloc[:linhas_comparaveis]
            anonimizado = self.df_anonimizado[coluna].iloc[:linhas_comparaveis]
            comparacao = self._comparar_series(original, anonimizado)
            celulas_preservadas += int(comparacao.sum())
            celulas_alteradas += int((~comparacao).sum())

        celulas_removidas = len(self.df_original) * len(colunas_removidas)
        celulas_fora_comparacao = abs(len(self.df_original) - len(self.df_anonimizado)) * len(
            colunas_compartilhadas
        )
        celulas_alteradas += celulas_removidas + celulas_fora_comparacao

        utilidade = celulas_preservadas / total_celulas_original
        utilidade = self._limitar_entre_zero_um(utilidade)

        return {
            "value": float(utilidade),
            "preserved_ratio": float(utilidade),
            "removed_columns": colunas_removidas,
            "changed_cells": int(celulas_alteradas),
            "preserved_cells": int(celulas_preservadas),
            "total_original_cells": int(total_celulas_original),
        }

    def calcular_risco_adversarial(self, k_alvo=2, l_alvo=2):
        qi_columns = self._obter_qis()
        sa_columns = self._obter_sas()

        if not qi_columns:
            return []

        grupos = self._gerar_classes_equivalencia()

        tabela = []

        for chave, grupo in grupos:

            tamanho = len(grupo)

            diversidade = None

            if sa_columns:
                diversidade = min(
                    grupo[coluna].nunique(dropna=False)
                    for coluna in sa_columns
                )

            vulneravel = False

            if tamanho < k_alvo:
                vulneravel = True

            if diversidade is not None and diversidade < l_alvo:
                vulneravel = True

            tabela.append(
                {
                    "Classe": str(self._serializar_chave_grupo(chave)),
                    "Tamanho": int(tamanho),
                    "Diversidade": diversidade,
                    "Status": (
                        "VULNERAVEL"
                        if vulneravel
                        else "SEGURO"
                    ),
                }
            )
        return tabela

    def calcular_risco_reidentificacao(self, metrics, k_alvo=2, l_alvo=2, t_limite=0.2):
        k = metrics.get("k_anonymity")
        l = metrics.get("l_diversity")
        t = metrics.get("t_closeness")

        risco = 1.0 if k in (None, 0) else 1.0 / k

        if l is None:
            risco += 0.15
        elif l < l_alvo:
            risco += min(0.25, (l_alvo - l) / max(l_alvo, 1))

        if t is None:
            risco += 0.15
        elif t > t_limite:
            excesso = (t - t_limite) / max(1.0 - t_limite, 0.0001)
            risco += min(0.25, excesso)

        return float(self._limitar_entre_zero_um(risco))

    def _definir_status(self, metrics, k_alvo, l_alvo, t_limite):
        k = metrics["k_anonymity"]
        l = metrics["l_diversity"]
        t = metrics["t_closeness"]
        risco = metrics["reidentification_risk"]

        if k is None or l is None or t is None:
            return "indeterminado"

        criterios = [
            k >= k_alvo,
            l >= l_alvo,
            t <= t_limite,
            risco <= 0.33,
        ]
        if all(criterios):
            return "seguro"
        if any(criterios):
            return "parcialmente_seguro"
        return "nao_seguro"

    def _gerar_resumo(self, metrics, status, qi_columns, sa_columns):
        qis = ", ".join(qi_columns) if qi_columns else "nenhuma coluna QI"
        sas = ", ".join(sa_columns) if sa_columns else "nenhuma coluna SA"
        k = self._formatar_metrica(metrics["k_anonymity"])
        l = self._formatar_metrica(metrics["l_diversity"])
        t = self._formatar_metrica(metrics["t_closeness"])
        utilidade = self._formatar_percentual(metrics["utility_score"])
        risco = self._formatar_percentual(metrics["reidentification_risk"])
        nivel_risco = self._classificar_risco(metrics["reidentification_risk"])

        return (
            f"O dataset foi avaliado usando {qis} como QI e {sas} como SA. "
            f"Após a anonimização, atingiu k={k}, l={l} e t={t}. "
            f"A utilidade estimada foi de {utilidade}, com risco de "
            f"reidentificacao {nivel_risco} ({risco}). Status: {status}."
        )

    def _preparar_coluna_sensivel_para_t(self, coluna):
        serie = self.df_anonimizado[coluna]
        numerica = pd.to_numeric(serie, errors="coerce")
        valores_validos = numerica.dropna()

        if not valores_validos.empty and valores_validos.nunique() > 1:
            bins = min(10, int(valores_validos.nunique()))
            try:
                discretizada = pd.qcut(numerica, q=bins, duplicates="drop")
                return discretizada.astype(str).where(numerica.notna(), VALOR_NULO)
            except ValueError:
                return serie

        return serie

    def _distribuicao(self, serie):
        return serie.value_counts(normalize=True, dropna=False)

    def _distancia_variacao_total(self, distribuicao_global, distribuicao_grupo, categorias):
        global_alinhada = distribuicao_global.reindex(categorias, fill_value=0)
        grupo_alinhada = distribuicao_grupo.reindex(categorias, fill_value=0)
        distancia = 0.5 * np.abs(grupo_alinhada - global_alinhada).sum()
        return self._limitar_entre_zero_um(distancia)

    def _comparar_series(self, original, anonimizado):
        original_norm = original.fillna(VALOR_NULO).astype(str).reset_index(drop=True)
        anonimizado_norm = anonimizado.fillna(VALOR_NULO).astype(str).reset_index(drop=True)
        return original_norm == anonimizado_norm

    def _resultado_l_vazio(self):
        return {
            "value": None,
            "min_diversity": None,
            "mean_diversity": None,
            "failing_groups": None,
            "group_count": 0,
            "by_group": [],
        }

    def _resultado_t_vazio(self):
        return {
            "value": None,
            "max_distance": None,
            "mean_distance": None,
            "worst_case": None,
            "by_group": [],
        }

    def _serializar_chave_grupo(self, chave):
        if isinstance(chave, tuple):
            return [str(valor) for valor in chave]
        return [str(chave)]

    def _classificar_risco(self, risco):
        if risco is None or (isinstance(risco, float) and math.isnan(risco)):
            return "indeterminado"
        if risco <= 0.33:
            return "baixo"
        if risco <= 0.66:
            return "medio"
        return "alto"

    def _formatar_metrica(self, valor):
        if valor is None:
            return "N/A"
        if isinstance(valor, float):
            return f"{valor:.2f}"
        return str(valor)

    def _formatar_percentual(self, valor):
        if valor is None:
            return "N/A"
        return f"{valor * 100:.0f}%"

    def _limitar_entre_zero_um(self, valor):
        if valor is None or pd.isna(valor):
            return None
        return min(1.0, max(0.0, float(valor)))

    def _adicionar_warning(self, mensagem):
        self.warnings.append(mensagem)

    def _warnings_unicos(self):
        return list(dict.fromkeys(self.warnings))
