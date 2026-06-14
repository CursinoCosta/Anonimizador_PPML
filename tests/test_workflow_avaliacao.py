import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.avaliacao import (
    DEFAULT_EVALUATION_THRESHOLDS,
    ajustar_para_k_anonimato,
    atualizar_resultado_avaliacao,
    identificar_upload,
    inicializar_thresholds_avaliacao,
    sincronizar_estado_dataset,
)


class FakeUpload:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


class WorkflowAvaliacaoTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "idade": [20, 20, 30, 30],
                "cidade": ["BH", "BH", "SP", "SP"],
                "doenca": ["A", "B", "A", "B"],
            }
        )

    def test_identificar_upload_muda_com_conteudo(self):
        upload_a = FakeUpload("base.csv", b"a,b\n1,2\n")
        upload_b = FakeUpload("base.csv", b"a,b\n1,3\n")

        self.assertNotEqual(identificar_upload(upload_a), identificar_upload(upload_b))

    def test_sincronizar_estado_dataset_recria_estado_quando_upload_muda(self):
        session_state = {}
        upload_1 = "arquivo-1"
        upload_2 = "arquivo-2"

        mudou = sincronizar_estado_dataset(session_state, self.df, upload_1)
        profiler_inicial = session_state["profiler"]
        anonymizer_inicial = session_state["anonymizer"]

        self.assertTrue(mudou)

        mudou = sincronizar_estado_dataset(session_state, self.df, upload_2)

        self.assertTrue(mudou)
        self.assertIsNot(session_state["profiler"], profiler_inicial)
        self.assertIsNot(session_state["anonymizer"], anonymizer_inicial)
        self.assertIsNone(session_state["evaluation_result"])

    def test_sincronizar_estado_dataset_preserva_estado_se_upload_igual(self):
        session_state = {}
        upload_id = "arquivo-1"

        sincronizar_estado_dataset(session_state, self.df, upload_id)
        profiler_inicial = session_state["profiler"]
        anonymizer_inicial = session_state["anonymizer"]

        mudou = sincronizar_estado_dataset(session_state, self.df, upload_id)

        self.assertFalse(mudou)
        self.assertIs(session_state["profiler"], profiler_inicial)
        self.assertIs(session_state["anonymizer"], anonymizer_inicial)

    def test_atualizar_resultado_avaliacao_salva_resultado(self):
        session_state = {}
        sincronizar_estado_dataset(session_state, self.df, "arquivo-1")
        session_state["profiler"].atualizar_classificacao(
            {"idade": "QI", "cidade": "QI", "doenca": "SA"}
        )

        resultado = atualizar_resultado_avaliacao(
            session_state,
            DEFAULT_EVALUATION_THRESHOLDS,
        )

        self.assertIs(session_state["evaluation_result"], resultado)
        self.assertEqual(resultado["metrics"]["k_anonymity"], 2)
        self.assertEqual(resultado["metrics"]["l_diversity"], 2)

    def test_inicializar_thresholds_preenche_defaults(self):
        session_state = {}

        thresholds = inicializar_thresholds_avaliacao(session_state)

        self.assertEqual(thresholds, DEFAULT_EVALUATION_THRESHOLDS)
        self.assertEqual(session_state["evaluation_thresholds"], DEFAULT_EVALUATION_THRESHOLDS)

    def test_mudanca_de_thresholds_recalcula_resultado(self):
        session_state = {}
        sincronizar_estado_dataset(session_state, self.df, "arquivo-1")
        session_state["profiler"].atualizar_classificacao(
            {"idade": "QI", "cidade": "QI", "doenca": "SA"}
        )

        resultado_padrao = atualizar_resultado_avaliacao(
            session_state,
            {"k_alvo": 2, "l_alvo": 2, "t_limite": 0.20},
        )
        resultado_mais_rigido = atualizar_resultado_avaliacao(
            session_state,
            {"k_alvo": 3, "l_alvo": 2, "t_limite": 0.20},
        )

        self.assertEqual(resultado_padrao["status"], "parcialmente_seguro")
        self.assertEqual(resultado_mais_rigido["status"], "parcialmente_seguro")
        self.assertEqual(session_state["evaluation_thresholds"]["k_alvo"], 3)

    def test_ajustar_para_k_sem_qi_retorna_warning(self):
        session_state = {}
        sincronizar_estado_dataset(session_state, self.df, "arquivo-1")
        session_state["profiler"].atualizar_classificacao({"doenca": "SA"})

        resumo = ajustar_para_k_anonimato(session_state, 2)

        self.assertFalse(resumo["atingiu_alvo"])
        self.assertTrue(resumo["warnings"])

    def test_ajustar_para_k_numerico_aumenta_ou_preserva_k(self):
        df = pd.DataFrame(
            {
                "idade": [21, 23, 31, 33],
                "cidade": ["BH", "BH", "SP", "SP"],
                "doenca": ["A", "B", "A", "B"],
            }
        )
        session_state = {}
        sincronizar_estado_dataset(session_state, df, "arquivo-1")
        session_state["profiler"].atualizar_classificacao(
            {"idade": "QI", "cidade": "QI", "doenca": "SA"}
        )

        antes = atualizar_resultado_avaliacao(session_state, DEFAULT_EVALUATION_THRESHOLDS)
        resumo = ajustar_para_k_anonimato(session_state, 2)
        depois = session_state["evaluation_result"]

        self.assertGreaterEqual(
            depois["metrics"]["k_anonymity"],
            antes["metrics"]["k_anonymity"],
        )
        self.assertTrue(
            any(step["type"] == "numeric_range" for step in resumo["steps_applied"])
        )

    def test_ajustar_para_k_textual_aplica_mascara_progressiva(self):
        df = pd.DataFrame(
            {
                "bairro": ["CentroA", "CentroB", "NorteA", "NorteB"],
                "cidade": ["BH", "BH", "SP", "SP"],
                "doenca": ["A", "B", "A", "B"],
            }
        )
        session_state = {}
        sincronizar_estado_dataset(session_state, df, "arquivo-1")
        session_state["profiler"].atualizar_classificacao(
            {"bairro": "QI", "cidade": "QI", "doenca": "SA"}
        )

        resumo = ajustar_para_k_anonimato(session_state, 2)

        self.assertTrue(
            any(step["type"] == "text_mask" for step in resumo["steps_applied"])
        )

    def test_ajuste_para_k_para_em_quatro_rodadas(self):
        df = pd.DataFrame(
            {
                "idade": [10, 21, 32, 43, 54, 65],
                "bairro": ["A1", "B2", "C3", "D4", "E5", "F6"],
                "doenca": ["A", "B", "C", "D", "E", "F"],
            }
        )
        session_state = {}
        sincronizar_estado_dataset(session_state, df, "arquivo-1")
        session_state["profiler"].atualizar_classificacao(
            {"idade": "QI", "bairro": "QI", "doenca": "SA"}
        )

        resumo = ajustar_para_k_anonimato(session_state, 4)

        self.assertLessEqual(len(resumo["steps_applied"]), 8)
        self.assertIn("k_final", resumo)


if __name__ == "__main__":
    unittest.main()
