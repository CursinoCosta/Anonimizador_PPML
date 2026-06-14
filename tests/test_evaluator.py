import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.evaluator import Evaluator


class FakeProfiler:
    def __init__(self, tipos_colunas):
        self.tipos_colunas = tipos_colunas

    def obter_colunas_por_tipo(self, tipo_coluna):
        return [
            coluna
            for coluna, tipo in self.tipos_colunas.items()
            if tipo == tipo_coluna
        ]


class EvaluatorTest(unittest.TestCase):
    def test_avalia_dataset_com_qi_e_sa_validos(self):
        df = pd.DataFrame(
            {
                "idade": [20, 20, 30, 30],
                "cidade": ["BH", "BH", "SP", "SP"],
                "doenca": ["A", "B", "A", "B"],
            }
        )
        profiler = FakeProfiler({"idade": "QI", "cidade": "QI", "doenca": "SA"})

        resultado = Evaluator(profiler, df, df).avaliar(k_alvo=2, l_alvo=2)

        self.assertEqual(resultado["metrics"]["k_anonymity"], 2)
        self.assertEqual(resultado["metrics"]["l_diversity"], 2)
        self.assertEqual(resultado["status"], "parcialmente_seguro")
        self.assertEqual(resultado["warnings"], [])

    def test_sem_qi_retorna_metricas_indeterminadas(self):
        df = pd.DataFrame({"doenca": ["A", "B", "A"]})
        profiler = FakeProfiler({"doenca": "SA"})

        resultado = Evaluator(profiler, df, df).avaliar()

        self.assertIsNone(resultado["metrics"]["k_anonymity"])
        self.assertIsNone(resultado["metrics"]["l_diversity"])
        self.assertIsNone(resultado["metrics"]["t_closeness"])
        self.assertEqual(resultado["status"], "indeterminado")
        self.assertTrue(any("QI" in aviso for aviso in resultado["warnings"]))

    def test_sem_sa_mantem_k_e_indetermina_l_e_t(self):
        df = pd.DataFrame({"idade": [20, 20, 30], "cidade": ["BH", "BH", "SP"]})
        profiler = FakeProfiler({"idade": "QI", "cidade": "QI"})

        resultado = Evaluator(profiler, df, df).avaliar()

        self.assertEqual(resultado["metrics"]["k_anonymity"], 1)
        self.assertIsNone(resultado["metrics"]["l_diversity"])
        self.assertIsNone(resultado["metrics"]["t_closeness"])
        self.assertEqual(resultado["status"], "indeterminado")
        self.assertTrue(any("SA" in aviso for aviso in resultado["warnings"]))

    def test_grupos_de_tamanho_um_geram_k_igual_um(self):
        df = pd.DataFrame(
            {
                "idade": [20, 21, 22],
                "cidade": ["BH", "SP", "RJ"],
                "doenca": ["A", "B", "C"],
            }
        )
        profiler = FakeProfiler({"idade": "QI", "cidade": "QI", "doenca": "SA"})

        resultado = Evaluator(profiler, df, df).avaliar()

        self.assertEqual(resultado["metrics"]["k_anonymity"], 1)
        self.assertEqual(resultado["metrics"]["l_diversity"], 1)
        self.assertEqual(resultado["status"], "nao_seguro")

    def test_coluna_sensivel_homogenea_gera_l_igual_um(self):
        df = pd.DataFrame(
            {
                "idade": [20, 20, 30, 30],
                "cidade": ["BH", "BH", "SP", "SP"],
                "doenca": ["A", "A", "A", "A"],
            }
        )
        profiler = FakeProfiler({"idade": "QI", "cidade": "QI", "doenca": "SA"})

        resultado = Evaluator(profiler, df, df).avaliar(l_alvo=2)

        self.assertEqual(resultado["metrics"]["l_diversity"], 1)
        self.assertEqual(resultado["details"]["l_diversity"]["failing_groups"], 2)

    def test_utilidade_penaliza_coluna_suprimida(self):
        df_original = pd.DataFrame(
            {
                "nome": ["Ana", "Bia"],
                "idade": [20, 30],
                "doenca": ["A", "B"],
            }
        )
        df_anonimizado = df_original.drop(columns=["nome"])
        profiler = FakeProfiler({"nome": "DI", "idade": "QI", "doenca": "SA"})

        resultado = Evaluator(profiler, df_original, df_anonimizado).avaliar()

        self.assertAlmostEqual(resultado["metrics"]["utility_score"], 4 / 6)
        self.assertEqual(resultado["details"]["utility"]["removed_columns"], ["nome"])

    def test_t_closeness_funciona_com_atributo_sensivel_numerico(self):
        df = pd.DataFrame(
            {
                "idade": [20, 20, 30, 30, 40, 40],
                "cidade": ["BH", "BH", "SP", "SP", "RJ", "RJ"],
                "renda": [1000, 1100, 5000, 5100, 9000, 9100],
            }
        )
        profiler = FakeProfiler({"idade": "QI", "cidade": "QI", "renda": "SA"})

        resultado = Evaluator(profiler, df, df).avaliar()
        t_closeness = resultado["metrics"]["t_closeness"]

        self.assertIsNotNone(t_closeness)
        self.assertGreaterEqual(t_closeness, 0)
        self.assertLessEqual(t_closeness, 1)


if __name__ == "__main__":
    unittest.main()
