import re


class DataProfiler:
    DIRECT_IDENTIFIER_PATTERNS = (
        "nome",
        "name",
        "email",
        "mail",
        "cpf",
        "rg",
        "telefone",
        "celular",
        "phone",
        "contato",
        "endereco",
        "address",
        "logradouro",
    )

    QUASI_IDENTIFIER_PATTERNS = (
        "idade",
        "age",
        "genero",
        "sexo",
        "gender",
        "cep",
        "cidade",
        "city",
        "bairro",
        "estado",
        "uf",
        "nascimento",
        "birth",
    )

    SENSITIVE_PATTERNS = (
        "diagnostico",
        "doenca",
        "disease",
        "renda",
        "salario",
        "income",
        "religiao",
        "religion",
        "saude",
        "health",
        "tratamento",
        "prontuario",
        "medic",
    )

    def __init__(self, df):
        self.colunas = df.columns.tolist()
        self.df = df
        self.tipos_colunas = {
            coluna: self._inferir_tipo_coluna(coluna)
            for coluna in self.colunas
        }

    def atualizar_classificacao(self, dicionario_classificacoes):
        self.tipos_colunas.update(dicionario_classificacoes)

    def obter_colunas_por_tipo(self, tipo_coluna):
        """Retorna a lista de colunas de um tipo especifico (DI, QI, SA, NSA)."""
        return [
            coluna
            for coluna, tipo in self.tipos_colunas.items()
            if tipo == tipo_coluna
        ]

    def _inferir_tipo_coluna(self, nome_coluna):
        nome_normalizado = self._normalizar_nome_coluna(nome_coluna)

        if self._contem_padrao(nome_normalizado, self.DIRECT_IDENTIFIER_PATTERNS):
            return "DI"
        if self._contem_padrao(nome_normalizado, self.SENSITIVE_PATTERNS):
            return "SA"
        if self._contem_padrao(nome_normalizado, self.QUASI_IDENTIFIER_PATTERNS):
            return "QI"
        return "NSA"

    def _contem_padrao(self, nome_normalizado, padroes):
        return any(padrao in nome_normalizado for padrao in padroes)

    def _normalizar_nome_coluna(self, nome_coluna):
        texto = str(nome_coluna).strip().lower()
        texto = texto.translate(
            str.maketrans(
                {
                    "á": "a",
                    "à": "a",
                    "ã": "a",
                    "â": "a",
                    "é": "e",
                    "ê": "e",
                    "í": "i",
                    "ó": "o",
                    "ô": "o",
                    "õ": "o",
                    "ú": "u",
                    "ç": "c",
                }
            )
        )
        return re.sub(r"[^a-z0-9]+", "_", texto)
