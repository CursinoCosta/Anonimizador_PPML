class DataProfiler:
    def __init__(self, df):
        self.colunas = df.columns.tolist()
        # Inicializa todas as colunas como Não Sensíveis (NSA) por padrão
        self.tipos_colunas = {col: 'NSA' for col in self.colunas}
        self.df = df
        
    def atualizar_classificacao(self, dicionario_classificacoes):
        """Atualiza a classificação com base nas entradas do utilizador."""
        self.tipos_colunas.update(dicionario_classificacoes)
        
    def obter_colunas_por_tipo(self, tipo_coluna):
        """Retorna a lista de colunas de um tipo específico (DI, QI, SA, NSA)."""
        return [col for col, tipo in self.tipos_colunas.items() if tipo == tipo_coluna]