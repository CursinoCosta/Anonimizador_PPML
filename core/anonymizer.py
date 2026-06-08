import pandas as pd
import numpy as np

class Anonymizer:
    def __init__(self, profiler):
        self.profiler = profiler
        self.df_anonimizado = profiler.df.copy()

    # --- SUPRESSÃO ---
    
    def suprimir_colunas(self, colunas_para_suprimir):
        """Remove colunas inteiras do dataset."""
        if colunas_para_suprimir:
            self.df_anonimizado = self.df_anonimizado.drop(columns=colunas_para_suprimir, errors='ignore')
        return self.df_anonimizado

    def suprimir_celulas_manualmente(self, coluna, indices):
        """Substitui o valor de células específicas por '***' com base no índice da linha."""
        if coluna in self.df_anonimizado.columns and indices:
            self.df_anonimizado.loc[indices, coluna] = "***"
        return self.df_anonimizado

    def suprimir_celulas_por_regra(self, coluna, palavras_proibidas):
        """Substitui a célula por '***' se contiver alguma das palavras listadas."""
        if coluna in self.df_anonimizado.columns and palavras_proibidas:
            # Escapa as palavras para evitar problemas com regex e junta com OR (|)
            padrao = '|'.join([pd.Series(p.strip()).str.escape() for p in palavras_proibidas if p.strip()])
            if padrao:
                mascara = self.df_anonimizado[coluna].astype(str).str.contains(padrao, case=False, na=False)
                self.df_anonimizado.loc[mascara, coluna] = "***"
        return self.df_anonimizado

    # --- GENERALIZAÇÃO ---
    
    def generalizar_por_mascara(self, coluna, num_chars, direcao):
        """
        Substitui caracteres por '*'. 
        direcao pode ser 'direita_para_esquerda' ou 'esquerda_para_direita'.
        """
        if coluna not in self.df_anonimizado.columns:
            return self.df_anonimizado
            
        def aplicar_mascara(valor):
            val_str = str(valor)
            if pd.isna(valor) or val_str == 'nan':
                return valor
                
            if len(val_str) <= num_chars:
                return "*" * len(val_str)
                
            if direcao == "direita_para_esquerda":
                # Ex: 00000000 -> 000000** (oculta os N últimos)
                return val_str[:-num_chars] + "*" * num_chars
            else:
                # Ex: 00000000 -> **000000 (oculta os N primeiros)
                return "*" * num_chars + val_str[num_chars:]
                
        self.df_anonimizado[coluna] = self.df_anonimizado[coluna].apply(aplicar_mascara)
        return self.df_anonimizado

    def generalizar_por_hierarquia(self, coluna, dicionario_mapeamento):
        """
        Substitui os valores da coluna com base num dicionário de hierarquia.
        Ex: {'BH': 'MG', 'Uberlandia': 'MG'}
        """
        if coluna in self.df_anonimizado.columns and dicionario_mapeamento:
            # O fillna garante que, se um valor não estiver no dicionário, ele mantenha o original
            self.df_anonimizado[coluna] = self.df_anonimizado[coluna].map(dicionario_mapeamento).fillna(self.df_anonimizado[coluna])
        return self.df_anonimizado
    
    def generalizar_por_faixas(self, coluna, tamanho_faixa):
        """
        Agrupa valores numéricos em faixas regulares.
        Ex: tamanho_faixa=3 transforma valores 0, 1, 2 no texto '[0-2]'.
        """
        if coluna in self.df_anonimizado.columns:
            # Converte a coluna para numérico para possibilitar a matemática de faixas
            self.df_anonimizado[coluna] = pd.to_numeric(self.df_anonimizado[coluna], errors='coerce')
            
            val_maximo = self.df_anonimizado[coluna].max()
            if pd.isna(val_maximo):
                return self.df_anonimizado
                
            # Cria os limites. Ex: se max é 10 e tamanho 3 -> [0, 3, 6, 9, 12]
            limites = np.arange(0, val_maximo + tamanho_faixa + 1, tamanho_faixa)
            
            # Cria os rótulos textuais baseados nos limites
            rotulos = [f"[{int(b)}-{int(b+tamanho_faixa-1)}]" for b in limites[:-1]]
            
            # Aplica a divisão
            self.df_anonimizado[coluna] = pd.cut(
                self.df_anonimizado[coluna], 
                bins=limites, 
                labels=rotulos, 
                right=False # Inclui a esquerda, exclui a direita (Ex: 0 a 2.99 vira [0-2])
            )
            
            # Converte o resultado final de volta para string
            self.df_anonimizado[coluna] = self.df_anonimizado[coluna].astype(str)

        return self.df_anonimizado
    
    # --- PERTURBAÇÃO ---

    def adicionar_ruido(self, coluna, distribuicao="Normal", media=0.0, desvio_padrao=1.0, limite_inf=-1.0, limite_sup=1.0, casas_decimais=None):
        """
        Adiciona ruído aleatório a uma coluna numérica.
        Pode utilizar distribuição Normal (Gaussiana) ou Uniforme.
        Permite arredondar as casas decimais do resultado final.
        """
        if coluna in self.df_anonimizado.columns:
            # Força a conversão para numérico
            self.df_anonimizado[coluna] = pd.to_numeric(self.df_anonimizado[coluna], errors='coerce')
            
            n_linhas = len(self.df_anonimizado)
            
            if distribuicao == "Normal":
                ruido = np.random.normal(loc=media, scale=desvio_padrao, size=n_linhas)
            else:
                ruido = np.random.uniform(low=limite_inf, high=limite_sup, size=n_linhas)
                
            self.df_anonimizado[coluna] = self.df_anonimizado[coluna] + ruido

            # Arredonda se o usuário tiver definido o número de casas decimais
            if casas_decimais is not None:
                self.df_anonimizado[coluna] = self.df_anonimizado[coluna].round(casas_decimais)
            
        return self.df_anonimizado

    def permutar_dados(self, coluna_alvo, colunas_particao=None):
        """
        Embaralha (permuta) os valores de uma coluna dentro de grupos definidos 
        por 'colunas_particao'. Se não houver partição, embaralha a coluna inteira.
        Ideal para quebrar a ligação entre QIs e o Atributo Sensível (SA).
        """
        if coluna_alvo not in self.df_anonimizado.columns:
            return self.df_anonimizado

        if colunas_particao and all(c in self.df_anonimizado.columns for c in colunas_particao):
            # Embaralha os dados APENAS dentro dos grupos (blocos) usando lambda
            self.df_anonimizado[coluna_alvo] = self.df_anonimizado.groupby(colunas_particao)[coluna_alvo].transform(lambda x: np.random.permutation(x.values))
        else:
            # Embaralha a coluna inteira (partição única)
            self.df_anonimizado[coluna_alvo] = np.random.permutation(self.df_anonimizado[coluna_alvo].values)

        return self.df_anonimizado