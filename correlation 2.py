import pandas as pd
from scipy.stats import spearmanr
import numpy as np

# 1. Carregar a base de dados
df = pd.read_csv('dados/dados_tcc.csv')

# Renomear as colunas do CSV para um padrão 
df = df.rename(columns={
    'Folha_Salarial_Mi': 'Folha Salarial (R$ Milhões)',
    'Valor_Mercado_Mi': 'Valor de Mercado (R$ Milhões)',
    'Receita_Total_Mi': 'Receita Total (R$ Milhões)',
    'Gastos_Transferencias_Mi': 'Gastos com Transferências (R$ Milhões)'
})

# Transformando os NA do csv em valores nulos com pandas
colunas_numericas = ['Folha Salarial (R$ Milhões)', 'Valor de Mercado (R$ Milhões)', 'Receita Total (R$ Milhões)', 'Gastos com Transferências (R$ Milhões)', 'Pontos']
for col in colunas_numericas:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 2. Definir a variável contendo o desempenho desportivo
var_dependente = 'Pontos'

# 3. Definir as variáveis com os indicadores financeiros
metricas_financeiras = [
    'Folha Salarial (R$ Milhões)',
    'Valor de Mercado (R$ Milhões)',
    'Receita Total (R$ Milhões)',
    'Gastos com Transferências (R$ Milhões)'
]

# 4. Executar o teste de Spearman iterando sobre todas as métricas
print("RESULTADOS DO TESTE DE CORRELAÇÃO DE SPEARMAN (2020-2024)\n")

for metrica in metricas_financeiras:
    # Cria um dataframe temporário ignorando os NAs apenas no par analisado
    valid_df = df.dropna(subset=[metrica, var_dependente])
    
    coeficiente, p_value = spearmanr(valid_df[metrica], valid_df[var_dependente])
    
    print(f"Variável: {metrica}")
    print(f"Coeficiente (R): {coeficiente:.4f}")
    print(f"P-Value: {p_value:.4f}")
    print("-" * 50)