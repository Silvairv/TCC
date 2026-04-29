import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Configuração da Página ---
st.set_page_config(
    page_title="Dashboard Finanças e Desempenho",
    page_icon="⚽",
    layout="wide",
)

# --- Camada de Dados: Carregamento ---
@st.cache_data
def carregar_dados():
    try:
        # Lê o arquivo CSV atualizado com colunas SAF e valores NA
        df = pd.read_csv('dados/dados_tcc.csv')
        
        # Renomeia para exibição amigável no Dashboard
        df = df.rename(columns={
            'Folha_Salarial_Mi': 'Folha Salarial (R$ Milhões)',
            'Valor_Mercado_Mi': 'Valor de Mercado (R$ Milhões)',
            'Receita_Total_Mi': 'Receita Total (R$ Milhões)',
            'Gastos_Transferencias_Mi': 'Gastos com Transferências (R$ Milhões)',
            'Pontos': 'Pontuação Final'
        })
        
        # Converte para numérico (NA vira NaN automaticamente)
        colunas_numericas = [
            'Folha Salarial (R$ Milhões)', 'Valor de Mercado (R$ Milhões)', 
            'Receita Total (R$ Milhões)', 'Gastos com Transferências (R$ Milhões)', 
            'Pontuação Final'
        ]
        
        for col in colunas_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Erro ao carregar a base de dados: {e}")
        return pd.DataFrame()

df = carregar_dados()

if df.empty:
    st.stop()

# Barra Lateral 
st.sidebar.header("🔍 Filtros de Análise")

metricas_disponiveis = [
    'Valor de Mercado (R$ Milhões)', 'Folha Salarial (R$ Milhões)',
    'Gastos com Transferências (R$ Milhões)', 'Receita Total (R$ Milhões)'
]
metrica_selecionada = st.sidebar.selectbox("Métrica Financeira (Eixo X):", metricas_disponiveis)
var_dependente = 'Pontuação Final'

# Filtro de Ano
anos_disponiveis = sorted(df['Ano'].dropna().unique())
anos_selecionados = st.sidebar.multiselect("Ano", anos_disponiveis, default=anos_disponiveis)

# Camada de Negócio: Processamento
df_filtrado = df[(df['Ano'].isin(anos_selecionados))]
df_filtrado = df_filtrado.dropna(subset=[metrica_selecionada, var_dependente])

# (K-Means)
if len(df_filtrado) >= 4:
    X = df_filtrado[[metrica_selecionada, var_dependente]].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_filtrado['Cluster_ID'] = kmeans.fit_predict(X_scaled)
    
    # Nomeação dinâmica dos clusters por desempenho
    media_fin_global = df_filtrado[metrica_selecionada].mean()
    media_pts_global = df_filtrado[var_dependente].mean()
    cluster_centers = df_filtrado.groupby('Cluster_ID')[[metrica_selecionada, var_dependente]].mean()
    
    categorias_map = {}
    for c_id in cluster_centers.index:
        fin = cluster_centers.loc[c_id, metrica_selecionada]
        pts = cluster_centers.loc[c_id, var_dependente]
        if fin >= media_fin_global and pts >= media_pts_global: perfil = "Potências"
        elif fin >= media_fin_global and pts < media_pts_global: perfil = "Ricos/Ineficientes"
        elif fin < media_fin_global and pts >= media_pts_global: perfil = "Milagres/Eficientes"
        else: perfil = "Vulneráveis"
        categorias_map[c_id] = perfil
        
    df_filtrado['Categoria'] = df_filtrado['Cluster_ID'].map(categorias_map)

# Cabeçalho e KPIs
st.title("⚽ Finanças e Desempenho no Futebol Brasileiro")
st.markdown("Análise estatística e de Machine Learning sobre o impacto da gestão financeira nos resultados esportivos.")

if not df_filtrado.empty:
    coef, p_val = spearmanr(df_filtrado[metrica_selecionada], df_filtrado[var_dependente])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spearman (R)", f"{coef:.4f}")
    c2.metric("P-Value", "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}")
    c3.metric("Média Financeira", f"R$ {df_filtrado[metrica_selecionada].mean():,.1f} Mi")
    c4.metric("Média de Pontos", f"{df_filtrado[var_dependente].mean():.1f}")

st.markdown("---")

# Menu Superior
m1, m2, m3, m4 = st.tabs(["🏠 Visão Geral", "🤖 K-Means", "🏢 Impacto SAF", "📋 Dados"])

with m1:
    col_a, col_b = st.columns(2)
    with col_a:
        fig_disp = px.scatter(df_filtrado, x=metrica_selecionada, y=var_dependente, 
                             hover_name='Clube', trendline="ols", color=var_dependente,
                             color_continuous_scale='Viridis', title="Correlação Financeiro vs Pontos")
        st.plotly_chart(fig_disp, use_container_width=True)
    with col_b:
        fig_hist = px.histogram(df_filtrado, x=var_dependente, title="Distribuição da Pontuação", color_discrete_sequence=['#2ca02c'])
        st.plotly_chart(fig_hist, use_container_width=True)

with m2:
    st.subheader("Clustering K-Means: Agrupamento por Perfil de Eficiência")
    col_ia1, col_ia2 = st.columns([2, 1])
    with col_ia1:
        fig_k = px.scatter(df_filtrado, x=metrica_selecionada, y=var_dependente, color='Categoria',
                          hover_name='Clube', title="Clusters Identificados")
        st.plotly_chart(fig_k, use_container_width=True)
    with col_ia2:
        fig_pie = px.pie(df_filtrado, names='Categoria', title="Distribuição de Perfis", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

with m3:
    st.subheader("Análise Comparativa: Clubes SAF vs Associativos")
    if 'SAF' in df_filtrado.columns:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_box = px.box(df_filtrado, x='SAF', y=var_dependente, color='SAF', title="Pontuação por Modelo de Gestão")
            st.plotly_chart(fig_box, use_container_width=True)
        with col_s2:
            fig_bar_saf = px.bar(df_filtrado.groupby('SAF')[metrica_selecionada].mean().reset_index(), 
                                x='SAF', y=metrica_selecionada, color='SAF', title="Média de Investimento")
            st.plotly_chart(fig_bar_saf, use_container_width=True)
    else:
        st.warning("Coluna SAF não detectada no CSV.")

with m4:
    st.dataframe(df_filtrado, use_container_width=True)
    csv_data = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Baixar Base Processada (CSV)", data=csv_data, file_name='tcc_processado.csv')