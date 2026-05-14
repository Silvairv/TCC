import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr, normaltest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Finanças & Desempenho — Futebol Brasileiro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Fundo geral */
    .stApp { background-color: #0e1117; }

    /* Cards de KPI */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 1px solid #2d3548;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] label {
        color: #8892a4 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }

    /* Cabeçalho */
    .header-container {
        background: linear-gradient(135deg, #1a2744, #0d3b6e);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        border: 1px solid #2a4a7f;
    }
    .header-container h1 {
        color: #e2e8f0;
        font-size: 2rem;
        margin: 0 0 6px 0;
    }
    .header-container p {
        color: #94a3b8;
        margin: 0;
        font-size: 0.95rem;
    }

    /* Caixas de interpretação */
    .insight-box {
        background: #1e2433;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .insight-box.green  { border-color: #22c55e; }
    .insight-box.yellow { border-color: #f59e0b; }
    .insight-box.red    { border-color: #ef4444; }
    .insight-box strong { color: #e2e8f0; }

    /* Seções */
    .section-title {
        color: #94a3b8;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 10px;
    }

    /* Abas */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1f2e;
        border-radius: 10px;
        padding: 4px;
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8892a4;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2d3548 !important;
        color: #e2e8f0 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #13161f; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CAMADA DE DADOS
# ─────────────────────────────────────────────
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv('dados/dados_tcc.csv')
        df = df.rename(columns={
            'Folha_Salarial_Mi':        'Folha Salarial (R$ Mi)',
            'Valor_Mercado_Mi':         'Valor de Mercado (R$ Mi)',
            'Receita_Total_Mi':         'Receita Total (R$ Mi)',
            'Gastos_Transferencias_Mi': 'Gastos com Transferências (R$ Mi)',
            'Pontos':                   'Pontuação Final',
        })
        cols_num = [
            'Folha Salarial (R$ Mi)', 'Valor de Mercado (R$ Mi)',
            'Receita Total (R$ Mi)', 'Gastos com Transferências (R$ Mi)',
            'Pontuação Final',
        ]
        for c in cols_num:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

df = carregar_dados()
if df.empty:
    st.stop()

# ─────────────────────────────────────────────
# BARRA LATERAL
# ─────────────────────────────────────────────
st.sidebar.markdown("## ⚽ Dashboard Futebol")
st.sidebar.markdown("---")

METRICAS = [
    'Valor de Mercado (R$ Mi)',
    'Folha Salarial (R$ Mi)',
    'Gastos com Transferências (R$ Mi)',
    'Receita Total (R$ Mi)',
]
metrica = st.sidebar.selectbox("📊 Métrica Financeira (Eixo X)", METRICAS)
var_dep = 'Pontuação Final'

anos_disp = sorted(df['Ano'].dropna().unique().astype(int))
anos_sel  = st.sidebar.multiselect("📅 Ano(s)", anos_disp, default=anos_disp)

clubes_disp = sorted(df['Clube'].dropna().unique())
clubes_sel  = st.sidebar.multiselect("🏟️ Clube(s)", clubes_disp, default=clubes_disp)

st.sidebar.markdown("---")

# ─────────────────────────────────────────────
# PROCESSAMENTO / FILTRAGEM
# ─────────────────────────────────────────────
df_f = df[df['Ano'].isin(anos_sel) & df['Clube'].isin(clubes_sel)].copy()
df_clean = df_f.dropna(subset=[metrica, var_dep])

# K-Means com 6 clusters
N_CLUSTERS = 6
if len(df_clean) >= N_CLUSTERS:
    X = df_clean[[metrica, var_dep]].copy()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df_clean['Cluster_ID'] = km.fit_predict(X_s)

    # Percentis para segmentação mais granular (33/66 em vez de média simples)
    p33_fin = df_clean[metrica].quantile(0.33)
    p66_fin = df_clean[metrica].quantile(0.66)
    p33_pts = df_clean[var_dep].quantile(0.33)
    p66_pts = df_clean[var_dep].quantile(0.66)

    centros = df_clean.groupby('Cluster_ID')[[metrica, var_dep]].mean()

    cat_map = {}
    for cid in centros.index:
        f, p = centros.loc[cid, metrica], centros.loc[cid, var_dep]
        # Faixa financeira: Alta / Média / Baixa
        if   f >= p66_fin: faixa_fin = "alta"
        elif f >= p33_fin: faixa_fin = "media"
        else:              faixa_fin = "baixa"
        # Faixa esportiva: Alta / Média / Baixa
        if   p >= p66_pts: faixa_pts = "alta"
        elif p >= p33_pts: faixa_pts = "media"
        else:              faixa_pts = "baixa"

        if   faixa_fin == "alta"  and faixa_pts == "alta":   perfil = "🏆 Elite"
        elif faixa_fin == "alta"  and faixa_pts == "media":  perfil = "💰 Investidores"
        elif faixa_fin == "alta"  and faixa_pts == "baixa":  perfil = "💸 Ricos/Ineficientes"
        elif faixa_fin == "media" and faixa_pts == "alta":   perfil = "✨ Eficientes"
        elif faixa_fin == "media" and faixa_pts == "media":  perfil = "⚖️ Equilíbrados"
        else:                                                  perfil = "⚠️ Vulneráveis"

        cat_map[cid] = perfil

    df_clean['Perfil'] = df_clean['Cluster_ID'].map(cat_map)

# Spearman em todas as métricas
resultados_spearman = {}
for m in METRICAS:
    tmp = df_f.dropna(subset=[m, var_dep])
    if len(tmp) > 4:
        coef, pv = spearmanr(tmp[m], tmp[var_dep])
        resultados_spearman[m] = {'R': coef, 'p': pv, 'n': len(tmp)}

# Regressão linear simples (para linha de tendência manual)
def regressao(x_ser, y_ser):
    m_ = np.isfinite(x_ser) & np.isfinite(y_ser)
    x_, y_ = x_ser[m_].values.reshape(-1,1), y_ser[m_].values
    reg = LinearRegression().fit(x_, y_)
    return reg.coef_[0], reg.intercept_, reg.score(x_, y_)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def interpreta_spearman(r, p):
    if p >= 0.05:
        return "red", "Correlação **não significativa** (p ≥ 0.05). Não há evidência estatística de relação entre esta métrica e a pontuação no período selecionado."
    mag = abs(r)
    if   mag >= 0.70: forca, cor = "muito forte",   "green"
    elif mag >= 0.50: forca, cor = "forte",          "green"
    elif mag >= 0.30: forca, cor = "moderada",       "yellow"
    else:             forca, cor = "fraca",           "yellow"
    direcao = "positiva" if r > 0 else "negativa"
    return cor, (f"Correlação de Spearman **{forca}** e **{direcao}** (R = {r:.3f}, p < 0.05). "
                 f"Clubes com maior **{metrica}** tendem a {'pontuar mais' if r>0 else 'pontuar menos'} no Campeonato Brasileiro — "
                 f"resultado consistente com a literatura (Dantas et al., 2015; Cruz et al., 2022).")

def cor_perfil():
    return {
        "🏆 Elite":              "#3b82f6",
        "💰 Investidores":       "#8b5cf6",
        "💸 Ricos/Ineficientes": "#f59e0b",
        "✨ Eficientes":         "#22c55e",
        "⚖️ Equilíbrados":      "#06b6d4",
        "⚠️ Vulneráveis":       "#ef4444",
    }

PALETTE = {"background": "#0e1117", "card": "#1a1f2e", "border": "#2d3548",
           "text": "#e2e8f0", "subtext": "#8892a4", "accent": "#3b82f6"}

def fig_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(color=PALETTE["text"], size=15), x=0.01),
        paper_bgcolor=PALETTE["card"],
        plot_bgcolor="#12172a",
        font=dict(color=PALETTE["subtext"], size=12),
        margin=dict(t=50, b=40, l=40, r=20),
        legend=dict(bgcolor="#12172a", bordercolor=PALETTE["border"], borderwidth=1,
                    font=dict(color=PALETTE["text"])),
        xaxis=dict(gridcolor="#1e2433", zerolinecolor="#2d3548", color=PALETTE["subtext"]),
        yaxis=dict(gridcolor="#1e2433", zerolinecolor="#2d3548", color=PALETTE["subtext"]),
    )
    return fig

# ─────────────────────────────────────────────
# CABEÇALHO
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-container">
  <h1>⚽ Finanças & Desempenho no Futebol Brasileiro</h1>
  <p>Análise estatística e de Machine Learning — Campeonato Brasileiro Série A (2020–2024)</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────
if not df_clean.empty:
    coef_kpi, pval_kpi = spearmanr(df_clean[metrica], df_clean[var_dep])
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Spearman (R)",       f"{coef_kpi:.4f}",
              delta="Sig. ✓" if pval_kpi < 0.05 else "Não sig. ✗",
              delta_color="normal" if pval_kpi < 0.05 else "off")
    k2.metric("P-Value",            "< 0.0001" if pval_kpi < 0.0001 else f"{pval_kpi:.4f}")
    k3.metric("Obs. válidas",       f"{len(df_clean)}")
    k4.metric("Média Financeira",   f"R$ {df_clean[metrica].mean():,.1f} Mi")
    k5.metric("Média de Pontos",    f"{df_clean[var_dep].mean():.1f}")

    # Bloco de interpretação principal
    cor_i, texto_i = interpreta_spearman(coef_kpi, pval_kpi)
    st.markdown(f'<div class="insight-box {cor_i}">💡 <strong>Interpretação:</strong> {texto_i}</div>',
                unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# ABAS
# ─────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "🏠 Visão Geral",
    "📈 Correlações",
    "🤖 K-Means",
    "🏢 Impacto SAF",
    "📋 Dados",
])

# ══════════════════════════════════════════════
# ABA 1 — VISÃO GERAL
# ══════════════════════════════════════════════
with t1:
    col_a, col_b = st.columns(2)

    # Dispersão com linha de tendência manual
    with col_a:
        if not df_clean.empty:
            slope, intercept, r2 = regressao(df_clean[metrica], df_clean[var_dep])
            x_range = np.linspace(df_clean[metrica].min(), df_clean[metrica].max(), 100)
            y_range = slope * x_range + intercept

            fig_disp = go.Figure()
            fig_disp.add_trace(go.Scatter(
                x=df_clean[metrica], y=df_clean[var_dep],
                mode='markers',
                marker=dict(
                    color=df_clean[var_dep], colorscale='Viridis',
                    size=9, opacity=0.85,
                    line=dict(color='#0e1117', width=1),
                    colorbar=dict(title="Pontos", thickness=12,
                                  tickfont=dict(color=PALETTE["subtext"])),
                ),
                text=df_clean.apply(lambda r: f"{r['Clube']} ({int(r['Ano'])})<br>"
                                              f"{metrica}: R$ {r[metrica]:.1f} Mi<br>"
                                              f"Pontos: {r[var_dep]:.0f}", axis=1),
                hovertemplate="%{text}<extra></extra>",
                name="Clubes",
            ))
            fig_disp.add_trace(go.Scatter(
                x=x_range, y=y_range,
                mode='lines',
                line=dict(color='#f59e0b', width=2, dash='dash'),
                name=f"Tendência (R²={r2:.2f})",
            ))
            fig_layout(fig_disp, f"{metrica} vs Pontuação Final")
            st.plotly_chart(fig_disp, use_container_width=True)

    # Evolução temporal
    with col_b:
        if 'Ano' in df_f.columns and len(anos_sel) > 1:
            ev = df_f.groupby('Ano').agg(
                Media_Fin=(metrica, 'mean'),
                Media_Pts=(var_dep, 'mean'),
            ).reset_index()
            fig_ev = make_subplots(specs=[[{"secondary_y": True}]])
            fig_ev.add_trace(go.Bar(
                x=ev['Ano'], y=ev['Media_Fin'],
                name="Média Financeira (R$ Mi)",
                marker_color='#3b82f6', opacity=0.7,
            ), secondary_y=False)
            fig_ev.add_trace(go.Scatter(
                x=ev['Ano'], y=ev['Media_Pts'],
                name="Média de Pontos",
                line=dict(color='#22c55e', width=3),
                mode='lines+markers',
                marker=dict(size=8),
            ), secondary_y=True)
            fig_ev.update_yaxes(title_text="R$ Milhões", secondary_y=False,
                                gridcolor="#1e2433", color=PALETTE["subtext"])
            fig_ev.update_yaxes(title_text="Pontos", secondary_y=True,
                                color=PALETTE["subtext"])
            fig_ev.update_layout(
                title=dict(text="Evolução Temporal: Finanças vs Pontuação",
                           font=dict(color=PALETTE["text"], size=15), x=0.01),
                paper_bgcolor=PALETTE["card"], plot_bgcolor="#12172a",
                font=dict(color=PALETTE["subtext"]),
                legend=dict(bgcolor="#12172a", bordercolor=PALETTE["border"], borderwidth=1,
                            font=dict(color=PALETTE["text"])),
                xaxis=dict(gridcolor="#1e2433", color=PALETTE["subtext"]),
                margin=dict(t=50, b=40, l=40, r=20),
            )
            st.plotly_chart(fig_ev, use_container_width=True)
        else:
            fig_hist = go.Figure(go.Histogram(
                x=df_clean[var_dep], nbinsx=15,
                marker=dict(color='#3b82f6', opacity=0.8,
                            line=dict(color='#0e1117', width=1)),
            ))
            fig_layout(fig_hist, "Distribuição da Pontuação Final")
            st.plotly_chart(fig_hist, use_container_width=True)

    # Top e Bottom clubes
    st.markdown("---")
    col_t, col_b2 = st.columns(2)
    with col_t:
        st.markdown("#### 🏆 Top 10 — Maior Investimento")
        top10 = (df_clean.sort_values(metrica, ascending=False)
                         .head(10)[['Clube', 'Ano', metrica, var_dep]]
                         .reset_index(drop=True))
        top10.index += 1
        st.dataframe(top10.style.background_gradient(cmap='Blues', subset=[metrica]),
                     use_container_width=True)
    with col_b2:
        st.markdown("#### 📉 Bottom 10 — Menor Investimento")
        bot10 = (df_clean.sort_values(metrica)
                         .head(10)[['Clube', 'Ano', metrica, var_dep]]
                         .reset_index(drop=True))
        bot10.index += 1
        st.dataframe(bot10.style.background_gradient(cmap='Reds_r', subset=[metrica]),
                     use_container_width=True)

# ══════════════════════════════════════════════
# ABA 2 — CORRELAÇÕES COMPLETAS
# ══════════════════════════════════════════════
with t2:
    st.subheader("Teste de Correlação de Spearman — Todas as Métricas")
    st.markdown(
        '<div class="insight-box">📚 <strong>Por que Spearman?</strong> '
        'A grande disparidade de receitas entre clubes como Flamengo e Palmeiras vs. clubes de menor orçamento '
        'gera <em>outliers</em> significativos, violando a premissa de normalidade do coeficiente de Pearson. '
        'O teste de Spearman avalia a associação por meio do <strong>ranqueamento</strong> das observações, '
        'sendo mais robusto para esta amostra (Bussab e Morettin, 2017).</div>',
        unsafe_allow_html=True,
    )

    if resultados_spearman:
        rows = []
        for m_, vals in resultados_spearman.items():
            r_, p_, n_ = vals['R'], vals['p'], vals['n']
            sig = "✅ Sim" if p_ < 0.05 else "❌ Não"
            if   abs(r_) >= 0.70: mag = "Muito forte"
            elif abs(r_) >= 0.50: mag = "Forte"
            elif abs(r_) >= 0.30: mag = "Moderada"
            else:                  mag = "Fraca"
            rows.append({"Métrica": m_, "R": f"{r_:.4f}", "P-Value": f"{p_:.4f}" if p_ >= 0.0001 else "< 0.0001",
                         "N": n_, "Significativa (p<0.05)": sig, "Magnitude": mag})
        df_res = pd.DataFrame(rows)
        st.dataframe(df_res, use_container_width=True, hide_index=True)

        # Gráfico de barras dos coeficientes
        r_vals  = [resultados_spearman[m_]['R'] for m_ in METRICAS if m_ in resultados_spearman]
        m_names = [m_.replace(" (R$ Mi)", "") for m_ in METRICAS if m_ in resultados_spearman]
        cores   = ['#22c55e' if r > 0 else '#ef4444' for r in r_vals]

        fig_bar = go.Figure(go.Bar(
            x=m_names, y=r_vals,
            marker=dict(color=cores, opacity=0.85,
                        line=dict(color='#0e1117', width=1)),
            text=[f"{r:.3f}" for r in r_vals],
            textposition='outside', textfont=dict(color=PALETTE["text"]),
        ))
        fig_bar.add_hline(y=0.5,  line=dict(color='#f59e0b', dash='dot', width=1))
        fig_bar.add_hline(y=-0.5, line=dict(color='#f59e0b', dash='dot', width=1))
        fig_bar.add_annotation(text="Limite: correlação forte (0.5)",
                               x=0, y=0.52, showarrow=False,
                               font=dict(color='#f59e0b', size=10))
        fig_layout(fig_bar, "Coeficiente de Spearman por Métrica Financeira")
        fig_bar.update_yaxes(range=[-1, 1.1])
        st.plotly_chart(fig_bar, use_container_width=True)

        # Interpretações individuais
        st.markdown("#### 💬 Interpretações")
        for m_ in METRICAS:
            if m_ in resultados_spearman:
                r_, p_ = resultados_spearman[m_]['R'], resultados_spearman[m_]['p']
                cor_i, texto_i = interpreta_spearman(r_, p_)
                st.markdown(
                    f'<div class="insight-box {cor_i}"><strong>{m_}</strong><br>{texto_i}</div>',
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════
# ABA 3 — K-MEANS
# ══════════════════════════════════════════════
with t3:
    st.subheader("🤖 Clustering K-Means: Agrupamento por Perfil de Eficiência")
    st.markdown(
        '<div class="insight-box">O algoritmo K-Means agrupa os clubes em <strong>6 perfis</strong> com base na '
        'combinação de <strong>investimento financeiro</strong> e <strong>pontuação esportiva</strong>. '
        'A classificação usa percentis (P33 e P66) em vez de simples médias, permitindo uma segmentação '
        'mais granular: <strong>Elite</strong> (alto investimento + alto desempenho), '
        '<strong>Investidores</strong> (alto investimento + desempenho médio), '
        '<strong>Ricos/Ineficientes</strong> (alto investimento + baixo desempenho), '
        '<strong>Eficientes</strong> (investimento médio + alto desempenho), '
        '<strong>Equilíbrados</strong> (médio/baixo investimento + desempenho médio) e '
        '<strong>Vulneráveis</strong> (baixo investimento + baixo desempenho).</div>',
        unsafe_allow_html=True,
    )

    if 'Perfil' not in df_clean.columns:
        st.warning("São necessários ao menos 6 registros para o K-Means.")
    else:
        col_ia1, col_ia2 = st.columns([2, 1])
        with col_ia1:
            paleta = cor_perfil()
            fig_k = px.scatter(
                df_clean, x=metrica, y=var_dep,
                color='Perfil',
                color_discrete_map=paleta,
                hover_name='Clube',
                hover_data={'Ano': True, metrica: ':.1f', var_dep: ':.0f'},
                size_max=14,
            )
            # Centroides
            centros2 = df_clean.groupby('Perfil')[[metrica, var_dep]].mean().reset_index()
            fig_k.add_trace(go.Scatter(
                x=centros2[metrica], y=centros2[var_dep],
                mode='markers+text',
                marker=dict(symbol='diamond', size=16, color='white',
                            line=dict(color='#0e1117', width=2)),
                text=centros2['Perfil'],
                textposition='top center',
                textfont=dict(color='white', size=10),
                name='Centroides',
                showlegend=True,
            ))
            fig_layout(fig_k, "Perfis de Eficiência Identificados pelo K-Means")
            st.plotly_chart(fig_k, use_container_width=True)

        with col_ia2:
            fig_pie = px.pie(
                df_clean, names='Perfil',
                color='Perfil', color_discrete_map=paleta,
                hole=0.45,
            )
            fig_pie.update_traces(textfont=dict(color='white'), pull=[0.03]*6)
            fig_pie.update_layout(
                title=dict(text="Distribuição de Perfis",
                           font=dict(color=PALETTE["text"], size=14), x=0.01),
                paper_bgcolor=PALETTE["card"],
                font=dict(color=PALETTE["subtext"]),
                legend=dict(bgcolor="#12172a", font=dict(color=PALETTE["text"])),
                margin=dict(t=50, b=20, l=10, r=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Estatísticas por perfil
        st.markdown("#### 📊 Estatísticas por Perfil")
        stats_perfil = (df_clean.groupby('Perfil')
                                .agg(N=('Clube', 'count'),
                                     Media_Fin=(metrica, 'mean'),
                                     Media_Pts=(var_dep, 'mean'),
                                     Clubes_Ex=('Clube', lambda x: ', '.join(x.unique()[:3])))
                                .reset_index()
                                .sort_values('Media_Pts', ascending=False))
        stats_perfil.columns = ['Perfil', 'N', f'Média {metrica}', 'Média Pontos', 'Exemplos']
        st.dataframe(stats_perfil, use_container_width=True, hide_index=True)

        # Legenda dos perfis
        st.markdown("#### 📖 Descrição dos Perfis")
        descricoes = {
            "🏆 Elite":              ("Alto investimento + alto desempenho.", "green"),
            "💰 Investidores":       ("Alto investimento + desempenho médio. Potencial não totalmente convertido em pontos.", "green"),
            "💸 Ricos/Ineficientes": ("Alto investimento + baixo desempenho. Gestão ou elenco aquém do esperado.", "yellow"),
            "✨ Eficientes":         ("Investimento médio + alto desempenho. Máximo aproveitamento dos recursos.", "green"),
            "⚖️ Equilíbrados":      ("Investimento e desempenho medianos. Perfil mais comum na Série A.", "yellow" ),
            "⚠️ Vulneráveis":       ("Baixo investimento + baixo desempenho. Alto risco de queda na tabela.", "red"),
        }
        cols_leg = st.columns(3)
        for i, (perfil, (desc, cor)) in enumerate(descricoes.items()):
            with cols_leg[i % 3]:
                st.markdown(
                    f'<div class="insight-box {cor}"><strong>{perfil}</strong><br>'
                    f'<span style="font-size:0.85rem">{desc}</span></div>',
                    unsafe_allow_html=True,
                )

        # Boxplots por perfil
        st.markdown("---")
        col_bx1, col_bx2 = st.columns(2)
        with col_bx1:
            fig_bx = px.box(df_clean, x='Perfil', y=var_dep,
                            color='Perfil', color_discrete_map=paleta,
                            points='all', hover_name='Clube')
            fig_layout(fig_bx, "Distribuição de Pontos por Perfil")
            st.plotly_chart(fig_bx, use_container_width=True)
        with col_bx2:
            fig_bx2 = px.box(df_clean, x='Perfil', y=metrica,
                             color='Perfil', color_discrete_map=paleta,
                             points='all', hover_name='Clube')
            fig_layout(fig_bx2, f"Distribuição de {metrica} por Perfil")
            st.plotly_chart(fig_bx2, use_container_width=True)

# ══════════════════════════════════════════════
# ABA 4 — IMPACTO SAF
# ══════════════════════════════════════════════
with t4:
    st.subheader("🏢 Análise Comparativa: Clubes SAF vs Associativos")
    st.markdown(
        '<div class="insight-box">A Lei nº 14.193/2021 (Lei da SAF) permitiu a transformação dos clubes em '
        '<strong>Sociedades Anônimas do Futebol</strong>, visando profissionalizar a gestão e atrair capital. '
        'Esta seção analisa se o modelo de gestão SAF impacta os indicadores financeiros e o desempenho esportivo '
        '(Monteiro, 2021).</div>',
        unsafe_allow_html=True,
    )

    if 'SAF' not in df_f.columns:
        st.warning("⚠️ Coluna 'SAF' não encontrada no CSV. Adicione uma coluna SAF (Sim/Não ou 1/0) para habilitar esta análise.")
    else:
        df_saf = df_f.dropna(subset=['SAF', var_dep])
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_box_saf = px.box(
                df_saf, x='SAF', y=var_dep, color='SAF',
                points='all', hover_name='Clube',
                color_discrete_sequence=['#3b82f6', '#f59e0b'],
            )
            fig_layout(fig_box_saf, "Pontuação Final por Modelo de Gestão")
            st.plotly_chart(fig_box_saf, use_container_width=True)

        with col_s2:
            df_saf2 = df_f.dropna(subset=['SAF', metrica])
            agg_saf = df_saf2.groupby('SAF')[metrica].mean().reset_index()
            fig_bar_saf = px.bar(
                agg_saf, x='SAF', y=metrica, color='SAF',
                color_discrete_sequence=['#3b82f6', '#f59e0b'],
                text_auto='.1f',
            )
            fig_layout(fig_bar_saf, f"Média de {metrica} por Modelo de Gestão")
            st.plotly_chart(fig_bar_saf, use_container_width=True)

        # Comparativo de médias
        st.markdown("#### 📊 Médias Comparativas")
        comp = (df_f.dropna(subset=['SAF'])
                    .groupby('SAF')
                    .agg(**{f'Média {m_}': (m_, 'mean') for m_ in METRICAS if m_ in df_f.columns},
                         **{'Média Pontos': (var_dep, 'mean'), 'N Obs.': ('Clube', 'count')})
                    .reset_index())
        st.dataframe(comp.style.format(precision=1), use_container_width=True)

        # Evolução SAF por ano
        if 'Ano' in df_f.columns and len(anos_sel) > 1:
            st.markdown("#### 📅 Evolução SAF por Ano")
            ev_saf = df_f.dropna(subset=['SAF', var_dep]).groupby(['Ano', 'SAF'])[var_dep].mean().reset_index()
            fig_ev_saf = px.line(
                ev_saf, x='Ano', y=var_dep, color='SAF',
                markers=True, color_discrete_sequence=['#3b82f6', '#f59e0b'],
            )
            fig_layout(fig_ev_saf, "Pontuação Média por Ano e Modelo de Gestão")
            st.plotly_chart(fig_ev_saf, use_container_width=True)

# ══════════════════════════════════════════════
# ABA 5 — DADOS
# ══════════════════════════════════════════════
with t5:
    st.subheader("📋 Base de Dados Processada")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        busca = st.text_input("🔍 Buscar clube", "")
    with col_f2:
        st.markdown(f"**{len(df_clean)} registros** exibidos após filtragem")

    df_show = df_clean.copy()
    if busca:
        df_show = df_show[df_show['Clube'].str.contains(busca, case=False, na=False)]

    st.dataframe(df_show, use_container_width=True)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_data = df_show.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Baixar CSV", data=csv_data,
                           file_name='dados_processados.csv', mime='text/csv')
    with col_dl2:
        st.markdown(
            f"**Período:** {min(anos_sel)} – {max(anos_sel)}"
            if anos_sel else "")

    # Estatísticas descritivas
    st.markdown("---")
    st.markdown("#### 📊 Estatísticas Descritivas")
    cols_desc = [c for c in METRICAS + [var_dep] if c in df_clean.columns]
    st.dataframe(
        df_clean[cols_desc].describe().round(2).style.format(precision=2),
        use_container_width=True,
    )