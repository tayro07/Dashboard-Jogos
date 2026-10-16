import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard - Mercado de Videojogos", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('Video_Games_Clean.csv')

df = load_data()

st.sidebar.header("Filtros Exploratórios")
filtro_genero = st.sidebar.multiselect(
    "Filtrar por Género", 
    df['Genre'].unique(), 
    default=['Action', 'Shooter', 'Sports', 'Role-Playing']
)
filtro_sucesso = st.sidebar.multiselect(
    "Nível de Sucesso Comercial", 
    df['Sucesso_Comercial'].unique(), 
    default=df['Sucesso_Comercial'].unique()
)

df_filtrado = df[(df['Genre'].isin(filtro_genero)) & (df['Sucesso_Comercial'].isin(filtro_sucesso))]

st.title("Análise de Vendas e Avaliações de Videojogos")

c1, c2, c3 = st.columns(3)
c1.metric("Vendas Globais (Média)", round(df_filtrado['Global_Sales'].mean(), 2))
c2.metric("Nota da Crítica (Média)", round(df_filtrado['Critic_Score'].mean(), 2))
c3.metric("Nota do Público (Média)", round(df_filtrado['User_Score_100'].mean(), 2))

st.markdown("---")

col_esq, col_dir = st.columns(2)

with col_esq:
    fig_scatter = px.scatter(
        df_filtrado, 
        x='Critic_Score', 
        y='Global_Sales', 
        color='Genre',
        trendline='ols',
        title="Relação Linear: Nota da Crítica vs Vendas Globais (Milhões)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_dir:
    fig_box = px.box(
        df_filtrado,
        x='Genre',
        y='Global_Sales',
        color='Sucesso_Comercial',
        title="Distribuição e Dispersão de Vendas por Género"
    )
    st.plotly_chart(fig_box, use_container_width=True)

fig_hist = px.histogram(
    df_filtrado,
    x='Score_Diff',
    nbins=50,
    title="Distribuição da Divergência de Notas (Crítica vs Público)"
)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("### Tabela de Dados Detalhados")
st.dataframe(df_filtrado)