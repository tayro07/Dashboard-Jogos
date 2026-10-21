import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api as sm
import streamlit as st
from sklearn.model_selection import train_test_split
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(page_title="Dashboard - Mercado de Videogames", layout="wide")

PREDITORES = {
    "Critic_Score": "Nota da crítica",
    "User_Score_100": "Nota do público (0-100)",
    "Critic_Count": "Nº de avaliações da crítica",
    "User_Count": "Nº de avaliações do público",
    "Log_Critic_Count": "log(1 + nº de avaliações da crítica)",
    "Log_User_Count": "log(1 + nº de avaliações do público)",
}
BASE_DO_LOG = {"Log_Critic_Count": "Critic_Count", "Log_User_Count": "User_Count"}
VARIAVEIS_DISPERSAO = ["Critic_Score", "User_Score_100", "Critic_Count", "User_Count"]


@st.cache_data
def load_data():
    df = pd.read_csv("Video_Games_Clean.csv")
    df["Log_Sales"] = np.log1p(df["Global_Sales"])
    df["Log_Critic_Count"] = np.log1p(df["Critic_Count"])
    df["Log_User_Count"] = np.log1p(df["User_Count"])
    return df


def ic_media(x, nivel):
    x = np.asarray(x, dtype=float)
    media = x.mean()
    margem = stats.sem(x) * stats.t.ppf((1 + nivel) / 2, len(x) - 1)
    return media, media - margem, media + margem


def ic_diferenca_welch(a, b, nivel):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    gl = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    dif = a.mean() - b.mean()
    margem = stats.t.ppf((1 + nivel) / 2, gl) * np.sqrt(va + vb)
    return dif, dif - margem, dif + margem, gl


def fmt(valor, casas=4):
    return f"{valor:.{casas}f}"


def fmt_p(valor):
    return f"{valor:.3e}" if valor < 0.001 else f"{valor:.4f}"


def metricas(real, previsto):
    erro = np.asarray(real) - np.asarray(previsto)
    real = np.asarray(real)
    return {
        "RMSE": float(np.sqrt((erro**2).mean())),
        "MAE": float(np.abs(erro).mean()),
        "R²": float(1 - (erro**2).sum() / ((real - real.mean()) ** 2).sum()),
    }


df = load_data()

st.sidebar.header("Filtros")
generos = sorted(df["Genre"].unique())
filtro_genero = st.sidebar.multiselect("Gênero", generos, default=generos)
plataformas = sorted(df["Platform"].unique())
filtro_plataforma = st.sidebar.multiselect("Plataforma", plataformas, default=plataformas)
ano_min, ano_max = int(df["Year_of_Release"].min()), int(df["Year_of_Release"].max())
filtro_ano = st.sidebar.slider("Ano de lançamento", ano_min, ano_max, (ano_min, ano_max))
escala_log = st.sidebar.radio(
    "Escala das vendas nos gráficos",
    ["Original (milhões)", "Logarítmica: log(1 + vendas)"],
)
usar_log = escala_log.startswith("Log")
ycol = "Log_Sales" if usar_log else "Global_Sales"
ylabel = "log(1 + vendas globais)" if usar_log else "Vendas globais (milhões)"

df_f = df[
    df["Genre"].isin(filtro_genero)
    & df["Platform"].isin(filtro_plataforma)
    & df["Year_of_Release"].between(filtro_ano[0], filtro_ano[1])
].copy()

st.title("Análise de Vendas e Avaliações de Videogames")

if len(df_f) < 30:
    st.warning("Poucos registros com os filtros atuais (mínimo de 30). Amplie a seleção.")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Jogos", f"{len(df_f):,}".replace(",", "."))
c2.metric("Vendas globais: média (mi)", round(df_f["Global_Sales"].mean(), 3))
c3.metric("Vendas globais: mediana (mi)", round(df_f["Global_Sales"].median(), 3))
c4.metric("Nota da crítica: média", round(df_f["Critic_Score"].mean(), 2))
c5.metric("Nota do público: média", round(df_f["User_Score_100"].mean(), 2))

aba_expl, aba_rel, aba_testes, aba_reg, aba_dados = st.tabs(
    ["Exploração", "Relações entre variáveis", "Testes de hipóteses", "Regressão e previsão", "Dados"]
)

with aba_expl:
    esq, dir_ = st.columns(2)
    with esq:
        st.caption("Objetivo: observar a forma da distribuição das vendas (assimetria, caudas).")
        fig = px.histogram(df_f, x=ycol, nbins=60, labels={ycol: ylabel}, title="Distribuição das vendas globais")
        st.plotly_chart(fig)
    with dir_:
        st.caption("Objetivo: comparar mediana, dispersão e valores extremos das vendas entre gêneros.")
        ordem = df_f.groupby("Genre")[ycol].median().sort_values(ascending=False).index
        fig = px.box(
            df_f, x="Genre", y=ycol, category_orders={"Genre": list(ordem)},
            labels={ycol: ylabel, "Genre": "Gênero"}, title="Vendas globais por gênero",
        )
        st.plotly_chart(fig)

    esq, dir_ = st.columns(2)
    with esq:
        st.caption("Objetivo: comparar médias de vendas entre gêneros com a precisão da estimativa (IC 95%).")
        nivel_ic = st.slider("Nível de confiança do IC", 0.80, 0.99, 0.95, 0.01, key="nivel_expl")
        linhas = []
        for genero, grupo in df_f.groupby("Genre"):
            if len(grupo) >= 2:
                m, lo, hi = ic_media(grupo["Global_Sales"], nivel_ic)
                linhas.append({"Genre": genero, "media": m, "erro_sup": hi - m, "erro_inf": m - lo, "n": len(grupo)})
        resumo = pd.DataFrame(linhas).sort_values("media", ascending=False)
        fig = px.scatter(
            resumo, x="Genre", y="media", error_y="erro_sup", error_y_minus="erro_inf",
            hover_data=["n"], labels={"media": "Vendas médias (milhões)", "Genre": "Gênero"},
            title="Vendas médias por gênero com IC",
        )
        st.plotly_chart(fig)
    with dir_:
        st.caption("Objetivo: acompanhar a evolução temporal das vendas e do volume de lançamentos.")
        por_ano = df_f.groupby("Year_of_Release").agg(
            vendas_medias=("Global_Sales", "mean"), jogos=("Global_Sales", "size")
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=por_ano["Year_of_Release"], y=por_ano["jogos"], name="Nº de jogos", opacity=0.35, yaxis="y2"))
        fig.add_trace(go.Scatter(x=por_ano["Year_of_Release"], y=por_ano["vendas_medias"], name="Vendas médias (mi)", mode="lines+markers"))
        fig.update_layout(
            title="Vendas médias e nº de jogos por ano",
            xaxis_title="Ano de lançamento",
            yaxis=dict(title="Vendas médias (milhões)"),
            yaxis2=dict(title="Nº de jogos", overlaying="y", side="right"),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig)

    st.caption("Objetivo: comparar, por gênero, a participação de jogos que atingem o limiar de 1 milhão de cópias.")
    taxa = (
        df_f.assign(alto=df_f["Sucesso_Comercial"].eq("Alto Sucesso"))
        .groupby("Genre")["alto"].mean().sort_values(ascending=False).reset_index()
    )
    fig = px.bar(taxa, x="Genre", y="alto", labels={"alto": "Proporção com ≥ 1 milhão de cópias", "Genre": "Gênero"},
                 title="Proporção de jogos com vendas ≥ 1 milhão por gênero")
    st.plotly_chart(fig)

with aba_rel:
    esq, dir_ = st.columns(2)
    with esq:
        st.caption("Objetivo: investigar a relação entre uma variável explicativa e as vendas, com reta de mínimos quadrados.")
        xcol = st.selectbox("Variável no eixo x", VARIAVEIS_DISPERSAO, format_func=lambda c: PREDITORES[c])
        separar = st.checkbox("Colorir por gênero (reta única para todos os pontos)")
        fig = px.scatter(
            df_f, x=xcol, y=ycol, color="Genre" if separar else None,
            trendline="ols", trendline_scope="overall", opacity=0.5,
            labels={xcol: PREDITORES[xcol], ycol: ylabel, "Genre": "Gênero"},
            hover_data=["Name", "Platform", "Year_of_Release"],
            title=f"{ylabel} vs {PREDITORES[xcol]}",
        )
        st.plotly_chart(fig)
        resultado = px.get_trendline_results(fig).px_fit_results.iloc[0]
        r_p, p_p = stats.pearsonr(df_f[xcol], df_f[ycol])
        r_s, p_s = stats.spearmanr(df_f[xcol], df_f[ycol])
        st.table(pd.DataFrame({
            "Estatística": ["Inclinação da reta", "R² da reta", "Correlação de Pearson", "p-valor (Pearson)", "Correlação de Spearman", "p-valor (Spearman)"],
            "Valor": [fmt(resultado.params[1]), fmt(resultado.rsquared), fmt(r_p), fmt_p(p_p), fmt(r_s), fmt_p(p_s)],
        }).set_index("Estatística").astype(str))
    with dir_:
        st.caption("Objetivo: medir a associação entre todas as variáveis numéricas e identificar redundâncias entre preditores.")
        metodo = st.radio("Método de correlação", ["pearson", "spearman"], horizontal=True)
        cols_corr = ["Global_Sales", "Critic_Score", "User_Score_100", "Critic_Count", "User_Count"]
        corr = df_f[cols_corr].corr(method=metodo)
        fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title=f"Matriz de correlação ({metodo})")
        st.plotly_chart(fig)

    st.caption("Objetivo: observar a distribuição da diferença entre a nota da crítica e a do público (base do teste pareado).")
    fig = px.histogram(df_f, x="Score_Diff", nbins=50, marginal="box",
                       labels={"Score_Diff": "Nota da crítica - nota do público"},
                       title="Divergência entre crítica e público")
    st.plotly_chart(fig)

with aba_testes:
    alpha = st.select_slider("Nível de significância α", options=[0.01, 0.05, 0.10], value=0.05)
    nivel = 1 - alpha

    st.subheader("Teste t pareado: nota da crítica vs nota do público")
    st.markdown("H₀: μ_crítica = μ_público | H₁: μ_crítica ≠ μ_público")
    dif = df_f["Score_Diff"]
    n = len(dif)
    t_p, p_t = stats.ttest_rel(df_f["Critic_Score"], df_f["User_Score_100"])
    media_d, lo_d, hi_d = ic_media(dif, nivel)
    d_z = media_d / dif.std(ddof=1)
    try:
        p_wil = stats.wilcoxon(dif)[1]
    except ValueError:
        p_wil = float("nan")
    st.table(pd.DataFrame({
        "Estatística": ["n (pares)", "Graus de liberdade", "Média crítica", "Média público", "Diferença média",
                        f"IC {nivel:.0%} inferior", f"IC {nivel:.0%} superior", "Estatística t", "p-valor (t pareado)",
                        "p-valor (Wilcoxon)", "α", "d de Cohen pareado"],
        "Valor": [n, n - 1, fmt(df_f["Critic_Score"].mean(), 2), fmt(df_f["User_Score_100"].mean(), 2), fmt(media_d),
                  fmt(lo_d), fmt(hi_d), fmt(t_p, 3), fmt_p(p_t), fmt_p(p_wil), alpha, fmt(d_z, 3)],
    }).set_index("Estatística").astype(str))

    st.divider()
    st.subheader("Comparação de vendas entre dois gêneros")
    st.markdown("H₀: μ_A = μ_B | H₁: μ_A ≠ μ_B")
    opcoes = sorted(df_f["Genre"].unique())
    cA, cB = st.columns(2)
    padrao_a = opcoes.index("Shooter") if "Shooter" in opcoes else 0
    padrao_b = opcoes.index("Action") if "Action" in opcoes else min(1, len(opcoes) - 1)
    gen_a = cA.selectbox("Gênero A", opcoes, index=padrao_a)
    gen_b = cB.selectbox("Gênero B", opcoes, index=padrao_b)

    va = df_f.loc[df_f["Genre"] == gen_a, "Global_Sales"]
    vb = df_f.loc[df_f["Genre"] == gen_b, "Global_Sales"]
    if gen_a == gen_b or len(va) < 3 or len(vb) < 3:
        st.warning("Escolha dois gêneros diferentes com pelo menos 3 jogos cada nos filtros atuais.")
    else:
        t_w, p_w = stats.ttest_ind(va, vb, equal_var=False)
        t_wl, p_wl = stats.ttest_ind(np.log1p(va), np.log1p(vb), equal_var=False)
        u, p_u = stats.mannwhitneyu(va, vb, alternative="two-sided")
        p_lev = stats.levene(va, vb)[1]
        dif_m, lo_m, hi_m, gl = ic_diferenca_welch(va, vb, nivel)

        st.table(pd.DataFrame({
            "Teste": ["Welch (vendas)", "Welch (log das vendas)", "Mann-Whitney (vendas)"],
            "Estatística": [fmt(t_w, 3), fmt(t_wl, 3), f"{u:,.0f}"],
            "p-valor": [fmt_p(p_w), fmt_p(p_wl), fmt_p(p_u)],
            "α": [alpha] * 3,
        }).set_index("Teste"))

        linhas = []
        for nome, serie in [(gen_a, va), (gen_b, vb)]:
            m, lo, hi = ic_media(serie, nivel)
            linhas.append({"Gênero": nome, "n": len(serie), "Média": m, f"IC {nivel:.0%} inferior": lo,
                           f"IC {nivel:.0%} superior": hi, "Mediana": serie.median(), "Desvio padrão": serie.std()})
        tabela_ic = pd.DataFrame(linhas).set_index("Gênero")
        st.table(tabela_ic.style.format({c: "{:.4f}" for c in tabela_ic.columns if c != "n"}))
        st.table(pd.DataFrame({
            "Estatística": ["Diferença de médias (A - B)", f"IC {nivel:.0%} inferior", f"IC {nivel:.0%} superior",
                            "Graus de liberdade (Welch)", "Levene (p-valor)"],
            "Valor": [fmt(dif_m), fmt(lo_m), fmt(hi_m), fmt(gl, 1), fmt_p(p_lev)],
        }).set_index("Estatística").astype(str))

        esq, dir_ = st.columns(2)
        with esq:
            st.caption("Objetivo: visualizar a precisão das médias estimadas e a sobreposição dos intervalos de confiança.")
            fig = go.Figure()
            for linha in linhas:
                fig.add_trace(go.Scatter(
                    x=[linha["Gênero"]], y=[linha["Média"]], mode="markers", name=linha["Gênero"],
                    error_y=dict(type="data", symmetric=False,
                                 array=[linha[f"IC {nivel:.0%} superior"] - linha["Média"]],
                                 arrayminus=[linha["Média"] - linha[f"IC {nivel:.0%} inferior"]]),
                ))
            fig.update_layout(title=f"Médias com IC {nivel:.0%}", yaxis_title="Vendas médias (milhões)", showlegend=False)
            st.plotly_chart(fig)
        with dir_:
            st.caption("Objetivo: comparar as distribuições completas dos dois grupos, não apenas as médias.")
            comp = df_f[df_f["Genre"].isin([gen_a, gen_b])]
            fig = px.violin(comp, x="Genre", y=ycol, box=True, points="outliers",
                            labels={ycol: ylabel, "Genre": "Gênero"}, title="Distribuição das vendas nos dois gêneros")
            st.plotly_chart(fig)

with aba_reg:
    st.subheader("Configuração do modelo")
    cfg1, cfg2, cfg3 = st.columns([1, 2, 1])
    resposta = cfg1.radio("Variável resposta", ["Global_Sales", "Log_Sales"],
                          format_func=lambda c: "Vendas globais (milhões)" if c == "Global_Sales" else "log(1 + vendas globais)")
    preds = cfg2.multiselect("Variáveis explicativas", list(PREDITORES), default=["Critic_Score", "User_Score_100", "Critic_Count", "User_Count"],
                             format_func=lambda c: PREDITORES[c])
    tamanho_teste = cfg3.slider("Proporção de teste", 0.10, 0.40, 0.20, 0.05)

    if len(preds) == 0 or len(df_f) < 50:
        st.warning("Selecione ao menos uma variável explicativa e mantenha pelo menos 50 jogos nos filtros.")
    else:
        X = sm.add_constant(df_f[preds])
        y = df_f[resposta]
        modelo = sm.OLS(y, X).fit()

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=tamanho_teste, random_state=42)
        modelo_tr = sm.OLS(ytr, Xtr).fit()
        prev_tr, prev_te = modelo_tr.predict(Xtr), modelo_tr.predict(Xte)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("R²", round(modelo.rsquared, 4))
        m2.metric("R² ajustado", round(modelo.rsquared_adj, 4))
        m3.metric("p-valor do teste F", fmt_p(modelo.f_pvalue))
        m4.metric("Breusch-Pagan (p)", fmt_p(het_breuschpagan(modelo.resid, modelo.model.exog)[1]))
        m5.metric("Condition Number", round(modelo.condition_number, 1))

        st.caption("Coeficientes estimados na amostra filtrada completa.")
        ic = modelo.conf_int()
        coefs = pd.DataFrame({
            "Coeficiente": modelo.params, "Erro padrão": modelo.bse, "t": modelo.tvalues,
            "p-valor": modelo.pvalues.map(fmt_p), "IC 95% inferior": ic[0], "IC 95% superior": ic[1],
        })
        st.table(coefs.style.format({c: "{:.5f}" for c in ["Coeficiente", "Erro padrão", "t", "IC 95% inferior", "IC 95% superior"]}))

        st.caption("Desempenho em treino e teste (ajuste separado, com a proporção de teste escolhida).")
        desempenho = {"Treino": metricas(ytr, prev_tr), "Teste": metricas(yte, prev_te)}
        if resposta == "Log_Sales":
            real_te = np.expm1(yte)
            prev_orig = np.expm1(prev_te).clip(lower=0)
            desempenho["Teste (milhões, escala original)"] = metricas(real_te, prev_orig)
        st.table(pd.DataFrame(desempenho).T.style.format("{:.4f}"))

        esq, dir_ = st.columns(2)
        with esq:
            st.caption("Objetivo: verificar linearidade e variância constante dos resíduos.")
            fig = px.scatter(x=modelo.fittedvalues, y=modelo.resid, opacity=0.4,
                             labels={"x": "Valores ajustados", "y": "Resíduos"}, title="Resíduos vs ajustados")
            fig.add_hline(y=0, line_color="red")
            st.plotly_chart(fig)
        with dir_:
            st.caption("Objetivo: verificar a normalidade dos resíduos.")
            (osm, osr), (slope, intercept, _) = stats.probplot(modelo.resid, dist="norm")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Resíduos", opacity=0.5))
            fig.add_trace(go.Scatter(x=[osm.min(), osm.max()], y=[slope * osm.min() + intercept, slope * osm.max() + intercept],
                                     mode="lines", name="Referência normal", line=dict(color="red")))
            fig.update_layout(title="QQ-plot dos resíduos", xaxis_title="Quantis teóricos", yaxis_title="Quantis amostrais")
            st.plotly_chart(fig)

        esq, dir_ = st.columns(2)
        with esq:
            st.caption("Objetivo: comparar valores previstos e observados no conjunto de teste.")
            fig = px.scatter(x=prev_te, y=yte, opacity=0.4, labels={"x": "Previsto", "y": "Observado"},
                             title="Previsto vs observado (teste)")
            limite = [float(min(prev_te.min(), yte.min())), float(max(prev_te.max(), yte.max()))]
            fig.add_trace(go.Scatter(x=limite, y=limite, mode="lines", line=dict(color="red"), name="y = x"))
            st.plotly_chart(fig)
        with dir_:
            st.caption("Objetivo: diagnosticar multicolinearidade entre as variáveis explicativas.")
            if len(preds) >= 2:
                vif = pd.Series(
                    {c: variance_inflation_factor(X.values, i) for i, c in enumerate(X.columns) if c != "const"}, name="VIF"
                )
                fig = px.bar(vif.reset_index(), x="index", y="VIF", labels={"index": "Variável"}, title="Fator de inflação da variância (VIF)")
                st.plotly_chart(fig)
            else:
                st.info("O VIF requer pelo menos duas variáveis explicativas.")

        st.divider()
        st.subheader("Simulador de previsão")
        st.caption("Ajuste os valores de um jogo hipotético; a previsão usa o modelo configurado acima.")
        entradas = {}
        colunas_sliders = st.columns(min(len(preds), 4))
        for i, p in enumerate(preds):
            base = BASE_DO_LOG.get(p, p)
            serie = df_f[base]
            lo, hi, med = float(serie.min()), float(serie.max()), float(serie.median())
            passo = 1.0 if base.endswith("Count") else 0.5
            with colunas_sliders[i % len(colunas_sliders)]:
                entradas[base] = st.slider(PREDITORES[base] if base in PREDITORES else base, lo, hi, med, passo, key=f"sim_{base}")

        linha = {"const": 1.0}
        for p in preds:
            linha[p] = np.log1p(entradas[BASE_DO_LOG[p]]) if p in BASE_DO_LOG else entradas[p]
        x_novo = pd.DataFrame([linha])[X.columns]
        pred = modelo.get_prediction(x_novo).summary_frame(alpha=0.05).iloc[0]

        if resposta == "Log_Sales":
            resumo = {
                "Previsão": np.expm1(pred["mean"]),
                "IC 95% da média (inferior)": max(0.0, np.expm1(pred["mean_ci_lower"])),
                "IC 95% da média (superior)": np.expm1(pred["mean_ci_upper"]),
                "Intervalo de predição 95% (inferior)": max(0.0, np.expm1(pred["obs_ci_lower"])),
                "Intervalo de predição 95% (superior)": np.expm1(pred["obs_ci_upper"]),
            }
        else:
            resumo = {
                "Previsão": pred["mean"],
                "IC 95% da média (inferior)": pred["mean_ci_lower"],
                "IC 95% da média (superior)": pred["mean_ci_upper"],
                "Intervalo de predição 95% (inferior)": pred["obs_ci_lower"],
                "Intervalo de predição 95% (superior)": pred["obs_ci_upper"],
            }
        s1, s2, s3 = st.columns(3)
        s1.metric("Vendas previstas (milhões)", round(resumo["Previsão"], 3))
        s2.metric("Intervalo de predição 95%", f'{resumo["Intervalo de predição 95% (inferior)"]:.3f} a {resumo["Intervalo de predição 95% (superior)"]:.3f}')
        s3.metric("IC 95% da média", f'{resumo["IC 95% da média (inferior)"]:.3f} a {resumo["IC 95% da média (superior)"]:.3f}')

with aba_dados:
    st.caption("Registros após os filtros. A tabela pode ser ordenada pelas colunas e exportada.")
    colunas_tabela = [c for c in df_f.columns if c not in ["Log_Sales", "Log_Critic_Count", "Log_User_Count"]]
    st.dataframe(df_f[colunas_tabela])
    st.download_button("Baixar dados filtrados (CSV)", df_f[colunas_tabela].to_csv(index=False).encode("utf-8"),
                       file_name="videogames_filtrado.csv", mime="text/csv")
