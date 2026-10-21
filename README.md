# Análise Estatística e Dashboard Interativo: Mercado de Videogames

Projeto avaliativo de **Modelagem Estatística**: análise exploratória, testes de hipóteses com intervalos de confiança, regressão linear múltipla com diagnóstico, validação e previsão, e um dashboard interativo em Streamlit.

---

## Integrantes da Equipe
* **Tayron Reis** (Dashboard Interativo, Validação Estatística, Testes de Hipóteses e Recomendações)
* **Ronald Bruno** (Engenharia de Dados, Tratamento, Engenharia de Variáveis e Regressão Linear)

---

## Visão Geral

**Pergunta de negócio:** quais fatores observáveis (recepção da crítica e do público, volume de avaliações, gênero) se associam a maiores vendas globais, e com que confiabilidade é possível prever vendas a partir deles?

A base reúne mais de 16 mil títulos, dos quais 7.007 possuem notas de crítica e público. A variável de vendas (`Global_Sales`) está em **milhões de cópias vendidas**, não em receita ou lucro.

---

## Estrutura do Repositório

```text
├── Video_Games.csv                      # Base bruta original
├── Video_Games_Clean.csv                # Base tratada e enriquecida (gerada pelo notebook)
├── analise_estatistica_videogames.ipynb # Notebook com toda a análise documentada (sem comentários no código)
├── app.py                               # Dashboard interativo em Streamlit
├── requirements.txt                     # Dependências
└── README.md
```

---

## Metodologia

### 1. Tratamento e engenharia de variáveis
* **Diagnóstico de ausentes** antes de qualquer decisão: cerca de 51% dos jogos não têm nota da crítica e 54% não têm nota do público (inclui o rótulo `tbd`, convertido em ausente).
* **Análise de casos completos** nas colunas usadas. Imputar notas foi descartado porque a ausência não é aleatória (jogos menores tendem a não ser avaliados) e a nota é a variável explicativa central.
* **Custo da decisão (viés de seleção):** a base cai de 16.928 para 7.007 linhas (58,6% descartadas) e a média de vendas sobe de 0,54 para 0,78 milhão. As conclusões valem para jogos avaliados.
* **Variáveis criadas:** `User_Score_100` (nota do público na escala 0-100), `Score_Diff` (crítica − público), `Sucesso_Comercial` (≥ 1 milhão de cópias), `Log_Sales` e logs das contagens de avaliações (as vendas têm assimetria 15,9, reduzida a 2,23 com log).

### 2. Testes de hipóteses (α = 0,05)

**Teste 1: crítica vs público (t pareado, gl = 7.006)**
* H₀: μ_crítica = μ_público. Resultado: t = −9,91; p = 5,4 × 10⁻²³; Wilcoxon concorda (p = 5,7 × 10⁻³⁶).
* Médias: 70,31 (crítica) e 71,85 (público). Diferença de −1,54 ponto, **IC 95% [−1,84; −1,23]**.
* **d de Cohen pareado = −0,12 (efeito pequeno):** estatisticamente significativo, mas pequeno na prática.

**Teste 2: vendas de Shooter vs Action (t de Welch)**
* H₀: μ_Shooter = μ_Action. Welch: t = 3,05; p = 0,0023. Welch no log das vendas: p = 0,0021.
* **IC 95%:** Action 0,73 milhão [0,66; 0,81]; Shooter 0,96 milhão [0,83; 1,08]; diferença 0,22 milhão [0,08; 0,36].
* **Robustez:** o Mann-Whitney **não** rejeita (p = 0,163) e as medianas são próximas (0,35 vs 0,29). A vantagem de Shooter está na média, puxada pelos grandes sucessos, e não no jogo típico. A escolha dos dois gêneros foi *post hoc* (limitação).

### 3. Regressão linear múltipla

| | Modelo 1 (nível) | Modelo 2 (log) |
|---|---|---|
| Resposta | `Global_Sales` | `log(1 + Global_Sales)` |
| Preditores | notas e contagens | notas e log das contagens |
| R² / R² ajustado | 0,127 / 0,127 | 0,265 / 0,265 |
| Assimetria / curtose dos resíduos | 17,9 / 622 | 1,67 / 6,4 |
| Breusch-Pagan (p) | < 0,001 | < 0,001 |
| Teste (20%): R² · MAE | 0,21* · 0,71 mi | 0,29 (log) · 0,28 (log); 0,62 mi na escala original |

\* O R² de teste do Modelo 1 supera o de treino porque os maiores outliers (ex.: Wii Sports) caíram no treino; não indica bom desempenho.

* **Modelo 1:** β_Critic_Score = +0,0198 (≈ 19.800 cópias por ponto), β_Critic_Count = +0,0183, β_User_Count = +0,0006, β_User_Score_100 = −0,0038 (p = 0,044, efeito parcial pequeno).
* **Diagnóstico:** o VIF máximo é 1,8, então **não há multicolinearidade relevante**; o Condition Number de 3.554 reflete a diferença de escala entre as variáveis. Resíduos do Modelo 1 violam normalidade e homocedasticidade, o que motivou o Modelo 2. No Modelo 2 a heterocedasticidade persiste, mas com erros-padrão robustos (HC3) todas as conclusões sobre os coeficientes se mantêm.
* **Previsão (jogo hipotético: crítica 85, público 80, 60 avaliações de críticos, 500 de usuários):** Modelo 2 prevê 1,29 milhão, intervalo de predição 95% [0,06; 3,93]. O Modelo 1 gera limite inferior negativo (−1,80), confirmando sua inadequação para previsão individual.

---

## Dashboard Interativo (Streamlit)

Sem textos que apresentem conclusões: o usuário explora e interpreta. Cada gráfico traz uma legenda com seu **objetivo** na análise.

* **Filtros globais:** gênero, plataforma, ano de lançamento e escala das vendas (original ou log).
* **Exploração:** distribuição das vendas, boxplot por gênero, médias com IC ajustável, evolução anual e proporção de jogos com ≥ 1 milhão por gênero.
* **Relações entre variáveis:** dispersão com reta de mínimos quadrados (variável do eixo x selecionável), matriz de correlação (Pearson ou Spearman) e histograma da divergência crítica × público.
* **Testes de hipóteses:** teste t pareado e comparação de dois gêneros quaisquer (Welch, Welch no log e Mann-Whitney), com α ajustável, ICs e tamanho de efeito. Os resultados são apresentados sem veredito escrito.
* **Regressão e previsão:** escolha da resposta (nível ou log) e dos preditores, coeficientes com IC, R², Breusch-Pagan, desempenho em treino e teste, resíduos, QQ-plot, previsto × observado, VIF e **simulador de previsão** com intervalo de predição.
* **Dados:** tabela filtrada e exportação em CSV.

---

## Recomendações

1. **Portfólio de gêneros.** Shooter tem maior potencial médio de vendas que Action, mas a vantagem se concentra nos grandes sucessos (medianas semelhantes e Mann-Whitney sem significância). É uma aposta de maior retorno potencial e maior risco. Adventure e Strategy têm médias baixas (≈ 0,31 e 0,25 milhão) e só 6,0% e 5,4% dos títulos chegam a 1 milhão de cópias.
2. **Qualidade e cobertura crítica.** Nota da crítica e número de avaliações da crítica estão positivamente associados às vendas, controlando as demais variáveis. É uma **associação**, não causalidade: jogos grandes tendem a receber mais cobertura.
3. **Triagem de projetos.** O Modelo 2 serve para comparar candidatos pelas vendas esperadas, sempre com o intervalo de predição (largo), e não para fixar metas.

## Limitações

* Viés de seleção (58,6% dos jogos descartados por falta de nota).
* Vendas em unidades, sem receita, orçamento, marketing, franquia ou microtransações.
* Poder explicativo limitado (R² de 26,5% no melhor modelo), heterocedasticidade persistente e cauda pesada nos resíduos.
* Resultados observacionais; `Critic_Count` pode refletir o porte do projeto.
* Comparação Shooter vs Action definida *post hoc*, sem correção para múltiplas comparações.
* Uma única divisão treino/teste; a base termina em 2016.

---

## Como Executar Localmente

Requer **Python 3.9+**.

```bash
git clone https://github.com/SEU-USUARIO/NOME-DO-REPOSITORIO.git
cd NOME-DO-REPOSITORIO
python -m pip install -r requirements.txt
```

**Notebook** (gera `Video_Games_Clean.csv`):
```bash
jupyter notebook analise_estatistica_videogames.ipynb
```

**Dashboard** (precisa de `Video_Games_Clean.csv` na mesma pasta):
```bash
python -m streamlit run app.py
```
O painel abre em `http://localhost:8501`.
