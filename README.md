# Análise Estatística e Dashboard Interativo: Mercado de Videojogos

Projeto avaliativo de **Modelagem Estatística** voltado à análise exploratória, inferência estatística, modelagem preditiva via Regressão Linear Múltipla e desenvolvimento de um dashboard interativo para suporte à tomada de decisões estratégicas na indústria de jogos eletrônicos.

---

## 👥 Integrantes da Equipe
* **Tayron Reis** (Dashboard Interativo, Validação Estatística, Testes de Hipóteses e Recomendações)
* **Ronald Bruno** (Engenharia de Dados, Tratamento, Engenharia de Variáveis e Regressão Linear)

---

## 📌 Visão Geral do Projeto

O objetivo central deste trabalho é simular a atuação de uma equipe de ciência e análise de dados em um cenário real de mercado. A partir de uma base com mais de 7.000 títulos avaliados criticamente e comercializados globalmente, o projeto investiga:
1. A relação de consistência entre a recepção de especialistas (crítica) e o público consumidor.
2. O impacto financeiro direto e a viabilidade comercial entre diferentes géneros de jogos.
3. O poder preditivo de métricas de engajamento e pontuações sobre as vendas mundiais.

Os resultados foram sintetizados em um **dashboard interativo** construído em Streamlit, sem textos indutivos de conclusão, permitindo exploração analítica livre e fundamentada em dados.

---

## 🗂️ Estrutura do Repositório

```text
├── Video_Games.csv                      # Base de dados bruta original
├── Video_Games_Clean.csv                # Base tratada e enriquecida após engenharia de atributos
├── analise_estatistica_videogames.ipynb # Notebook com toda a modelagem estatística documentada
├── app.py                               # Aplicação do Dashboard Interativo em Streamlit
└── README.md                            # Documentação geral do repositório
```

---

## 🔬 Metodologia Estatística

### 1. Tratamento e Engenharia de Variáveis
* **Tratamento de Inconsistências**: Conversão da variável `User_Score` para ponto flutuante, tratando rótulos não numéricos (como `"tbd"`), e descarte direcionado de registros incompletos nas dimensões críticas de análise.
* **Harmonização de Escala**: Criação da variável `User_Score_100` ($User\_Score \times 10$) para equalizar a base comparativa com o `Critic_Score` (0 a 100).
* **Métrica de Divergência**: Criação de `Score_Diff` ($Critic\_Score - User\_Score\_100$) para quantificar descolamentos de percepção.
* **Categorização de Desempenho**: Criação da variável qualitativa `Sucesso_Comercial` (limiar de $\ge 1{,}0$ milhão de cópias vendidas globalmente).

---

### 2. Validação Inferencial e Testes de Hipóteses

* **Teste 1: Consistência Avaliativa (Crítica vs. Público)**
  * **Técnica**: Teste $t$ de Student Pareado ($df = 7.006$).
  * **Hipóteses**: $H_0: \mu_{\text{crítica}} = \mu_{\text{público}}$ vs. $H_1: \mu_{\text{crítica}} \neq \mu_{\text{público}}$.
  * **Resultado**: $t = -9{,}91$ | $p\text{-valor} = 5{,}38 \times 10^{-23}$.
  * **Conclusão**: Rejeição de $H_0$. Há diferença estatisticamente significativa: a imprensa especializada tende a pontuar de forma ligeiramente mais severa (média: $70{,}31$) do que os usuários (média: $71{,}85$).

- **Teste 2: Retorno Financeiro por Gênero (Shooter vs. Action)**
  - **Técnica:** Teste *t* de Student para Amostras Independentes com correção de Welch (`equal_var=False`).
  - **Hipóteses:** $H_0: \mu_{\text{Shooter}} = \mu_{\text{Action}}$ vs. $H_1: \mu_{\text{Shooter}} \neq \mu_{\text{Action}}$.
  - **Resultado:** $t = 3{,}05$ | $p$-valor = $0{,}0023$.
  - **Intervalos de Confiança (95%):**
    - **Action:** Média de 0,73M cópias | **IC 95% = [0,66; 0,81]** milhões de cópias.
    - **Shooter:** Média de 0,96M cópias | **IC 95% = [0,83; 1,08]** milhões de cópias.
  - **Conclusão:** Rejeição de $H_0$. Títulos do gênero *Shooter* demonstram rentabilidade média unitária superior e estatisticamente comprovada.

---

### 3. Modelagem Preditiva (Regressão Linear Múltipla - OLS)

Modelo formulado para prever o volume de vendas mundiais (`Global_Sales`):

$$\text{Global Sales} = \beta_0 + \beta_1(\text{Critic Score}) + \beta_2(\text{User Score 100}) + \beta_3(\text{Critic Count}) + \beta_4(\text{User Count})$$

- **$R^2$ / $R^2$ Ajustado:** 0,127 (12,7% da variância das vendas globais explicada pelo modelo).
- **Significância Geral:** *F-statistic* = 255,1 ($p$-valor = $6{,}16 \times 10^{-205}$).
- **Coeficientes Principais:**
  - **$\beta_{\text{Critic Score}} = +0{,}0198$** ($p < 0{,}001$): cada ponto extra na avaliação da imprensa agrega cerca de 20.000 cópias vendidas adicionais.
  - **$\beta_{\text{Critic Count}} = +0{,}0183$** ($p < 0{,}001$): ampla cobertura da imprensa profissional tem impacto expressivo e positivo na faturação.
  - **$\beta_{\text{User Score 100}} = -0{,}0038$** ($p = 0{,}044$): nota do público com ligeiro coeficiente negativo no modelo multivariado.
  - **$\beta_{\text{User Count}} = +0{,}0006$** ($p < 0{,}001$): volume de engajamento do público com impacto positivo marginal.

---

## 📊 Dashboard Interativo (Streamlit)

O painel foi construído sob preceitos rígidos de visualização de dados:
* **Exploração Livre**: Sem caixas de texto com interpretações tendenciosas; a tomada de decisão é orientada pela exploração visual autônoma do usuário.
* **Filtros Dinâmicos**: Segmentação simultânea por géneros de jogos e categorias de sucesso comercial.
* **Componentes Visuais Dedicados**:
  1. *Cartões de Métricas*: Médias dinâmicas de vendas e notas reavaliadas em tempo real.
  2. *Gráfico de Dispersão com Linha OLS*: Demonstração visual do ajuste linear entre notas e volume comercial.
  3. *Boxplot Comparativo*: Distribuição de quartis e detecção de outliers por segmento.
  4. *Histograma de Divergência*: Dispersão empírica da diferença entre notas especializadas e do público.
  5. *Tabela Dinâmica*: Auditoria e ordenação de registros individuais.

---

## 🎯 Recomendações Práticas e Tomada de Decisão

1. **Priorização de Portfólio (Shooter > Action)**: Estúdios que buscam maior previsibilidade de receita devem alocar orçamento no desenvolvimento de shooters, já que o limite inferior do IC de 95% do gênero ($0{,}83$M) supera o retorno médio de títulos de ação ($0{,}73$M).
2. **Estratégia de Relações Públicas (PR)**: O volume de veículos avaliadores (`Critic_Count`) e a nota crítica demonstraram coeficientes altamente significativos ($p < 0{,}001$). Ações voltadas ao envio de cópias antecipadas e realização de eventos de lançamento com a imprensa especializada são fundamentais para assegurar tração inicial nas vendas.

---

## ⚠️ Limitações do Estudo

* **Baixo $R^2$ ($12{,}7\%$)**: Fatores mercadológicos externos cruciais não estão contemplados na base (ex.: orçamento de marketing publicitário, presença de microtransações e efeito de franquias consagradas).
* **Diagnóstico de Multicolinearidade**: O sumário OLS indicou um *Condition Number* elevado ($3.550$), sugerindo interdependência entre contagens e notas que requer cautela ao isolar efeitos parciais de parâmetros individuais.

---

## 🚀 Como Executar o Projeto Localmente

### Pré-requisitos
Certifique-se de possuir o **Python 3.9+** instalado.

### 1. Clonar o Repositório
```bash
git clone https://github.com/SEU-USUARIO/NOME-DO-REPOSITORIO.git
cd NOME-DO-REPOSITORIO
```

### 2. Instalar as Dependências
```bash
python -m pip install pandas numpy scipy statsmodels streamlit plotly
```

### 3. Rodar o Dashboard Interativo
```bash
python -m streamlit run app.py
```
O painel abrirá automaticamente no seu navegador padrão no endereço `http://localhost:8501`.
