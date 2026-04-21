# CLONALG - Classificação de Flores Iris usando Algoritmo Imunológico

Este projeto implementa o **CLONALG (Algoritmo de Seleção Clonal)** para resolver o problema de classificação de flores Iris. CLONALG é um algoritmo imunológico inspirado na capacidade do sistema imunológico biológico de reconhecer e responder a antígenos através de seleção clonal, mutação e maturação de afinidade.

## Estrutura do Projeto

```
ImmuneAlgorithmFlowers/
├── clonalg.py                      # Implementação central do CLONALG
├── iris_classification.ipynb       # Notebook Jupyter principal com análise completa
├── population_analysis.py          # Script autossuficiente para análise de tamanho de população
├── README.md                       # Este arquivo
└── *.png                          # Gráficos gerados
```

## Descrição dos Arquivos

### 1. `clonalg.py`
Implementação central do algoritmo CLONALG com duas classes principais:

- **`Antibody`**: Representa um anticorpo (candidato de solução)
  - `weights`: Pesos de características representando o protótipo da classe
  - `affinity`: Valor de aptidão (acurácia de classificação)
  - `class_id`: Qual classe este anticorpo reconhece

- **`CLONALG`**: Implementação principal do algoritmo
  - `_initialize_population()`: Criar anticorpos aleatórios iniciais
  - `_evaluate_population()`: Calcular aptidão (afinidade)
  - `_selection()`: Selecionar os melhores M anticorpos
  - `_cloning()`: Clonar anticorpos proporcionalmente à afinidade
  - `_hypermutation()`: Mutar clones inversamente à afinidade
  - `_generate_new_antibodies()`: Adicionar anticorpos aleatórios para diversidade
  - `fit()`: Treinar o modelo
  - `predict()`: Fazer previsões em novos dados
  - `score()`: Calcular acurácia

### 2. `iris_classification.ipynb`
Notebook Jupyter completo com:

1. **Importar Bibliotecas Necessárias** - NumPy, Pandas, Scikit-learn, Matplotlib
2. **Carregar e Preparar Dataset Iris** - Carregar dados e dividir 70/30 treino/teste
3. **Treinar Modelo CLONALG** - Treinar com tamanho de população 50, 50 gerações
4. **Avaliar Desempenho** - Acurácia de teste, precisão, recall, matriz de confusão
5. **Analisar Evolução da Acurácia** - Gráfico de como a acurácia melhora entre gerações
6. **Analisar Impacto do Tamanho da População** - Testar tamanhos 10-50, comparar resultados
7. **Resumo e Conclusões** - Principais achados e observações

### 3. `population_analysis.py`
Script Python autossuficiente para análise rápida do impacto do tamanho da população sem Jupyter.

## Visão Geral do Algoritmo

### Passos do CLONALG:

1. **Inicialização**: Criar população de P anticorpos aleatórios (um por classe + aleatórios)
2. **Avaliação**: Calcular aptidão (afinidade) para cada anticorpo
3. **Seleção**: Selecionar os melhores M anticorpos com maior afinidade
4. **Clonagem**: Clonar cada anticorpo selecionado proporcionalmente à sua afinidade
   - Afinidade maior → Mais clones
5. **Hipermutação**: Mutar todos os clones inversamente à afinidade
   - Afinidade menor → Taxa de mutação maior
6. **Metadinâmica**: Substituir piores anticorpos por N novos aleatórios
7. **Loop**: Repetir passos 2-6 até critério de parada

### Fórmulas Matemáticas:

**Número de clones para o anticorpo k:**
$$QC_k = \left(\frac{af_k}{\sum_{i=1}^{n} af_i}\right) \times CL$$

**Taxa de mutação (inversamente proporcional à afinidade):**
$$hyperAnt_k = (1 - \frac{afin_k}{afin_{max}}) \times \beta$$

onde:
- `QC_k` = Número de clones para o anticorpo k
- `af_k` = Afinidade do anticorpo k
- `CL` = Número total de clones a gerar
- `hyperAnt_k` = Taxa de hipermutação
- `afin_k` = Afinidade do anticorpo k
- `afin_max` = Máxima afinidade na população
- `β` = Fator de mutação (tipicamente 0.1 a 10)

## Dataset Iris

- **Amostras**: 150 flores iris
- **Características**: 4 (comprimento de sépala, largura de sépala, comprimento de pétala, largura de pétala)
- **Classes**: 3 (setosa, versicolor, virginica)
- **Divisão**: 70% treinamento (105), 30% teste (45)

## Como Usar

### Executando o Notebook Jupyter:

```bash
jupyter notebook iris_classification.ipynb
```

O notebook irá:
1. Treinar CLONALG com tamanho de população 50
2. Exibir acurácia de teste e treinamento
3. Mostrar matriz de confusão e relatório de classificação
4. Traçar evolução da acurácia em 50 gerações
5. Analisar impacto de tamanhos de população 10-50
6. Gerar visualizações

### Executando o Script de Análise de População:

```bash
python population_analysis.py
```

Este script irá:
1. Testar tamanhos de população de 10 a 50 (passo 5)
2. Treinar cada um por 50 gerações
3. Exibir resultados em uma tabela
4. Criar gráficos de comparação
5. Salvar resultados como PNG

## Parâmetros

### Parâmetros Padrão do CLONALG:
- **Tamanho da População (P)**: 50 anticorpos
- **Gerações**: 50 iterações
- **Classes**: 3 (espécies de Iris)
- **Características**: 4 (medidas de flores)
- **Beta (fator de mutação)**: 1.0

### Parâmetros Configuráveis:
```python
clonalg = CLONALG(
    population_size=50,    # M = population_size // 3
    n_generations=50,      # Número de iterações
    n_classes=3,          # Número de espécies de iris
    n_features=4,         # Número de características
    beta=1.0              # Fator de taxa de mutação
)
```

## Resultados e Desempenho

### Resultados Esperados:
- **Acurácia de Teste**: 85-95%
- **Acurácia de Treinamento**: 90-98%
- **Convergência**: Tipicamente melhora até a geração 30-40

### Impacto do Tamanho da População:
- Populações maiores → Melhor acurácia (mais exploração)
- Populações menores → Mais rápido mas pode ficar preso
- Compromisso entre custo computacional e qualidade

## Principais Insights

1. **Estratégia de Clonagem**: Clonagem proporcional baseada em aptidão garante que soluções de alta qualidade tenham mais chances de melhorar

2. **Controle de Mutação**: Taxa de mutação inversa previne sobre-mutação de boas soluções enquanto mantém exploração

3. **Manutenção da Diversidade**: Geração de anticorpos aleatórios previne convergência prematura

4. **Eficaz para Classificação**: CLONALG aprende com sucesso limites de classe através de evolução

## Gráficos Gerados

1. **accuracy_evolution.png**: Acurácia de teste e treino entre gerações
2. **population_size_impact.png**: Análise de impacto com duas visualizações
3. **population_size_analysis.png**: Resultados de análise autossuficiente

## Requisitos

```
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.5.0
jupyter>=1.0.0
```

Instalar com:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy jupyter
```

## Referências

A implementação é baseada no algoritmo CLONALG descrito em:

de Castro, L. N., & Von Zuben, F. J. (2002). Learning and optimization in the immune system. 
IEEE Transactions on Evolutionary Computation, 6(3), 239-251.

## Notas

- O algoritmo usa distância Euclidiana para classificação
- Anticorpos são representados como vetores de peso no espaço de características
- Aptidão é calculada com base em acurácia de classificação
- O algoritmo mantém uma população em estado estável com elitismo

## Autor

Implementação para: Trabalho Prático 2 - Inteligência Artificial (5º Período)

## Licença

Somente Uso Educacional
