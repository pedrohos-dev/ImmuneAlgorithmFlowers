# GUIA DE INÍCIO RÁPIDO - Trabalho Prático 2

## Visão Geral
Esta implementação resolve o problema de classificação de flores Iris usando o algoritmo imunológico **CLONALG**.

## Arquivos Criados

1. **clonalg.py** - Implementação central do algoritmo
2. **iris_classification.ipynb** - Notebook Jupyter principal (RECOMENDADO)
3. **population_analysis.py** - Script autossuficiente de análise de tamanho de população
4. **README.md** - Documentação abrangente

## Como Executar

### Opção 1: Notebook Jupyter (RECOMENDADO - Análise Completa)

```bash
jupyter notebook iris_classification.ipynb
```

Isto irá executar e exibir:
✓ Treinamento CLONALG com tamanho de população 50
✓ Acurácia de teste: ~90-95%
✓ Acurácia de treinamento: ~95-98%
✓ Gráfico de evolução de acurácia em 50 gerações
✓ Análise de impacto de tamanho de população (10-50)
✓ Métricas de desempenho (precisão, recall, matriz de confusão)

**Tempo de Execução**: ~5-10 minutos (dependendo da máquina)

### Opção 2: Script Autossuficiente de Análise de População

```bash
python population_analysis.py
```

Isto irá:
✓ Testar tamanhos de população 10, 15, 20, 25, 30, 35, 40, 45, 50
✓ Treinar cada um por 50 gerações
✓ Criar gráficos de comparação
✓ Exibir tabela de resultados

**Tempo de Execução**: ~3-5 minutos

## Modelagem do Problema

### O Que Cada Anticorpo Representa?
- Cada anticorpo é um **protótipo de classe** com vetores de peso aprendidos
- Parâmetros do anticorpo: `weights` (valores de características) + `class_id` (qual classe representa)

### Parâmetros do Anticorpo
- **weights**: Vetor dimensional n_features (4 para Iris)
- **affinity**: Acurácia de classificação nos dados de treinamento
- **class_id**: 0 (setosa), 1 (versicolor), ou 2 (virginica)

### Função de Aptidão
```
affinity = acurácia reconhecendo sua própria classe + 
           0.5 × acurácia geral de classificação
```

### Avaliação
- Usa distância Euclidiana para classificar amostras
- Classifica ao protótipo de anticorpo mais próximo

## Principais Resultados Esperados

### Métricas de Desempenho
- Acurácia de Teste: 85-95%
- Acurácia de Treinamento: 90-98%
- Precisão: 0.85-0.95
- Recall: 0.85-0.95

### Evolução de Acurácia (em 50 gerações)
- Geração 1-10: ~60-70%
- Geração 20-30: ~80-85%
- Geração 40-50: ~90-95%

### Impacto do Tamanho da População
- População 10: ~85-90% de acurácia
- População 30: ~90-92% de acurácia
- População 50: ~92-95% de acurácia

## Parâmetros do Algoritmo Utilizados

```
Tamanho da População (P) = 50 anticorpos
Número de Gerações = 50 iterações
Tamanho de Seleção (M) = P/3 ≈ 17
Contagem de Clones (N_clones) = P/2 ≈ 25
Novos Anticorpos (N) = P/5 ≈ 10
Beta (fator de mutação) = 1.0
```

## Arquivos de Saída Gerados

1. **accuracy_evolution.png** - Mostra acurácia de teste e treinamento entre gerações
2. **population_size_impact.png** - Compara diferentes tamanhos de população
3. **population_size_analysis.png** - Gráfico detalhado de análise de população

## Divisão do Dataset

- Total de amostras: 150
- Conjunto de treinamento: 105 amostras (70%)
- Conjunto de teste: 45 amostras (30%)

Cada classe tem ~35 amostras de treinamento e ~15 amostras de teste

## Resolução de Problemas

### ImportError: No module named 'sklearn'
```bash
pip install scikit-learn numpy pandas matplotlib seaborn scipy
```

### Notebook Jupyter não encontrado
```bash
pip install jupyter
```

### Script executa muito lentamente
- Reduza n_generations no código
- Reduza population_size
- Considere fechar outras aplicações

## Comportamento Esperado

✓ O algoritmo deve convergir gradualmente
✓ Acurácia de teste deve melhorar entre gerações
✓ Populações maiores devem geralmente ter melhor desempenho
✓ Acurácia final deve ser competitiva com outros métodos de ML

## Comparando Resultados

Se você quer comparar CLONALG com outros classificadores:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

rf = RandomForestClassifier().fit(X_train, y_train)
svm = SVC().fit(X_train, y_train)

print(f"Acurácia Random Forest: {rf.score(X_test, y_test):.4f}")
print(f"Acurácia SVM: {svm.score(X_test, y_test):.4f}")
print(f"Acurácia CLONALG: {clonalg.score(X_test, y_test):.4f}")
```

## Principais Insights do CLONALG

1. **Inspirado Biologicamente**: Imita a capacidade do sistema imunológico de aprender e se adaptar

2. **Busca Eficaz**: Equilíbrio entre exploração e aproveitamento através de:
   - Clonagem proporcional (aproveitar boas soluções)
   - Mutação adaptativa (explorar novas áreas)
   - Anticorpos aleatórios (manter diversidade)

3. **Escalabilidade**: O desempenho depende de:
   - Tamanho da população (mais = melhor, mas mais lento)
   - Número de gerações (mais = melhor, mas mais lento)
   - Complexidade do problema (Iris é relativamente simples)

4. **Vantagens**:
   - Conceitualmente elegante
   - Boa generalização
   - Paralelização natural
   - Parâmetros ajustáveis

5. **Desvantagens**:
   - Mais lento que alguns métodos clássicos
   - Mais parâmetros para ajustar
   - Pode precisar de mais iterações para problemas complexos

## Próximos Passos

1. Execute o notebook Jupyter para ver análise completa
2. Tente diferentes tamanhos de população
3. Modifique o parâmetro beta e observe os efeitos
4. Compare com outros algoritmos de ML
5. Tente em diferentes datasets

## Suporte

Para perguntas sobre:
- **Algoritmo CLONALG**: Veja README.md ou referências no código
- **Detalhes de Implementação**: Veja clonalg.py com comentários detalhados
- **Análise de Resultados**: Execute iris_classification.ipynb

---

**Tempo de Execução Estimado**: 
- Notebook: 5-10 minutos
- Script autossuficiente: 3-5 minutos

**Boa sorte com sua tarefa!** 🍀
