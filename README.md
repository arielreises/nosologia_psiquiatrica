# Ciência de Dados Aplicada à Nosologia Psiquiátrica

Este repositório contém o código, dados e resultados do projeto sobre o uso de ciência de dados na nosologia psiquiátrica, com foco em técnicas de clustering para identificação de padrões em transtornos mentais. Este projeto visa propor uma abordagem mais dimensional e baseada em dados, em contraste com as classificações categóricas convencionais, como o DSM-5.

## Resumo

O estudo explorou a aplicação de algoritmos de agrupamento de clusters para reclassificação de transtornos mentais, identificando três principais clusters de transtornos com base em dados clínicos e biomarcadores. Este novo modelo sugere uma organização mais refinada, considerando sobreposições e comorbidades entre diagnósticos.

## Estrutura do Repositório

- Contém os datasets gerados e utilizados, com dados clínicos simulados, incluindo biomarcadores e sintomas.
- Inclui os scripts em Python usados para processamento de dados e clustering.
- Arquivos de saída dos modelos e análises, como gráficos de clusters e resultados de ANOVA.
- Descrição do projeto e instruções de uso.

## Principais Algoritmos e Técnicas

- **K-Means**: para particionar os dados em clusters com base em variância mínima dentro dos grupos.
- **DBSCAN**: ideal para lidar com clusters de formas arbitrárias e ruído, comuns em dados clínicos.
- **Clustering Hierárquico**: utilizado para revelar hierarquias naturais entre os transtornos mentais.
- **Análise de Variância (ANOVA)**: para verificar diferenças significativas entre grupos.
- **Coeficiente de Silhueta**: para avaliar a qualidade dos clusters formados.

## Requisitos

Para executar o projeto, você precisará instalar as seguintes bibliotecas Python:

```bash
pip install pandas numpy scipy scikit-learn matplotlib
