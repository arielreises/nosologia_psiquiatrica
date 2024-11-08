# pip install pandas numpy scipy scikit-learn matplotlib

# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 1. Carregar o Dataset
df = pd.read_csv('dataset_pesquisa_transtornos.csv')

# Verificar as colunas disponíveis no dataset
print("Colunas disponíveis no dataset:", df.columns)
print("")

# 2. Selecionar as colunas relevantes para clustering (excluindo ID e Diagnóstico)
features = df[['Idade', 'Biomarcador_fMRI', 'HAMD_score', 'BAI_score']].dropna()

# ================================
# 1. Teste ANOVA
# ================================
# Verificar se a coluna Diagnóstico existe e realizar ANOVA com grupos de diagnóstico
print("Teste ANOVA")
if 'Diagnostico' in df.columns:
    # Separar os grupos de acordo com o diagnóstico
    grupos = df['Diagnostico'].unique()
    anova_grupos_idade = [df[df['Diagnostico'] == grupo]['Idade'] for grupo in grupos]
    anova_grupos_fmri = [df[df['Diagnostico'] == grupo]['Biomarcador_fMRI'] for grupo in grupos]

    # Teste ANOVA para 'Idade'
    anova_result_idade = f_oneway(*anova_grupos_idade)
    print('ANOVA para Idade - Estatística F:', anova_result_idade.statistic, 'Valor-p:', anova_result_idade.pvalue)

    # Teste ANOVA para 'Biomarcador_fMRI'
    anova_result_fmri = f_oneway(*anova_grupos_fmri)
    print('ANOVA para Biomarcador_fMRI - Estatística F:', anova_result_fmri.statistic, 'Valor-p:', anova_result_fmri.pvalue)
else:
    print("Coluna 'Diagnostico' não encontrada, pulando a análise ANOVA.")
print("")

# ================================
# 2. K-Means
# ================================
# Método do Cotovelo para encontrar o número ideal de clusters
print("K-Means")
inercia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(features)
    inercia.append(kmeans.inertia_)

# Visualizando o Método do Cotovelo
plt.plot(K, inercia, 'bx-')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo - K-Means')
plt.show()
print("")

# Definindo K=3 como o número ideal de clusters para este exemplo
kmeans = KMeans(n_clusters=3, random_state=42)
clusters_kmeans = kmeans.fit_predict(features)

# Adicionando os clusters ao DataFrame
df['cluster_kmeans'] = clusters_kmeans

# Avaliando o Coeficiente de Silhueta para os clusters do K-Means
silhouette_kmeans = silhouette_score(features, clusters_kmeans)
print("Coeficiente de Silhueta para K-Means: ", silhouette_kmeans)

# ================================
# 3. DBSCAN
# ================================
# Definindo o DBSCAN
print("DBSCAN")
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters_dbscan = dbscan.fit_predict(features)

# Adicionando os clusters ao DataFrame
df['cluster_dbscan'] = clusters_dbscan

# Visualizando os clusters formados pelo DBSCAN
plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=clusters_dbscan, cmap='plasma')
plt.title('Clusters formados pelo DBSCAN')
plt.show()
print("")

# ================================
# 4. Clustering Hierárquico
# ================================
print("Clustering Hierárquico")
# Gerando a matriz de linkage para o dendrograma
Z = linkage(features, method='ward')

# Visualizando o dendrograma com melhorias na formatação
plt.figure(figsize=(12, 8))  # Aumentar o tamanho do gráfico
dendrogram(
    Z,
    leaf_rotation=90.,  # Rotacionar os rótulos no eixo X para evitar sobreposição
    leaf_font_size=15.,  # Ajustar o tamanho da fonte dos rótulos
    color_threshold=0.7 * max(Z[:, 2]),  # Colocar um limite para a cor dos clusters
    above_threshold_color='grey'  # Definir a cor dos ramos acima do limite
)
plt.title('Dendrograma de Clustering Hierárquico')
plt.xlabel('Amostras')
plt.ylabel('Distância Euclidiana')
plt.tight_layout()  # Ajustar o layout para que tudo caiba no gráfico
plt.show()

# Definindo o Clustering Aglomerativo com 3 clusters
hierarchical_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
clusters_hierarchical = hierarchical_clustering.fit_predict(features)

# Adicionando os clusters ao DataFrame
df['cluster_hierarchical'] = clusters_hierarchical

# Avaliando o Coeficiente de Silhueta para Clustering Hierárquico
silhouette_hierarchical = silhouette_score(features, clusters_hierarchical)
print("Coeficiente de Silhueta para Clustering Hierárquico: ", silhouette_hierarchical)

# ================================
# Salvando o DataFrame com os clusters
# ================================
# Adicionando os resultados no CSV de saída
df.to_csv('resultado_clusters_atualizado.csv', index=False)
print("Arquivo com os clusters salvo como 'resultado_clusters_atualizado.csv'")
