import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
from ucimlrepo import fetch_ucirepo

# Cargamos los datos de Iris
print("Cargando datos")
iris = fetch_ucirepo(id=53)
X = iris.data.features
y = iris.data.targets
df = X.copy()
df['class'] = y['class']

# Definimos un diccionario de colores para las clases
colores_dict = {'Iris-setosa': 'purple', 'Iris-versicolor': 'teal', 'Iris-virginica': 'gold'}

# PASO 1: Analizamos los datos (Sépalos vs Pétalos)
print("Generando gráfica de comparación")

sns.set_style("whitegrid")
g = sns.pairplot(df, hue='class', palette=colores_dict, markers=["o", "s", "D"])

# Ajustamos el título superior
g.fig.suptitle("Distribución Cruzada: Sépalos vs Pétalos", y=1.02)

# Guardamos la gráfica
plt.savefig('distribucion_total_iris.png')
plt.show()

# PASO 2: Selección de Representación
# Seleccionamos los datos de los pétalos para el entrenamiento del modelo K-means ya que visualmente parecen ser más diferenciables entre las clases.
X_entrenamiento = X[['petal length', 'petal width']].copy()

# PASO 3: Aplicar algoritmo K-Means
print("Entrenando modelo K-Means")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_entrenamiento)
print("Modelo entrenado")

# PASO 4: Mapeo entre culster y clase
print("Realizando mapeo entre cluster y clase")
mapping = {}
for i in range(3):
    moda = df[df['cluster'] == i]['class'].mode()[0]
    mapping[i] = moda
    print(f"Cluster {i} se mapea a clase {moda}")
df['prediccion'] = df['cluster'].map(mapping)

# PASO 5: Calculamos las metricas de evaluación
print("Calculando métricas de evaluación")
acc = accuracy_score(df['class'], df['prediccion'])
print(f"Exactitud del modelo K-Means: {acc:.2f}")
print(classification_report(df['class'], df['prediccion']))

# PASO 6: Gráfica final
print("Generando comparativa: Ground Truth vs Clusters")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


# Creamos la subgráfica 1 con las etiquetas reales
especies_reales = df['class'].unique()
for i, especie in enumerate(especies_reales):
    subset = df[df['class'] == especie]
    ax1.scatter(subset['petal length'], subset['petal width'], 
                c=colores_dict[especie], label=especie, s=50, alpha=0.7)
ax1.set_title('Ground Truth (Etiquetas Reales)')
ax1.set_xlabel('Largo del Pétalo')
ax1.set_ylabel('Ancho del Pétalo')
ax1.legend()

# Creamos la subgráfica 2 con las predicciones del modelo K-Means
especies_pred = df['prediccion'].unique()
for i, cluster in enumerate(especies_pred):
    subset = df[df['prediccion'] == cluster]
    ax2.scatter(subset['petal length'], subset['petal width'], 
                c=colores_dict[cluster], label=f'Cluster {cluster}', s=50, alpha=0.7)

# Añadimos los centroides en la gráfica de predicción
centros = kmeans.cluster_centers_
ax2.scatter(centros[:, 0], centros[:, 1], 
            c='red', marker='x', s=150, label='Centroides')

ax2.set_title(f'Resultados K-Means - Accuracy: {acc:.0%}')
ax2.set_xlabel('Largo del Pétalo')
ax2.legend()

plt.tight_layout()
plt.savefig('comparativa_final_iris.png')
plt.show()

# Hacemos un Excel con los resultados
print("Guardando resultados en Excel")
df.to_excel('resultados_kmeans_iris.xlsx', index=False)