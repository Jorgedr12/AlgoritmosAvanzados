import pandas as pd
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

# PASO 1: Analizamos los datos (Sépalos vs Pétalos)
print("Generando gráfica de comparación")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
colores_map = {'Iris-setosa': 'purple', 'Iris-versicolor': 'teal', 'Iris-virginica': 'gold'}
colores_reales = df['class'].map(colores_map)

# Gráfica 1: Sépalos
ax1.scatter(df['sepal length'], df['sepal width'], c=colores_reales, alpha=0.6)
ax1.set_title('Sépalos')
ax1.set_xlabel('Largo del Sépalo')
ax1.set_ylabel('Ancho del Sépalo')

# Gráfica 2: Pétalos
ax2.scatter(df['petal length'], df['petal width'], c=colores_reales, alpha=0.6)
ax2.set_title('Pétalos')
ax2.set_xlabel('Largo del Pétalo')
ax2.set_ylabel('Ancho del Pétalo')

# Mostramos ambas gráficas
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

# PASO 6: Grafica final
print("Generando gráfica final con clusters")
plt.figure(figsize=(8, 6))

especies = df['prediccion'].unique()
colors_list = ['purple', 'teal', 'gold']

for i, especie in enumerate(especies):
    subset = df[df['prediccion'] == especie]
    plt.scatter(subset['petal length'], subset['petal width'], 
                c=colors_list[i], label=especie, s=50, alpha=0.7)

centros = kmeans.cluster_centers_
plt.scatter(centros[:, 0], centros[:, 1], 
            c='red', marker='X', s=100, label='Centroides')

plt.title(f'Resultados K-Means (Solo Pétalos) - Accuracy: {acc:.0%}')
plt.xlabel('Largo del Pétalo')
plt.ylabel('Ancho del Pétalo')
plt.legend()
plt.show()

# Hacemos un Excel con los resultados
print("Guardando resultados en Excel")
df.to_excel('resultados_kmeans_iris.xlsx', index=False)