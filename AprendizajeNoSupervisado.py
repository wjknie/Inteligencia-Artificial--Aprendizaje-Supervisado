### **Actividad 4 - Métodos de aprendizaje no supervisado**
#Siguiendo las técnicas de agrupamiento (Clustering) explicadas en el Capítulo 16 de Palma Méndez (2008), el objetivo es encontrar grupos naturales u homogéneos dentro de los datos sin etiquetas previas. Para un sistema de transporte masivo, se identifican las siguientes variables clave para agrupar estaciones o franjas horarias:
#Afluencia de Pasajeros (Entradas/Salidas): Permite identificar estaciones de alta o baja demanda.
#Tiempo de Espera Promedio (minutos): Ayuda a caracterizar la eficiencia del servicio.
#Índice de Ocupación de los Vehículos (%): Identifica saturación en horas pico.
#Frecuencia de Paso de Vehículos (por hora): Describe la oferta de transporte.
#Al no contar con un dataset público inmediato con todas estas especificaciones locales, procedemos a desarrollar un dataset sintético que simule el comportamiento de 100 estaciones de transporte masivo para aplicar un modelo de aprendizaje no supervisado (K-Means).

#MODULO 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Generar un dataset sintético para 100 estaciones de transporte masivo
np.random.seed(42)

# Simular 3 perfiles típicos de estaciones:
# - Grupo 1: Estaciones residenciales (Afluencia media-baja, tiempos de espera medios)
# - Grupo 2: Estaciones troncales / terminales (Afluencia muy alta, tiempos de espera bajos por alta frecuencia)
# - Grupo 3: Estaciones periféricas o de baja demanda (Afluencia baja, tiempos de espera altos)

n_samples = 100
afluencia = np.concatenate([
    np.random.normal(3000, 500, 40),  # Residenciales
    np.random.normal(8000, 1000, 30), # Troncales
    np.random.normal(1000, 300, 30)   # Periféricas
])

tiempo_espera = np.concatenate([
    np.random.normal(6, 1.5, 40),
    np.random.normal(3, 0.8, 30),
    np.random.normal(12, 2.5, 30)
])

# Asegurar valores positivos
afluencia = np.clip(afluencia, 100, None)
tiempo_espera = np.clip(tiempo_espera, 1, None)

# Crear el DataFrame
df_transporte = pd.DataFrame({
    'ID_Estacion': [f'Estacion_{i+1}' for i in range(n_samples)],
    'Afluencia_Diaria': afluencia,
    'Tiempo_Espera_Prom_Min': tiempo_espera
})
#MODULO 2
# Display de los primeros registros
print("Vista previa de las fuentes de datos generadas:")
display(df_transporte.head())

# 2. Preprocesamiento: Escalar los datos (Crucial para K-Means)
features = ['Afluencia_Diaria', 'Tiempo_Espera_Prom_Min']
X = df_transporte[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Aplicar K-Means (Buscamos 3 clústeres basados en nuestro diseño teórico)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_transporte['Cluster'] = kmeans.fit_predict(X_scaled)

# Mostrar el resumen de los clústeres identificados
resumen_clusters = df_transporte.groupby('Cluster')[features].mean().reset_index()
print("\nValores promedio por Clúster:")
display(resumen_clusters)

# 4. Visualización de los Clústeres
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Afluencia_Diaria',
    y='Tiempo_Espera_Prom_Min',
    hue='Cluster',
    palette='viridis',
    data=df_transporte,
    s=100,
    alpha=0.8
)
plt.title('Segmentación de Estaciones de Transporte Masivo (K-Means)')
plt.xlabel('Afluencia Diaria de Pasajeros')
plt.ylabel('Tiempo de Espera Promedio (Minutos)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()