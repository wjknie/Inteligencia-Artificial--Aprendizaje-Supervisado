#Métodos de aprendizaje supervisado
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Dataset transporte
np.random.seed(42)
n = 300

df = pd.DataFrame({
    'hora_pico': np.random.choice([0, 1], size=n, p=[0.5, 0.5]),
    'clima': np.random.choice(['Soleado', 'Lluvia'], size=n, p=[0.6, 0.4]),
    'pasajeros': np.random.randint(50, 300, size=n),
    'falla_mecanica': np.random.choice([0, 1], size=n, p=[0.85, 0.15])
})

#Variable objetivo: 1 = bus retrasado, 0 = a tiempo
df['retrasado'] = np.where((df['hora_pico'] == 1) & (df['clima'] == 'Lluvia') | (df['pasajeros'] > 220), 1, 0)

#Análisis Descriptivo
print("=== RESUMEN ESTADÍSTICO DE LOS DATOS ===")
print(df.describe().round(2))

#Gráfico Exploratorio de Datos
plt.figure(figsize=(10, 4))
sns.countplot(data=df, x='clima', hue='retrasado', palette='Set2')
plt.title("Análisis Exploratorio: Impacto del Clima en los Retrasos del Transporte")
plt.xlabel("Estado del Clima")
plt.ylabel("Cantidad de Buses")
plt.legend(title="Estado", labels=['A Tiempo', 'Retrasado'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

le = LabelEncoder()
df['clima_num'] = le.fit_transform(df['clima'])

#Separarar variables y dividor 80% train / 20% test
X = df[['hora_pico', 'clima_num', 'pasajeros', 'falla_mecanica']]
y = df['retrasado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#Division de datos
print(f"\n Preprocesamiento Exitoso:")
print(f"- Muestras de Entrenamiento (X_train): {X_train.shape[0]} buses")
print(f"- Muestras de Prueba (X_test): {X_test.shape[0]} buses")
print("\nPrimeras filas del conjunto listo para entrenar:")
X_train.head()

#Modulo 2
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

#Entrenar árbol de decisión
modelo = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
modelo.fit(X_train, y_train)

#Gráfico árbol de decisión
plt.figure(figsize=(12, 5))
plot_tree(modelo, feature_names=X.columns, class_names=['A Tiempo', 'Retrasado'], filled=True, rounded=True)
plt.title("Modelo Entrenado: Árbol de Decisión")
plt.show()

#Evaluar conjunto de prueba
y_pred = modelo.predict(X_test)
acc = accuracy_score(y_test, y_pred)

#Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['A Tiempo', 'Retrasado']).plot(cmap='Blues')
plt.title(f"Matriz de Confusión - Exactitud: {acc:.2%}")
plt.show()

#Prueba unitaria
nuevo_bus = [[1, 1, 250, 0]]
#Hora pico=1, clima lluvia=1, pasajeros=250, falla=0
pred = modelo.predict(nuevo_bus)

print("=== PRUEBA DEL COMPONENTE EN TIEMPO REAL ===")
print("Entrada:", nuevo_bus)
print(" Predicción del Modelo:", "BUS RETRASADO" if pred[0] == 1 else "BUS A TIEMPO")
