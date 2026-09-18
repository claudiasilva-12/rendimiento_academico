import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos
datos = pd.read_csv("data/StudentsPerformance.csv")


# Limpieza y preprocesamiento de datos
columnas_calificaciones = ["math score", "reading score", "writing score"]

datos[columnas_calificaciones] = datos[columnas_calificaciones].apply(
    pd.to_numeric, errors="coerce"
)

print("\nValores faltantes:")
print(datos.isnull().sum())

print("\nFilas duplicadas:", datos.duplicated().sum())

datos = datos.drop_duplicates()

datos = datos.dropna(subset=columnas_calificaciones)

# Mostrar las primeras filas
print(datos.head())

# Mostrar información del dataset
print("\nInformación del dataset:")
print(datos.info())

# Mostrar el tamaño del dataset
print("\nCantidad de filas y columnas:")
print(datos.shape)

# Estadísticas de las calificaciones
print("\nEstadísticas de las calificaciones:")
print(datos[["math score", "reading score", "writing score"]].describe())

# Promedio de cada materia
promedios = datos[["math score", "reading score", "writing score"]].mean()

# Crear gráfica
plt.figure(figsize=(8, 5))
promedios.plot(kind="bar")

plt.title("Promedio de calificaciones por materia")
plt.xlabel("Materia")
plt.ylabel("Promedio")
plt.ylim(0, 100)
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("outputs/promedio_materias.png")

# Promedio de calificaciones según el curso de preparación
preparacion = datos.groupby("test preparation course")[
    ["math score", "reading score", "writing score"]
].mean()

print("\nPromedio según el curso de preparación:")
print(preparacion)

# Crear gráfica
preparacion.plot(kind="bar", figsize=(8, 5))

plt.title("Promedio de calificaciones según curso de preparación")
plt.xlabel("Curso de preparación")
plt.ylabel("Promedio")
plt.ylim(0, 100)
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("outputs/preparacion_calificaciones.png")

# Promedio de calificaciones según el nivel educativo de los padres
educacion_padres = datos.groupby("parental level of education")[
    ["math score", "reading score", "writing score"]
].mean()

print("\nPromedio según el nivel educativo de los padres:")
print(educacion_padres)

# Crear gráfica
educacion_padres.plot(kind="bar", figsize=(10, 6))

plt.title("Promedio de calificaciones según educación de los padres")
plt.xlabel("Nivel educativo de los padres")
plt.ylabel("Promedio")
plt.ylim(0, 100)
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig("outputs/educacion_padres_calificaciones.png")

# Relación entre las calificaciones

correlacion = datos[["math score", "reading score", "writing score"]].corr()

print("\nCorrelación entre las calificaciones:")
print(correlacion)

# Crear gráfica de correlación

plt.figure(figsize=(8, 6))

plt.imshow(correlacion, cmap="Blues")

plt.colorbar()

plt.xticks(range(len(correlacion.columns)), correlacion.columns, rotation=45)
plt.yticks(range(len(correlacion.columns)), correlacion.columns)

plt.title("Correlación entre las calificaciones")

plt.tight_layout()

plt.savefig("outputs/correlacion_calificaciones.png")



# Calcular el promedio general de cada estudiante

datos["promedio_general"] = datos[
    ["math score", "reading score", "writing score"]
].mean(axis=1)

print("\nPromedio general de los estudiantes:")
print(datos["promedio_general"].describe())

# Crear gráfica de distribución del promedio general

plt.figure(figsize=(8, 5))

datos["promedio_general"].plot(kind="hist", bins=10)

plt.title("Distribución del promedio general")

plt.xlabel("Promedio")

plt.ylabel("Cantidad de estudiantes")

plt.xlim(0, 100)

plt.tight_layout()

plt.savefig("outputs/distribucion_promedio_general.png")


# Clasificar a los estudiantes según su promedio general

def clasificar_rendimiento(promedio):
    if promedio < 60:
        return "Bajo"
    elif promedio < 80:
        return "Medio"
    else:
        return "Alto"


datos["nivel_rendimiento"] = datos["promedio_general"].apply(
    clasificar_rendimiento
)


print("\nCantidad de estudiantes por nivel de rendimiento:")
print(datos["nivel_rendimiento"].value_counts())


# Crear gráfica de niveles de rendimiento
rendimiento = datos["nivel_rendimiento"].value_counts()

plt.figure(figsize=(8, 5))
rendimiento.plot(kind="bar")
plt.title("Cantidad de estudiantes por nivel de rendimiento")
plt.xlabel("Nivel de rendimiento")
plt.ylabel("Cantidad de estudiantes")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/niveles_rendimiento.png")
