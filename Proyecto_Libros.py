from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Definir los géneros y la base de la URL
generos = ["Literatura-ficción", "Religión-espiritualidad", "libros-infantiles", "Salud-bienestar", "Humor-entretenimiento", "Infantil", "Autoayuda", "Académico", "Clásicos",
           "libros-novelas", "libros-no-ficcion", "Deporte", "Historia", "Ingles", "Misterio", "Novela", "Romance", "Salud-bienestar", "Comics-manga", "Crianza-familia"]

base_url = "https://listado.mercadolibre.com.co/libros-revistas-comics/{}/libros_Desde_{}"

# Definir un namedtuple para almacenar la información de los libros
Información_libros = namedtuple('Información_libros', [
    'Titulo',
    'Precio',
    'Calificacion',
    'Numero_reseña',
    'Cupon',
    'Cuotas',
    'Envio_gratis',
    'Link_libro',
    'libros_vendidos',
    'Genero',
])

lista_libros = []

# Agregar encabezados para simular un navegador real
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36',
}

# Iterar sobre los géneros
for genero in generos:
    # Iterar sobre las páginas
    for offset in range(0, 751, 50):  # Cambia 451 al número máximo de elementos que esperas
        url = base_url.format(genero, offset)

        # Realizar la solicitud a la página
        r = requests.get(url, headers=headers)

        # Revisar el código de estado de la respuesta
        print(f"Estado de la respuesta para {url}: {r.status_code}")

        # Ver el contenido HTML si la respuesta fue exitosa
        if r.status_code == 200:
            html = r.text
            html_soup = bs(html, 'html.parser')

            # Encontrar todos los libros en la página
            Libros = html_soup.find_all(
                'li', class_="ui-search-layout__item shops__layout-item")
            print(f"Número de libros encontrados en {url}: {len(Libros)}")

            # Iterar sobre cada libro encontrado
            for libro in Libros:
                titulo = libro.find('h2', class_=["poly-box poly-component__title"]).text if libro.find(
                    'h2', class_=["poly-box poly-component__title"]) else None
                precio = libro.find('span', class_=["andes-money-amount__fraction"]).text if libro.find(
                    'span', class_=["andes-money-amount__fraction"]) else None
                calificacion = libro.find('span', class_=[
                                          "poly-reviews__rating"]).text if libro.find('span', class_=["poly-reviews__rating"]) else None
                numero_reseña = libro.find('span', class_=[
                                           "poly-reviews__total"]).text if libro.find('span', class_=["poly-reviews__total"]) else None
                cupon = libro.find('span', class_=["poly-coupons__coupon"]).text if libro.find(
                    'span', class_=["poly-coupons__coupon"]) else None

                # Extraer información sobre cuotas
                cuotas = libro.find('span', class_=["poly-text-primary"])
                cuotas_texto = cuotas.find_next(
                    string=True).strip() if cuotas else None

                # Extraer la información sobre envío gratis
                envio_gratis = "Envío gratis" if libro.find(
                    'span', class_=["poly-component__shipping"]) else "No disponible"

                # Extraer la cantidad de libros vendidos
                libros_vendidos = libro.find('span', class_=["ui-pdp-subtitle"]).text if libro.find(
                    'span', class_=["ui-pdp-subtitle"]) else "No disponible"

                # Extraer el enlace del libro
                link = libro.find('h2', class_="poly-box poly-component__title").find('a')[
                    'href'] if libro.find('h2', class_="poly-box poly-component__title") else None

                # Crear una instancia de Información_libros y añadirla a la lista
                lista_libros.append(Información_libros(
                    titulo, precio, calificacion, numero_reseña, cupon, cuotas_texto, envio_gratis, libros_vendidos, link, genero
                ))

        else:
            print(f"No se pudo acceder a la página: {url}")

# Convertir la lista de libros a un DataFrame de pandas
df_libros = pd.DataFrame(lista_libros)

# Mostrar las primeras filas del DataFrame
print(df_libros.head(10))

# Obtener el total de libros recolectados
total_libros = len(df_libros)
print(f"Total de libros recolectados: {total_libros}")

# Eliminar filas con valores faltantes
df_libros.dropna(inplace=True)

# Limpiar la columna 'Precio' (Eliminar puntos y convertir a numérico)
df_libros['Precio'] = df_libros['Precio'].str.replace(
    '.', '', regex=False).astype(float)

# Convertir 'Calificacion' a numérico, manejando posibles valores nulos o no numéricos
df_libros['Calificacion'] = pd.to_numeric(
    df_libros['Calificacion'], errors='coerce')

# Limpiar la columna 'Numero_reseña' (eliminar paréntesis y convertir a numérico)
df_libros['Numero_reseña'] = df_libros['Numero_reseña'].str.extract(
    '(\d+)').astype(float)

# Verificar si las columnas han sido convertidas correctamente
print(df_libros[['Precio', 'Calificacion', 'Numero_reseña']].head())

# Convierte columnas categóricas a variables numéricas
df_libros = pd.get_dummies(df_libros, columns=['Genero'], drop_first=True)

# Análisis con estadísticas descriptivas
print(df_libros.describe())

# REGRESIÓN LINEAL
libros_data = df_libros.dropna()

# Variables dependientes y independientes
X = libros_data[['Calificacion', 'Numero_reseña']]
Y = libros_data['Precio']  # dependiente

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

# Crear objeto de regresión lineal
regr = LinearRegression()

# Entrenar el modelo usando los conjuntos de entrenamiento
regr.fit(X_train, Y_train)

# Hacer predicciones usando el conjunto de prueba
Y_pred = regr.predict(X_test)

# Coeficientes
print("Coeficientes de la regresión lineal: \n", regr.coef_)
# Intercepto
print("Intercepto de la regresión lineal: \n", regr.intercept_)
# Error cuadrático medio
print("Error cuadrático medio de la regresión lineal: %.2f" %
      mean_squared_error(Y_test, Y_pred))
# Coeficiente de determinación: 1 es predicción perfecta
print("Coeficiente de determinación de la regresión lineal: %.2f" %
      r2_score(Y_test, Y_pred))

# REGRESIÓN POLINOMIAL
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train_poly, X_test_poly, Y_train_poly, Y_test_poly = train_test_split(
    X_poly, Y, test_size=0.2, random_state=42)

# Crear objeto de regresión lineal para polinomios
regr_poly = LinearRegression()
regr_poly.fit(X_train_poly, Y_train_poly)

# Hacer predicciones usando el conjunto de prueba
Y_pred_poly = regr_poly.predict(X_test_poly)

# Evaluar el modelo de regresión polinomial
print("Error cuadrático medio de la regresión polinomial: %.2f" %
      mean_squared_error(Y_test_poly, Y_pred_poly))
print("Coeficiente de determinación de la regresión polinomial: %.2f" %
      r2_score(Y_test_poly, Y_pred_poly))

# ÁRBOL DE DECISIÓN
tree_regressor = DecisionTreeRegressor(random_state=42)
tree_regressor.fit(X_train, Y_train)

# Hacer predicciones
Y_pred_tree = tree_regressor.predict(X_test)

# Evaluar el árbol de decisión
print("Error cuadrático medio del árbol de decisión: %.2f" %
      mean_squared_error(Y_test, Y_pred_tree))
print("Coeficiente de determinación del árbol de decisión: %.2f" %
      r2_score(Y_test, Y_pred_tree))


# RandomForest

# Crear y ajustar el modelo de Random Forest
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train, Y_train)

# Hacer predicciones
Y_pred_rf = rf_regressor.predict(X_test)

# Evaluar el modelo
mse_rf = mean_squared_error(Y_test, Y_pred_rf)
r2_rf = r2_score(Y_test, Y_pred_rf)

print("MSE del modelo de Random Forest: %.2f" % mse_rf)
print("R² del modelo de Random Forest: %.2f" % r2_rf)

# BOSQUE ALEATORIO
forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
forest_regressor.fit(X_train, Y_train)

# Hacer predicciones
Y_pred_forest = forest_regressor.predict(X_test)

# Evaluar el bosque aleatorio
mse_forest = mean_squared_error(Y_test, Y_pred_forest)
r2_forest = r2_score(Y_test, Y_pred_forest)
mae_forest = mean_absolute_error(Y_test, Y_pred_forest)
rmse_forest = mean_squared_error(Y_test, Y_pred_forest, squared=False)
print("Error cuadrático medio del bosque aleatorio:", mse_forest)
print("Coeficiente de determinación del bosque aleatorio (R²):", r2_forest)
print("Error absoluto medio del bosque aleatorio:", mae_forest)
print("Raíz del error cuadrático medio del bosque aleatorio:", rmse_forest)

# --------------------------RECOMENDACIÓN------------------------------------

import pandas as pd

# Función para calcular el puntaje de cada libro
def calcular_puntaje(libro):
    puntaje = 0

    # Asignar puntos si hay descuento (Cupon)
    if libro['Cupon'] is not None:
        puntaje += 2

    # Asignar puntos si hay envío gratis
    if libro['Envio_gratis'] == "Envío gratis":
        puntaje += 1

    # Asignar puntos basados en la calificación
    if pd.notna(libro['Calificacion']):
        puntaje += libro['Calificacion']  # Directamente el valor de la calificación

    return puntaje  # Devolver el puntaje

# Calcular puntaje a cada libro
df_libros['Puntaje'] = df_libros.apply(calcular_puntaje, axis=1)

# Función para recomendar libros
def recomendar_libro(presupuesto, genero_seleccionado):
    # Filtrar libros según el presupuesto y el género
    libros_filtrados = df_libros[(df_libros['Precio'] <= presupuesto) & (df_libros['Genero'] == genero_seleccionado)]

    if libros_filtrados.empty:
        return "No hay libros disponibles en ese género dentro de tu presupuesto."

    # Ordenar los libros filtrados por puntaje (de mayor a menor)
    libros_filtrados = libros_filtrados.sort_values(by='Puntaje', ascending=False)

    # Seleccionar los tres libros con el puntaje más alto
    libros_recomendados = libros_filtrados.head(3)

    recomendaciones = []
    for index, libro in libros_recomendados.iterrows():
        recomendaciones.append(f"'{libro['Titulo']}', con un precio de {libro['Precio']}, una calificación de {libro['Calificacion']}, tiene {libro['Envio_gratis']} y se encuentra en el link: {libro['Link_libro']} .")

    return "Te recomendamos los siguientes libros:\n" + "\n".join(recomendaciones)


# Solicitar presupuesto y género al usuario
presupuesto = float(input("Ingresa tu presupuesto para el libro: "))
genero_seleccionado = input(f"Ingresa el género de libro que te interesa (opciones: {', '.join(df_libros['Genero'].unique())}): ")
# Mostrar la recomendación
print(recomendar_libro(presupuesto, genero_seleccionado))
