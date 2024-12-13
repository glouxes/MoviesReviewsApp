import streamlit as st
import requests
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Lista simple de stopwords en español
stopwords = set([
    'of', 'the', 'and', 'to', 'in', 'is', 'that', 'with', 'for', 'it', 'on', 'as', 'a', 'about', 'at', 'between', 'more', 'but', 'has', 'this', 'are', 'was', 'have', 'be', 'everything', 'they', 'does', 'very', 'we', 'me', 'you', 'i', 'yes', 'no', 'was', 'if', 'well', 'she', 'when', 'those', 'we', 'some'
])

# Función para buscar película por título
def buscar_pelicula_por_titulo(titulo, api_key):
    url = f"https://api.themoviedb.org/3/search/movie?query={titulo}&api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        resultados = response.json().get('results', [])
        return resultados
    else:
        st.error("Error al buscar la película.")
        return []

# Función para obtener detalles de una película
def obtener_detalles_pelicula(movie_id, api_key):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error al obtener detalles de la película.")
        return {}

# Función para obtener reseñas de una película
def obtener_resenas(movie_id, api_key):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json().get('results', []))
    else:
        st.error("Error al obtener las reseñas.")
        return pd.DataFrame()

# Función para analizar sentimientos
def analizar_sentimientos(df):
    if 'content' in df.columns:
        df['sentiment'] = df['content'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['label'] = df['sentiment'].apply(lambda x: "Positivo" if x > 0 else ("Negativo" if x < 0 else "Neutral"))
    return df

# Función para generar nube de palabras
def generar_nube_de_palabras(texto):
    wordcloud = WordCloud(
        stopwords=stopwords,
        background_color="black",
        width=800,
        height=400,
        colormap="Set2"
    ).generate(texto)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    st.pyplot(plt)

# Interfaz de Streamlit
st.title("🎥 Análisis de Sentimientos en Películas")
st.markdown("Introduce el título de la película y tu API Key para analizar las reseñas.")

# Inputs iniciales
titulo_pelicula = st.text_input("Título de la película")
api_key = st.text_input("API Key de The Movie Database", type="password")

# Buscar películas y mostrar opciones
if st.button("Buscar"):
    if titulo_pelicula and api_key:
        peliculas = buscar_pelicula_por_titulo(titulo_pelicula, api_key)
        if peliculas:
            st.session_state['peliculas'] = peliculas
        else:
            st.warning("No se encontraron películas con ese título.")
    else:
        st.warning("Por favor, ingresa un título y tu API Key.")

# Mostrar opciones de películas si están cargadas
if 'peliculas' in st.session_state:
    peliculas = st.session_state['peliculas']
    opciones = [f"{p['title']} ({p.get('release_date', 'N/A')[:4]})" for p in peliculas]
    seleccion = st.selectbox("Selecciona una película", opciones)
    seleccion_index = opciones.index(seleccion)
    seleccion_id = peliculas[seleccion_index]['id']

    # Obtener detalles de la película seleccionada
    detalles = obtener_detalles_pelicula(seleccion_id, api_key)
    if detalles:
        st.subheader("📽️ Detalles de la Película")
        st.write(f"**Título Original:** {detalles.get('original_title', 'N/A')}")
        st.write(f"**Sinopsis:** {detalles.get('overview', 'N/A')}")
        st.write(f"**Fecha de Lanzamiento:** {detalles.get('release_date', 'N/A')}")
        st.write(f"**Popularidad:** {detalles.get('popularity', 'N/A')}")
        st.write(f"**Calificación Promedio:** {detalles.get('vote_average', 'N/A')} (basado en {detalles.get('vote_count', 0)} votos)")

    # Obtener reseñas de la película seleccionada
    reseñas = obtener_resenas(seleccion_id, api_key)
    if not reseñas.empty:
        reseñas_analizadas = analizar_sentimientos(reseñas)

        # Mostrar información y análisis
        st.subheader("📝 Reseñas")
        st.write(reseñas_analizadas[['author', 'content', 'label']])
        st.bar_chart(reseñas_analizadas['label'].value_counts())

        # Generar nube de palabras
        texto_reseñas = " ".join(reseñas_analizadas['content'].dropna())
        st.subheader("☁️ Nube de Palabras")
        generar_nube_de_palabras(texto_reseñas)
    else:
        st.warning("No hay reseñas disponibles para esta película.")
