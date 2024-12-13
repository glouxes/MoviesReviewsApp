import requests
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string

# Función para hacer peticiones a la API de TMDB
def peticion_api(url, headers):
    response = requests.get(url, headers=headers)
    return response.json()

# Función para preprocesar el texto
def preprocesar_texto(texto):
    # Convertir el texto a minúsculas
    texto = texto.lower()
    
    # Eliminar puntuación
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    
    # Eliminar stopwords (usamos las stopwords en inglés por defecto)
    stop_words = set(stopwords.words("english"))
    palabras = texto.split()
    palabras = [palabra for palabra in palabras if palabra not in stop_words]
    
    # Unir las palabras nuevamente
    return " ".join(palabras)

# Función para analizar el sentimiento de una reseña usando VADER
def analizar_sentimiento(texto):
    # Preprocesar el texto
    texto = preprocesar_texto(texto)
    
    # Inicializamos el analizador de sentimientos de VADER
    sia = SentimentIntensityAnalyzer()
    # Obtenemos el puntaje de sentimiento
    puntajes = sia.polarity_scores(texto)
    
    # Si el puntaje positivo es mayor que los otros, la emoción es positiva
    if puntajes['compound'] >= 0.05:
        return "Positivo", puntajes['compound']
    # Si el puntaje negativo es mayor, la emoción es negativa
    elif puntajes['compound'] <= -0.05:
        return "Negativo", puntajes['compound']
    # Si el puntaje es neutral, la emoción es neutral
    else:
        return "Neutral", puntajes['compound']
