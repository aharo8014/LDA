import streamlit as st
import pandas as pd
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import plotly.express as px
import re
from nltk.corpus import stopwords
import nltk

# Descargar los stopwords de NLTK
nltk.download('stopwords')

# Título y descripción de la app
st.title("Gensim - Procesamiento de Lenguaje Natural para la Investigación Científica")
st.markdown("""
**Creado por [Alexander Haro S.](https://scholar.google.com/citations?user=dFRviMUAAAAJ&hl=es&oi=ao)**  
Aplicación para el análisis de texto científico utilizando modelos de temas con Gensim.
""")

# Subir archivo CSV o XLSX
uploaded_file = st.file_uploader("Sube un archivo CSV o XLSX con una columna 'Abstract'", type=["csv", "xlsx"])

if uploaded_file:
    # Determinar el formato del archivo
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)

    # Verificar que la columna 'Abstract' exista
    if 'Abstract' not in data.columns:
        st.error("El archivo no contiene una columna llamada 'Abstract'.")
    else:
        # Filtrar abstracts válidos
        data = data[data['Abstract'] != '[No abstract available]'].dropna(subset=['Abstract'])

        # Selección de idiomas para las stopwords
        available_languages = sorted(stopwords.fileids())
        selected_languages = st.sidebar.multiselect(
            "Selecciona los idiomas de las stopwords:",
            available_languages,
            default=['english', 'spanish']
        )

        # Crear conjunto de stopwords basado en los idiomas seleccionados
        selected_stopwords = set()
        for lang in selected_languages:
            selected_stopwords.update(stopwords.words(lang))

        # Tokenizador básico y limpieza
        def simple_tokenizer(text):
            return re.findall(r'\b[a-zA-Z]+\b', text.lower())

        data['cleaned'] = data['Abstract'].apply(
            lambda x: ' '.join([word for word in simple_tokenizer(x) if word not in selected_stopwords])
        )

        # Crear diccionario y corpus
        dictionary = corpora.Dictionary(data['cleaned'].apply(lambda x: x.split()))
        corpus = [dictionary.doc2bow(text.split()) for text in data['cleaned']]

        # Configuración del modelo LDA
        num_topics = st.sidebar.slider("Número de temas", min_value=2, max_value=10, value=5, step=1)
        passes = st.sidebar.slider("Número de iteraciones (passes)", min_value=5, max_value=50, value=15, step=5)
        ldamodel = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)

        # Mostrar temas
        st.header("Temas identificados")
        topics = ldamodel.print_topics(num_words=5)
        for i, topic in enumerate(topics):
            st.subheader(f"Tema {i + 1}")
            st.write(topic[1])

        # Visualizar distribución de palabras por tema
        st.header("Visualización de temas")
        for i in range(num_topics):
            # Obtener las palabras y sus pesos para cada tema
            topic_terms = ldamodel.get_topic_terms(i, topn=10)
            words = [dictionary[word_id] for word_id, weight in topic_terms]
            weights = [weight for word_id, weight in topic_terms]

            # Crear gráfico de pastel
            fig = px.pie(
                names=words,
                values=weights,
                title=f"Distribución de palabras en el Tema {i + 1}",
                hole=0.3
            )
            st.plotly_chart(fig)

        # Mostrar datos procesados
        st.header("Datos procesados")
        st.dataframe(data[['Abstract', 'cleaned']].head(10))