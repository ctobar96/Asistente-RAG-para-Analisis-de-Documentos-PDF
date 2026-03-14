import streamlit as st
import os   
from dotenv import load_dotenv

# Importar funciones modulares del backend
from src.loader import cargar_documento
from src.embeddings import crear_embeddings
from src.vectorstore import crear_vectorstore
from src.rag_chain import configurar_cadena_rag 

# 1. Configuración de la aplicación
st.set_page_config(page_title="Asistente RAG", page_icon="🤖", layout="centered")
st.title("🤖 Asistente RAG para Análisis de Documentos PDF")
st.write("Carga un documento PDF y hazle preguntas para obtener respuestas basadas en su contenido.")

# 2. Cargar variables de entorno (como la clave de API de Google)
load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    st.error("¡Falta la API Key! Verifica tu archivo .env")
    st.stop()  # Detiene la ejecución si no hay API Key 

ruta_pdf = "data/pdf/Gobernanza de datos.pdf"  # Ruta relativa al proyecto


# 3. Inicializar el motor RAG (El caché evita que recargue el PDF en cada pregunta)
@st.cache_resource(show_spinner="Cargando documento y configurando el asistente...") # Esta función se ejecutará solo una vez y su resultado se almacenará en caché para futuras llamadas, evitando recargas innecesarias.
def iniciar_motor():
    splits = cargar_documento(ruta_pdf)
    modelo_embeddings = crear_embeddings()
    vectorstore = crear_vectorstore(splits, modelo_embeddings)
    retriever = vectorstore.as_retriever()
    rag_chain = configurar_cadena_rag(retriever)
    return rag_chain

try:
    cadena_rag = iniciar_motor()
except Exception as e:
    st.error(f"❌ Error al iniciar el backend: {e}")
    st.stop()

# 4. Interfaz de Chat (¡Más interactiva y moderna!)

# Inicializamos el historial de chat en la memoria del navegador
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Dibujamos los mensajes anteriores en la pantalla
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])

# La caja de texto flotante en la parte inferior (reemplaza al st.button)
pregunta = st.chat_input("Escribe tu pregunta sobre el documento:")

if pregunta:
    # 1. Mostrar y guardar la pregunta del usuario al instante
    with st.chat_message("user"):
        st.markdown(pregunta)
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})

    # 2. Mostrar el asistente pensando y luego su respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando miles de vectores..."):
            try:
                # Llamamos a tu motor backend
                respuesta = cadena_rag.invoke({"input": pregunta})
                
                # Mostramos el resultado
                st.markdown("### 🤖 Respuesta del Asistente RAG:")
                st.success(respuesta["answer"]) # Usamos st.success para un cuadro verde bonito
                
                # Guardamos la respuesta en el historial
                st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta["answer"]})
                
            except Exception as e:
                st.error(f"❌ Ocurrió un error en el servidor: {e}")
else:
    st.warning("⚠️ Por favor, escribe una pregunta primero.") 