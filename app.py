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

# 4. Interfaz de usuario para hacer preguntas
pregunta = st.text_input("Escribe tu pregunta sobre el documento:")

if st.button("Consultar", type="primary"):
    if pregunta:
        # Creamos un espacio temporal para el mensaje de carga
        mensaje_estado = st.empty()
        mensaje_estado.info("⏳ Analizando el documento y generando respuesta, por favor espera...")
        
        try:
            # Llamamos a nuestro motor RAG
            respuesta = cadena_rag.invoke({"input": pregunta})
            
            # Borramos el mensaje de carga
            mensaje_estado.empty()
            
            # Mostramos el resultado final
            st.markdown("### 🤖 Respuesta del Asistente RAG:")
            st.success(respuesta["answer"]) # Usamos st.success para un cuadro verde bonito
            
        except Exception as e:
            mensaje_estado.empty()
            st.error(f"❌ Ocurrió un error en el servidor: {e}")
    else:
        st.warning("⚠️ Por favor, escribe una pregunta primero.")