import streamlit as st
import os   
from dotenv import load_dotenv
import tempfile 

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
    st.stop()  

# Widget para cargar el PDF (reemplaza la ruta fija)
archivo_subido = st.file_uploader("Carga tu documento PDF para analizar", type=["pdf"])

# 3. Inicializar el motor RAG
if archivo_subido is not None:
    # Creamos un archivo temporal físico en el servidor
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as archivo_temporal:
        archivo_temporal.write(archivo_subido.getvalue())
        ruta_temporal = archivo_temporal.name

    try:
        # Usamos st.spinner solo para la inicialización
        with st.spinner("Cargando documento y configurando el asistente..."):
            
            # --- FUNCIÓN DE CACHÉ ---
            @st.cache_resource(show_spinner=False) 
            def iniciar_motor(ruta):
                splits = cargar_documento(ruta)
                modelo_embeddings = crear_embeddings()
                vectorstore = crear_vectorstore(splits, modelo_embeddings) 
                retriever = vectorstore.as_retriever()
                rag_chain = configurar_cadena_rag(retriever)
                return rag_chain
            # ------------------------

            try:
                # Llamamos a la función con caché pasándole la ruta temporal
                cadena_rag = iniciar_motor(ruta_temporal)
                st.success("¡Documento procesado con éxito! Ya puedes hacer preguntas.")
            except Exception as e:
                st.error(f"❌ Error al iniciar el backend: {e}")
                st.stop()

        # 4. Interfaz de Chat 
        
        # Inicializamos el historial de chat en la memoria del navegador
        if "mensajes" not in st.session_state:
            st.session_state.mensajes = []

        # Dibujamos los mensajes anteriores en la pantalla
        for mensaje in st.session_state.mensajes:
            with st.chat_message(mensaje["rol"]):
                st.markdown(mensaje["contenido"])

        # La caja de texto flotante en la parte inferior
        pregunta = st.chat_input("Escribe tu pregunta sobre el documento:")

        if pregunta:
            # Mostrar y guardar la pregunta del usuario
            with st.chat_message("user"):
                st.markdown(pregunta)
            st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})

            # Mostrar el asistente pensando y luego su respuesta
            with st.chat_message("assistant"):
                with st.spinner("Analizando miles de vectores..."):
                    try:
                        respuesta = cadena_rag.invoke({"input": pregunta})
                        st.markdown("### 🤖 Respuesta del Asistente RAG:")
                        st.success(respuesta["answer"]) 
                        st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta["answer"]})
                    except Exception as e:
                        st.error(f"❌ Ocurrió un error en el servidor: {e}")

    finally:
        # Limpieza: Borramos el archivo temporal del servidor
        if os.path.exists(ruta_temporal):
            os.remove(ruta_temporal)

else:
    st.info("☝️ Por favor, sube un archivo PDF para comenzar.")
