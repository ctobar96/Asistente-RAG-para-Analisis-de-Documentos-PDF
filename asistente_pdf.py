import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Cargar la API Key de Google Generative AI, de forma segura desde el archivo de entorno
load_dotenv()  # Carga las variables de entorno desde un archivo .env

# Verificación de seguridad: Asegúrate de que la API Key esté presente
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("La variable de entorno 'GOOGLE_API_KEY' no está configurada. Por favor, configúrala en tu archivo .env.")


def analizar_documento(ruta_pdf, pregunta):
    """
    Carga un PDF, lo vectoriza y usa un LLM para responder preguntas sobre él.
    """
    print(f"Cargando documento: {ruta_pdf}...")
    
    # 2. Extracción de Datos: Cargar el PDF
    loader = PyPDFLoader(ruta_pdf)
    docs = loader.load()

    # 3. Ingeniería de Datos: Dividir el texto en fragmentos (chunks)
    # Esto es crucial para no saturar la memoria del modelo
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 4. Base de Datos Vectorial: Convertir texto a números (Embeddings)
    print("Creando base de datos vectorial temporal...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
        persist_directory="./chroma_db"
    )
    # Crear un "recuperador" para buscar la info más relevante
    retriever = vectorstore.as_retriever()

    # 5. Configurar el Modelo Generativo (LLM)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # 6. Definir las instrucciones (Prompt Engineering)
    system_prompt = (
        "Eres un asistente experto en analizar documentos técnicos y legales. "
        "Usa los siguientes fragmentos de contexto recuperados para responder a la pregunta. "
        "Si no sabes la respuesta basándote en el documento, di claramente que no lo sabes. "
        "Sé conciso y profesional."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 7. Crear la cadena de ejecución (Chain)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 8. Ejecutar la consulta
    print("Analizando y generando respuesta...\n")
    response = rag_chain.invoke({"input": pregunta})
    
    return response["answer"]

# --- Prueba del Asistente ---
if __name__ == "__main__":
    # Asegúrate de tener un archivo PDF en la misma carpeta para probar
    archivo_prueba = "data/pdf/Gobernanza de datos.pdf" 
    
    # Escribe lo que quieres saber del documento
    mi_pregunta = "Haz un resumen de los puntos más importantes de este documento."
    
    try:
        respuesta = analizar_documento(archivo_prueba, mi_pregunta)
        print("--- RESPUESTA DEL ASISTENTE ---")
        print(respuesta)
    except Exception as e:
        print(f"Ocurrió un error: {e}\n(Verifica que el archivo PDF exista y tu API Key sea válida).")