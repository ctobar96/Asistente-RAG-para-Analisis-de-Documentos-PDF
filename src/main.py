# Archivo principal de la aplicación
import os
from dotenv import load_dotenv
# Importar funciones modulares
from src.loader import cargar_documento
from src.embeddings import crear_embeddings
from src.vectorstore import crear_vectorstore
from src.rag_chain import configurar_cadena_rag

def main():
    # Cargar variables de entorno (como la clave de API de Google)
    load_dotenv(".../.env")
    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("¡Falta la API Key! Verifica tu archivo .env")
    
    # Rutas con "../" porque ejecutaremos esto desde adentro de la carpeta "src"
    ruta_pdf = "../data/pdf/Gobernanza de datos.pdf" 
    directorio_db = "../chroma_db"
    pregunta = "¿Cuáles son las normas internacionales mencionadas en el documento y para qué sirven?"
    
    try:
        # Paso 1: Cargar y dividir el documento en fragmentos
        splits = cargar_documento(ruta_pdf)
        
        # Paso 2: Crear el modelo de embeddings
        modelo_embeddings = crear_embeddings()
        
        # Paso 3: Crear o cargar la base de datos vectorial
        vectorstore = crear_vectorstore(splits, modelo_embeddings)
        
        # Paso 4: Configurar la cadena RAG con el retriever del vectorstore
        retriever = vectorstore.as_retriever()  # Convierte el vectorstore en un retriever para la cadena RAG
        rag_chain = configurar_cadena_rag(retriever)
        
        print("\nAnalizando y generando respuesta...\n")
        respuesta = rag_chain.invoke({"input": pregunta})
        
        print("="*60)
        print("🤖 RESPUESTA DEL ASISTENTE RAG:")
        print("="*60)
        print(respuesta["answer"])
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Ocurrió un error en el pipeline: {e}")
    

if __name__ == "__main__":
    main()
    

    

    
