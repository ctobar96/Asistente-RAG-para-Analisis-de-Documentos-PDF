# Modelo de representación de texto (Embeddings)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def crear_embeddings():
    """Inicializa el modelo de embeddings de Google Generative AI."""
    print("Inicializando modelo de embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    print("Modelo de embeddings inicializado.")
    return embeddings