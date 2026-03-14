# Base de datos vectorial
"""
A este concepto se le llama Persistencia (o caché local), 
y su objetivo es ahorrarte tiempo, memoria y llamadas a la API de Google.
"""

from langchain_chroma import Chroma
import os   

def crear_vectorstore(splits, modelo_embeddings, directorio_persistencia="chroma_db"):
    """
    Crea una base de datos vectorial utilizando Chroma a partir de los fragmentos de texto y los embeddings
    O cargar una existente para ahorrar tiempo y recursos.
    """
    if os.path.exists(directorio_persistencia) and os.listdir(directorio_persistencia):
        print("Cargando base de datos vectorial existente...")
        vectorstore = Chroma(
            persist_directory=directorio_persistencia,
            embedding_function=modelo_embeddings
        ) 
    else:
        print("Creando y guardando nueva base de datos vectorial...")
        vectorstore = Chroma.from_documents(
            documents=splits,               # Toma todos los fragmentos de texto (splits).
            embedding=modelo_embeddings,    # Se conecta a la API de Google usando el modelo.
            persist_directory=directorio_persistencia # Guarda la base de datos para futuras cargas.
        )
    
    return vectorstore
    