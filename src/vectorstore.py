# Base de datos vectorial
"""
A este concepto se le llama Persistencia (o caché local), 
y su objetivo es ahorrarte tiempo, memoria y llamadas a la API de Google.
"""

import chromadb
from langchain_chroma import Chroma

def crear_vectorstore(splits, modelo_embeddings):
    # 1. Creamos un cliente aislado y desechable para engañar al bug de Chroma
    cliente_aislado = chromadb.EphemeralClient()
    
    # 2. Le pasamos este cliente específico a nuestra base de datos
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=modelo_embeddings,
        client=cliente_aislado
    )
    
    return vectorstore
    