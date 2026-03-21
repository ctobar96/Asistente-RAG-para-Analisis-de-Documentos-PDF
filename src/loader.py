# Ingesta de datos
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def cargar_documento(ruta_pdf):
    """
Carga un PDF y lo divide en fragmentos (chunks) para su posterior procesamiento.
    """
    print(f"Cargando documento: {ruta_pdf}...")
    # Extracción de Datos: Cargar el PDF
    loader = PyPDFLoader(ruta_pdf)
    docs = loader.load()
    
    print("Dividiendo el documento en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Documento dividido en {len(splits)} fragmentos.")
    
    return splits
