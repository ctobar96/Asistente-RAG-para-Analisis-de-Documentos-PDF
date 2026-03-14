# La "inteligencia" y la capacidad de hablar
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def configurar_cadena_rag(retriever):
    """Configura el LLM Gemini 2.5 Flash y crea la cadena RAG."""
    print("Configurando el modelo generativo...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") # El modelo más potente de Google para tareas de generación de texto.

    system_prompt = ( 
        "Eres un asistente experto en analizar documentos técnicos y legales. "
        "Usa los siguientes fragmentos de contexto recuperados para responder a la pregunta. "
        "Si no sabes la respuesta basándote en el documento, di claramente que no lo sabes. "
        "Sé conciso, estructurado y profesional."
        "\n\n"
        "{context}" # Aquí se insertarán los fragmentos de texto relevantes recuperados por el retriever.
    )
    
    # La Plantilla de Conversación 
    prompt = ChatPromptTemplate.from_messages([ # Define el formato del prompt para el modelo generativo, combinando el sistema y la entrada del usuario.
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Crear la Cadena RAG: Combina el modelo generativo con el retriever para crear una cadena de ejecución que 
    # primero recupera información relevante y luego genera una respuesta basada en esa información.
    question_answer_chain = create_stuff_documents_chain(llm, prompt) 
    
    # Crea la cadena RAG que integra el retriever con el modelo generativo para responder preguntas basadas en los documentos recuperados.
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)    
     
    return rag_chain