# 📄 Asistente RAG para Análisis de Documentos PDF

Un asistente virtual impulsado por Inteligencia Artificial Generativa capaz de analizar, extraer información y responder preguntas específicas sobre documentos PDF utilizando la arquitectura RAG (Retrieval-Augmented Generation).

## 🎯 Objetivo del Proyecto
Este proyecto demuestra la implementación práctica de un flujo de **Ingeniería de Datos** e **IA Generativa**, permitiendo que un Modelo de Lenguaje Grande (LLM) interactúe con datos no estructurados de forma segura y precisa, mitigando el riesgo de "alucinaciones" al restringir las respuestas estrictamente al contexto recuperado del documento.

## 🏗️ Arquitectura y Flujo de Datos

El sistema sigue el estándar de la industria para aplicaciones RAG:

1. **Ingesta de Datos:** Carga de documentos PDF locales.
2. **Procesamiento y División (Chunking):** El texto se divide en fragmentos semánticos superpuestos para optimizar la retención de contexto sin saturar la memoria del modelo.
3. **Embeddings:** Transformación del texto a representaciones vectoriales utilizando modelos de Google Generative AI.
4. **Almacenamiento Vectorial:** Uso de ChromaDB como base de datos vectorial temporal para indexar y buscar los fragmentos de texto más relevantes.
5. **Generación (LLM):** Integración con Gemini 1.5 Flash para sintetizar la respuesta final basándose únicamente en el contexto recuperado.

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python 3.x
* **Framework LLM:** LangChain
* **Modelo Generativo & Embeddings:** Google Gemini API (`gemini-1.5-flash`)
* **Base de Datos Vectorial:** ChromaDB
* **Procesamiento de Documentos:** PyPDF


## Estructura del proyecto
```bash
Asistente-RAG-para-Analisis-de-Documentos-PDF
│
├── data/
│   └── pdf/                 # PDFs que analizará el sistema
│
├── chroma_db/               # Base vectorial (se puede ignorar en git)
│
├── src/                     # Código fuente
│   ├── loader.py            # Carga de PDFs
│   ├── embeddings.py        # Generación de embeddings
│   ├── vectorstore.py       # Base vectorial
│   ├── rag_chain.py         # Pipeline RAG
│   └── main.py              # Programa principal
│
├── app/
│   └── streamlit_app.py    # interfaz web
│
├── notebooks/               # Experimentos
│   └── pruebas_rag.ipynb
│
├── architecture/
│   └── rag_pipeline.png
│
├── .env                     # API keys (NO subir)
├── .gitignore
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
└── asistente_pdf.py         # versión simple del asistente
```


## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/ctobar96/Asistente-RAG-para-Analisis-de-Documentos-PDF.git 

cd Asistente-RAG-para-Analisis-de-Documentos-PDF
```

### 2. Configurar el Entorno Virtual
Para evitar conflictos de versiones, crea y activa un entorno virtual en la carpeta del proyecto:
En Windows:
```bash
python -m venv venv_RAG
.\venv\Scripts\activate
```

### 3. Instalar dependencias
Instala todas las librerías necesarias ejecutando el archivo de requerimientos:

```bash
pip install -r requirements.txt
```

### 4. Configuración de Variables de Entorno
Obtén una API Key gratuita desde [Google AI Studio](https://aistudio.google.com/) y configúrala en tu entorno o reemplázala en el script asistente_pdf.py:

```py
os.environ["GOOGLE_API_KEY"] = "TU_API_KEY_AQUI"
```

### 5. Ejecución  
Coloca un archivo PDF en el directorio raíz del proyecto y ejecuta el script:
```bash
python asistente_pdf.py
```

## 👨‍💻 Autor

**Cristian Matías Tobar Morales**  
Ingeniero Civil Geológico | Magíster en Data Science  

GitHub: [@ctobar96](https://github.com/ctobar96)