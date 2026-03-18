# 📄 Asistente RAG para Análisis de Documentos PDF

Un asistente web interactivo impulsado por Inteligencia Artificial Generativa capaz de analizar, extraer información y responder preguntas específicas sobre cualquier documento PDF cargado dinámicamente, utilizando la arquitectura **RAG (Retrieval-Augmented Generation)**.

## 🎯 Objetivo del Proyecto
Este proyecto demuestra la implementación práctica de un flujo de **Ingeniería de Datos**, **MLOps** e **IA Generativa**. Permite que un Modelo de Lenguaje Grande (LLM) interactúe con datos no estructurados de forma segura y precisa, mitigando el riesgo de "alucinaciones" al restringir las respuestas estrictamente al contexto recuperado del documento.

## 🏗️ Arquitectura y Flujo de Datos

El sistema sigue el estándar de la industria para aplicaciones RAG, optimizado para entornos web:

1. **Ingesta Dinámica:** Interfaz web (Streamlit) para cargar documentos PDF en tiempo real, procesados en memoria temporal.
2. **Procesamiento y División (Chunking):** El texto se divide en fragmentos semánticos superpuestos para optimizar la retención de contexto sin saturar la memoria del modelo.
3. **Embeddings:** Transformación del texto a representaciones vectoriales de alta precisión utilizando los modelos de Google Generative AI.
4. **Almacenamiento Vectorial Efímero:** Uso de clientes aislados en memoria RAM (`EphemeralClient` de ChromaDB) para evitar colisiones de datos y garantizar un espacio de trabajo limpio por cada documento.
5. **Generación (LLM):** Integración con Gemini 1.5 Flash para sintetizar la respuesta final basándose únicamente en el contexto recuperado.

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python 3.11
* **Frontend:** Streamlit
* **Framework LLM:** LangChain
* **Modelo Generativo & Embeddings:** Google Gemini API (`gemini-1.5-flash`)
* **Base de Datos Vectorial:** ChromaDB
* **Procesamiento de Documentos:** PyPDF

## 📂 Estructura del Proyecto

```bash
Asistente-RAG-para-Analisis-de-Documentos-PDF/
│
├── .devcontainer/           # Configuración para contenedores de desarrollo (VS Code)
│   └── devcontainer.json
├── architecture/            # Diagramas de arquitectura del sistema
├── data/
│   └── pdf/                 # Carpeta para almacenar documentos PDF de prueba local
│       └── Gobernanza de datos.pdf
├── notebooks/               # Cuadernos Jupyter para experimentación
├── src/                     # Módulos core del backend
│   ├── embeddings.py        # Configuración de modelos de embeddings
│   ├── loader.py            # Lógica de carga y división de PDFs
│   ├── main.py              # Lógica principal del backend
│   ├── rag_chain.py         # Orquestación del pipeline RAG y prompts
│   └── vectorstore.py       # Configuración de base de datos vectorial
│
├── venv_RAG/                # Entorno virtual de Python (Ignorado en Git)
├── .dockerignore            # Archivos a ignorar en la construcción de la imagen Docker
├── .env                     # Variables de entorno (API keys - NO subir a Git)
├── .gitignore               # Archivos a ignorar en el control de versiones
├── app.py                   # 🚀 Archivo principal de la aplicación web Streamlit
├── Asistente RAG.md         # Notas o documentación adicional del proyecto
├── asistente_pdf.py         # Versión alternativa de ejecución por consola (CLI)
├── demo_interactiva.ipynb   # Demo interactiva del funcionamiento
├── Dockerfile               # Configuración para contenerización
├── LICENSE                  # Licencia de uso del proyecto
├── README.md                # Documentación principal
└── requirements.txt         # Dependencias y librerías del proyecto
```

## 🚀 Instalación y Uso Local
### 1. Clonar el repositorio
```bash
git clone https://github.com/ctobar96/Asistente-RAG-para-Analisis-de-Documentos-PDF.git
cd Asistente-RAG-para-Analisis-de-Documentos-PDF
```

### 2. Configurar el Entorno Virtual
Para evitar conflictos de versiones, crea y activa un entorno virtual:

```bash
# En Windows:
python -m venv venv_RAG
.\venv_RAG\Scripts\activate

# En Linux/Mac:
source venv_RAG/bin/activate
```

### 3. Instalar dependencias
Instala todas las librerías necesarias ejecutando:
```bash
pip install -r requirements.txt
```

### 4. Configuración de Variables de Entorno
Obtén una API Key gratuita desde [Google AI Studio](https://aistudio.google.com/). Crea un archivo .env en la raíz del proyecto y agrega tu clave:

```py
GOOGLE_API_KEY="TU_API_KEY_AQUI"
```

### 5. Ejecución Web (Recomendado)
Levanta la interfaz gráfica interactiva en tu navegador:

```bash
streamlit run app.py
```

## 👨‍💻 Autor

**Cristian Matías Tobar Morales**  
Ingeniero Civil Geológico | Magíster en Data Science  

GitHub: [@ctobar96](https://github.com/ctobar96)