# 🧠 ¿Qué es un Asistente RAG?

RAG significa **Retrieval-Augmented Generation.**

Es una arquitectura que combina:

1️⃣ Búsqueda de información en documentos.  
2️⃣ Generación de respuestas con un modelo de lenguaje.

Modelos como Google Gemini o GPT pueden responder preguntas, pero no conocen tus documentos privados.

RAG soluciona esto.

## ⚙️ Cómo funciona un sistema RAG con PDFs

El flujo típico es este:
``` bash
PDF → extracción de texto
       ↓
dividir en fragmentos (chunks)
       ↓
crear embeddings (vector numérico)
       ↓
guardar en base vectorial
       ↓
pregunta del usuario
       ↓
buscar fragmentos relevantes
       ↓
enviar contexto al modelo (Gemini)
       ↓
respuesta generada
```

## 📊 Por qué RAG es mejor que solo usar un LLM
**Sin RAG:**
``` bash
Pregunta → LLM → respuesta (puede inventar información)
``` 
**Con RAG:**
``` bash
Pregunta → búsqueda en documentos → LLM → respuesta basada en el PDF
``` 
Esto reduce alucinaciones del modelo.


