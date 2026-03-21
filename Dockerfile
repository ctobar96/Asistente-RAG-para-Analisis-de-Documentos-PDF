FROM python:3.11-slim
WORKDIR /app

# Instalar dependencias
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Exponer el puerto si es necesario (opcional)
# EXPOSE 8000
# Comando para ejecutar la aplicación
CMD ["python", "asistente_pdf.py"]
