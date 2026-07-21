FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar paquetes
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requerimientos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código fuente del proyecto
COPY . .

# Google Cloud Run inyecta la variable de entorno $PORT (por defecto 8080)
# Streamlit debe escuchar en este puerto específico y en la dirección 0.0.0.0
ENV PORT=8080
EXPOSE $PORT

# El punto de entrada para iniciar la aplicación
CMD streamlit run src/app.py --server.port ${PORT} --server.address 0.0.0.0
