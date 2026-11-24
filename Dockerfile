FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libx11-dev \
    unzip \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY lib/nbis_build /opt/nbis_build
ENV PATH="/opt/nbis_build/bin:$PATH"

# Copiar app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 5000
CMD ["python", "afis_web_app_bozorth3.py"]
