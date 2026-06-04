# Dockerfile for Hugging Face Spaces (Docker SDK).
# Pins Python 3.11 so chromadb's protobuf/opentelemetry stack stays compatible.
FROM python:3.11-slim

# HF Spaces run the container as a non-root user (UID 1000).
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    ANONYMIZED_TELEMETRY=False

WORKDIR /home/user/app

# System deps for chromadb / pdfplumber.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .
RUN chown -R user:user /home/user

USER user

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Bind 0.0.0.0 and relax CORS/XSRF so the app works behind HF's iframe proxy.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
