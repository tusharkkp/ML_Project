# Production Dockerfile for Hugging Face Spaces (Streamlit SDK via Docker)
# Hugging Face Spaces exposes port 7860 by default on the Free CPU Basic tier.
FROM python:3.12-slim

WORKDIR /app

# Set environment variables for Python & Streamlit
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code, models, and reports
COPY . .

# Hugging Face default port
EXPOSE 7860

# Launch Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
