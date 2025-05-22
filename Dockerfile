FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create cache dirs and set safe env vars
RUN mkdir -p /tmp/.cache/huggingface && \
    mkdir -p /tmp/.streamlit && \
    chmod -R 777 /tmp/.cache /tmp/.streamlit

ENV TRANSFORMERS_CACHE=/tmp/.cache
ENV HF_HOME=/tmp/.cache/huggingface
ENV XDG_CACHE_HOME=/tmp/.cache
ENV STREAMLIT_HOME=/tmp/.streamlit
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

EXPOSE 7860

# Add inside Dockerfile before CMD
RUN mkdir -p /app/.streamlit && \
    echo "\
[server]\n\
headless = true\n\
port = 7860\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
" > /app/.streamlit/config.toml

CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
