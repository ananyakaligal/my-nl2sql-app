# Use a slim Python runtime
FROM python:3.10-slim

# 1) Set working dir
WORKDIR /app

# 2) Copy and install your Python dependencies (including pytest)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy in your application code + tests
COPY . .

# 4) Create and chmod cache + .streamlit dirs for Hugging Face / Streamlit
RUN mkdir -p /tmp/.cache/huggingface /tmp/.streamlit && \
    chmod -R 777 /tmp/.cache /tmp/.streamlit

# 5) Set environment variables so HF / Streamlit use those safe paths
ENV TRANSFORMERS_CACHE=/tmp/.cache
ENV HF_HOME=/tmp/.cache/huggingface
ENV XDG_CACHE_HOME=/tmp/.cache
ENV STREAMLIT_HOME=/tmp/.streamlit
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# 6) Streamlit config to disable CORS, XSRF, stats, run headless
RUN mkdir -p /app/.streamlit && \
    printf "\
[server]\n\
headless = true\n\
port = 7860\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
" > /app/.streamlit/config.toml

# 7) Expose Streamlit port
EXPOSE 7860

# 8) Default command
CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
