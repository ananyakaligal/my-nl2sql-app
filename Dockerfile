FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 7860

# Set safe cache directories
ENV TRANSFORMERS_CACHE=/home/user/.cache
ENV HF_HOME=/home/user/.cache/huggingface
ENV XDG_CACHE_HOME=/home/user/.cache
ENV STREAMLIT_HOME=/home/user/.streamlit
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
