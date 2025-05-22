FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV TRANSFORMERS_CACHE=/tmp/.cache
ENV HF_HOME=/tmp/.cache/huggingface
ENV XDG_CACHE_HOME=/tmp/.cache
ENV STREAMLIT_HOME=/tmp/.streamlit

EXPOSE 7860

# Set safe cache directories
ENV TRANSFORMERS_CACHE=/home/user/.cache
ENV HF_HOME=/home/user/.cache/huggingface
ENV XDG_CACHE_HOME=/home/user/.cache
ENV STREAMLIT_HOME=/home/user/.streamlit
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
