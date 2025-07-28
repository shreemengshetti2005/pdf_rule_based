FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y build-essential libmupdf-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN pip install torch==2.7.1 --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY models/Gemma-1B.Q4_K_M.gguf /models/Gemma-1B.Q4_K_M.gguf

RUN python -c "import nltk; nltk.download('punkt')"

COPY . /app

ENTRYPOINT ["python", "main.py"]
