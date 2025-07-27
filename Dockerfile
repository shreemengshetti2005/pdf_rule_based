FROM python:3.10-slim

# Install system dependencies for PyMuPDF (fitz) and wget (optional)
RUN apt-get update && \
    apt-get install -y build-essential libmupdf-dev && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt ./

# Install Python dependencies (no torch!)
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Copy pre-downloaded Gemma GGUF model into container
COPY models/Gemma-1B.Q4_K_M.gguf /models/Gemma-1B.Q4_K_M.gguf

# ✅ Ensure NLTK tokenizer is available
RUN python -c "import nltk; nltk.download('punkt')"

# Copy all project files
COPY . /app

# Entrypoint
ENTRYPOINT ["python", "main.py"]
