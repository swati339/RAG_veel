FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install curl and uv
RUN apt-get update && apt-get install -y curl build-essential \
    && curl -Ls https://astral.sh/uv/install.sh | bash \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only the files needed to install dependencies first
COPY pyproject.toml .

# Install dependencies using uv
RUN uv pip install --system --no-cache

# Copy rest of the application code
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI app with uvicorn
CMD ["uvicorn", "rag_chat.main:app", "--host", "0.0.0.0", "--port", "8000"]
