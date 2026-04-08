# Agent Memory System Docker Image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package
RUN pip install --no-cache-dir -e ".[all]"

# Create data directory
RUN mkdir -p /data

# Environment variables
ENV AGENT_MEMORY_SQLITE_PATH=/data/memory.db
ENV AGENT_MEMORY_CHROMA_PATH=/data/chroma

# Expose MCP server port (if running HTTP mode)
EXPOSE 3333

# Default command: run CLI
ENTRYPOINT ["mem"]
CMD ["--help"]
