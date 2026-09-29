# ==========================================
# Stage 1: Build Frontend
# ==========================================
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# ==========================================
# Stage 2: Final Backend & Application Image
# ==========================================
FROM python:3.11-slim

# Install C/C++ compilers and build tools required for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory to project root
WORKDIR /app

# 1. Copy full project root context into container
COPY . .

# 2. Copy compiled static files into backend/dist
COPY --from=frontend-builder /app/frontend/dist ./backend/dist

# 3. Download model from project root
RUN uv run scripts/download_model.py

# 4. Install requirements.txt from project root
RUN uv pip install --system -r requirements.txt

# 5. Execute main.py inside backend/ directory
WORKDIR /app/backend
CMD ["uv", "run", "main.py"]

