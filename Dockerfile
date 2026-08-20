FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies
ENV UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache --no-install-project

# Copy project files
COPY . .

# Set path to use uv's virtual environment
ENV PATH="/app/.venv/bin:$PATH"
# Ensure Python logs are sent straight to terminal
ENV PYTHONUNBUFFERED=1 

EXPOSE 8501

# The command to run Streamlit
CMD ["streamlit", "run", "ui.py", "--server.address", "0.0.0.0"]