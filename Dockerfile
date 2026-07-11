FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml uv.lock ./
COPY vessel ./vessel

RUN uv sync --frozen --no-dev

EXPOSE 8080

ENV PORT=8080 HOST=0.0.0.0

CMD ["uv", "run", "python", "-m", "vessel.main"]
