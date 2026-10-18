# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /usr/local/bin/uv

ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./

# Vector search is optional; the default image needs no embedding model.
ARG INSTALL_VECTOR=false
RUN --mount=type=cache,target=/root/.cache/uv \
    case "$INSTALL_VECTOR" in \
      false) uv sync --locked --no-dev ;; \
      true) uv sync --locked --no-dev --extra vector ;; \
      *) echo "INSTALL_VECTOR must be true or false" >&2; exit 1 ;; \
    esac

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/state/huggingface \
    FEEDBACK_PATH=/app/state/feedback.jsonl \
    CHROMA_DIR=/app/state/chroma

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /app/state \
    && chown app:app /app/state

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY styles/ ./styles/
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY data/*.json ./data/
COPY scripts/build_index.py ./scripts/build_index.py
COPY design.svg ./design.svg

USER app
EXPOSE 8501 8000

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
