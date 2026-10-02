# syntax=docker/dockerfile:1.7
# ---- build stage: install dependencies into a virtualenv ----
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY coach ./coach
COPY evals ./evals
COPY experiments ./experiments
COPY loadtest ./loadtest
ARG EXTRAS="live,otel"
RUN pip install ".[${EXTRAS}]"

# ---- runtime stage: small image, non-root user ----
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/venv/bin:$PATH" \
    PORT=8080 DB_PATH=/tmp/coach.db
RUN useradd --create-home --uid 10001 app
COPY --from=build /venv /venv
WORKDIR /app
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"
# Cloud Run sets $PORT. One worker per container; Cloud Run scales containers, not workers.
CMD ["sh", "-c", "exec uvicorn --factory coach.app:create_app --host 0.0.0.0 --port ${PORT} --ws websockets --timeout-graceful-shutdown 20"]
