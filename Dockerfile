# ── Dockerfile ────────────────────────────────────────────────
#
# WHY python:3.11-slim (not full python:3.11):
#   The full image is ~900MB. Slim is ~130MB.
#   Smaller images = faster CI builds, cheaper cloud storage,
#   smaller attack surface for security. Always use slim in prod.

FROM python:3.11-slim

WORKDIR /app

# WHY copy requirements first (before source code):
#   Docker caches each layer. If we copy source first, every code
#   change invalidates the pip install layer — slow rebuilds.
#   Copying requirements first means pip install is only re-run
#   when requirements.txt actually changes.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools==67.8.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and model artefacts
COPY src/ ./src/
COPY models/ ./models/

# WHY non-root user:
#   Running as root inside a container is a security risk.
#   Industry standard is to create a dedicated app user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Health check — Docker restarts container if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# WHY not --reload: dev only. In production files don't change at runtime.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
