# Container image for self-hosting the dashboard.
#
# Build and run from the ipo-capital-engine directory:
#     docker build -t ipo-capital-engine .
#     docker run --rm -p 8501:8501 ipo-capital-engine
# then open http://localhost:8501

FROM python:3.12-slim

# Fail fast and keep the image free of .pyc clutter and pip's cache.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so that editing application code does not invalidate the
# (slow) dependency layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as an unprivileged user: nothing in the app needs root.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Streamlit's own readiness endpoint, checked without adding curl to the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
