FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system --gid 10001 app && adduser --system --uid 10001 --ingroup app app

COPY --chown=app:app requirements-local-demo.txt /app/requirements-local-demo.txt
RUN python -m pip install --no-cache-dir -r /app/requirements-local-demo.txt

COPY --chown=app:app . /app

USER 10001:10001

EXPOSE 8000

CMD ["python", "local_chaos_metrics_server.py", "--host", "0.0.0.0", "--port", "8000"]
