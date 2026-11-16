# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=/app/src
WORKDIR /app

COPY requirements.txt .
# Behind a TLS-inspecting proxy or antivirus, pass its root CA:
#   docker build --secret id=ca,src=path/to/root.pem .
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then cat /etc/ssl/certs/ca-certificates.crt /run/secrets/ca > /tmp/ca.pem; export PIP_CERT=/tmp/ca.pem; fi; \
    pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

COPY src src
COPY data data

ENTRYPOINT ["python", "-m", "btcforecast"]
CMD ["backtest"]
