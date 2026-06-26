FROM docker.n8n.io/n8nio/n8n:latest

USER root

RUN apk add --no-cache python3 py3-pip

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir pandas numpy scikit-learn joblib

USER node