FROM python:slim
COPY requirements.txt requirements_dev.txt ./
RUN apt-get update && \
    apt-get install build-essential procps -y && \
    rm -rf /var/lib/apt/lists/* && \
    pip install -r requirements.txt && \
    pip install -r requirements_dev.txt
