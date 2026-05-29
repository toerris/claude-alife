FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN make -C agent_host

ENV ALIFE_WEB_HOST=0.0.0.0 \
    ALIFE_WEB_PORT=8000 \
    ALIFE_AGENT_HOST=0.0.0.0 \
    ALIFE_AGENT_PORT=9000 \
    ALIFE_LOCAL_AGENT_HOST=127.0.0.1

EXPOSE 8000 9000
CMD ["bash", "./run.sh"]
