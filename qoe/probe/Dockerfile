FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY probe.py /app/probe.py
USER 10001:10001
EXPOSE 9194
CMD ["python", "/app/probe.py"]
