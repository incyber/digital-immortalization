# Piper sidecar. Everything GPL-licensed in this project lives inside this
# image and nowhere else; the application talks to it over HTTP.
FROM python:3.12-slim

RUN pip install --no-cache-dir "piper-tts>=1.7,<2"

WORKDIR /app
COPY server.py /app/server.py

ENV PIPER_VOICES_DIR=/voices \
    PIPER_PORT=5050

EXPOSE 5050
CMD ["python", "-u", "/app/server.py"]
