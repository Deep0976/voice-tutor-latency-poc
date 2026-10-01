# Container image for hosting the voice tutor (e.g. on Hugging Face Spaces).
# Hugging Face Spaces expect the app to listen on port 7860.
FROM python:3.12-slim

# ffmpeg helps with audio handling; the rest are common build needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face runs the container as a non-root user (uid 1000).
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME=/home/user/.cache \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# Install Python deps first (better build caching).
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the app (the .dockerignore keeps secrets/venv/samples out).
COPY --chown=user . .

# Serve on 0.0.0.0:7860 using Daily's cloud transport (works for friends anywhere).
# API keys come from environment variables / Space secrets — never baked into the image.
EXPOSE 7860
CMD ["python", "bot.py", "--host", "0.0.0.0", "--port", "7860", "--transport", "daily"]
