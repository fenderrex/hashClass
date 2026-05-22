FROM python:3.10-slim-bullseye

# Install system deps for SDL/OpenGL and X virtual framebuffer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    xvfb \
    x11-utils \
    libgl1-mesa-glx \
    libglu1-mesa \
    freeglut3 \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
    libportmidi0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install python deps
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . /app

# Ensure entrypoint is executable
RUN chmod +x /app/entrypoint.sh

ENV DISPLAY=:99

ENTRYPOINT ["/app/entrypoint.sh"]
