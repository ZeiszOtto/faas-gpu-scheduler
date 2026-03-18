FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --break-system-packages -r requirements.txt && \
    pip install --no-cache-dir --break-system-packages torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

COPY . .

RUN python3 -c "from retinaface import RetinaFace; RetinaFace.build_model()" || true

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]