FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Install CUDA toolkit components needed by TensorFlow for PTX compilation.
# Without these, TF falls back to JIT compilation through the NVIDIA driver, adding latency to GPU operations.
RUN apt-get update && apt-get install -y \
    cuda-nvcc-12-8 \
    cuda-cudart-dev-12-8 \
    libnvvm4 \
    && rm -rf /var/lib/apt/lists/*

# Tell TensorFlow's XLA backend where to find CUDA toolkit components (libdevice, ptxas).
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install all Python dependencies from a fully pinned requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages \
    --timeout 120 \
    --retries 5 \
    -r requirements.txt

COPY . .

# Pre-bake the RetinaFace model weights into the image to avoid runtime downloads during cold-start.
RUN python3 -c "from retinaface import RetinaFace; RetinaFace.build_model()" || true

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]