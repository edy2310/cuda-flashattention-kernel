FROM nvcr.io/nvidia/tritonserver:24.02-py3

WORKDIR /workspace

# Install build tools and PyTorch with CUDA support.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
        torch==2.2.2 \
        --index-url https://download.pytorch.org/whl/cu121

# Copy the full repo so the model can compile the extension sources.
COPY . /workspace

EXPOSE 8000 8001 8002

CMD ["tritonserver", "--model-repository=/workspace/triton_deploy/model_repository"]
