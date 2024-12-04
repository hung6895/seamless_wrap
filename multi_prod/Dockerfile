# Use a PyTorch runtime with CUDA
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    sox \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install specific versions of additional packages
RUN pip install transformers==4.35.0 sentencepiece

# Copy application code
COPY . /app

# Generate gRPC code
#RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. seamless_m4t.proto
RUN pip install git+https://github.com/huggingface/transformers.git
# Debug the transformers package and download the model
#RUN python -c "import transformers; print(transformers.__version__); print(dir(transformers))"
#RUN python -c "from transformers import AutoProcessor, AutoModel; \
#    model_name = 'facebook/seamless-m4t-v2-large'; \
#    AutoProcessor.from_pretrained(model_name); \
#    AutoModel.from_pretrained(model_name)"

# Expose the port
EXPOSE 8080

# Command to run
CMD ["python", "server51.py"]


