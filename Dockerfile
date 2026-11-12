# UAT-LITE: reproducible GPU environment (PyTorch 2.3.1, CUDA 12.1, Python 3.10)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/hf_cache

WORKDIR /workspace/uat-lite
COPY pyproject.toml requirements.txt README.md ./
COPY uat_lite ./uat_lite
RUN pip install -e . pytest

COPY scripts ./scripts
COPY tests ./tests
COPY Makefile ./

# checkpoints, data, results and the HF cache are mounted at run time
VOLUME ["/workspace/uat-lite/experiments", "/workspace/uat-lite/results", "/workspace/hf_cache"]
CMD ["bash"]
