FROM nvcr.io/nvidia/tritonserver:25.02-vllm-python-py3

WORKDIR /app

RUN pip install --no-cache-dir nest_asyncio numpy tritonclient runpod fastapi uvicorn tritonclient[all] requests numpy trism

COPY rp_handler_1.py /app/
COPY run.sh /app/
COPY --chown=1000:1000 triton_server /models

###############################
# FOR TESTING LOCAL ONLY
# COPY test_input.json /app/
###############################

ENV VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_LOGGING_LEVEL=DEBUG \
    VLLM_TRACE_FUNCTION=1 \
    NCCL_DEBUG=TRACE \
    VLLM_RPC_TIMEOUT=100000 \
    VLLM_ENGINE_HC_TIMEOUT=100000 \
    VLLM_GPU_MEMORY_UTILIZATION=0.8

EXPOSE 8000 8001 8002 8080

CMD ["bash", "run.sh"]
