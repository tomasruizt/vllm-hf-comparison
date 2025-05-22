enroot start --rw --root \
    --mount /dss/dsshome1/0D/di38bec/code:/workspace/code \
    --mount $DSS_HOME:$DSS_HOME \
    my_custom_pt


export PYTHONUNBUFFERED=1

vllm serve Qwen/Qwen3-0.6B \
    --dtype auto \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --task generate \
    --disable-log-requests \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.5 \
    --port 8001