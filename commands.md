enroot create --name my_custom_pt /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/tomasruiz/my_custom_pt.sqsh

enroot start --rw --root \
    --mount /dss/dsshome1/0D/di38bec/code:/workspace/code \
    --mount $DSS_HOME:$DSS_HOME \
    my_custom_pt


export PYTHONUNBUFFERED=1

vllm serve Qwen/Qwen3-14B \
    --dtype auto \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --task generate \
    --disable-log-requests \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --port 8001