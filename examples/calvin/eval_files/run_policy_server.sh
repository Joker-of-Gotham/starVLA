#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export star_vla_python=${star_vla_python:-$(pwd)/env/starvla-py310/bin/python}
your_ckpt=${your_ckpt:-results/Checkpoints/0118_starvla_qwenpi_calvin_task_ABC_D/checkpoints/steps_30000_pytorch_model.pt}
gpu_id=${gpu_id:-0}
port=${port:-5694}
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
