mpirun -np 2 \
    -hostfile /inspire/hdd/global_user/huxiaohe-p-huxiaohe/liuda/nccl-tests-master/aaaaa \
    --allow-run-as-root \
    -x NCCL_DEBUG=version \
    -x MASTER_PORT=29500 \
    -x UCX_NET_DEVICES=bond0 \
    -x NCCL_IB_QPS_PER_CONNECTION=8 \
    -x NCCL_IB_TC=186 \
    -x NCCL_IB_GID_INDEX=3 \
    ./build/sendrecv_perf -b 256MB -e 256MB -f 2 -g 1 -i 100