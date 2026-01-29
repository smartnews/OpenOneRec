#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="OpenOneRec/OneRec-8B"
OUTPUT_DIR="$ROOT_DIR/output/SN-8B-pretrain"
DATASET_CONFIG="test/pretrain/pretrain.json"

export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export PYTORCH_ALLOC_CONF=expandable_segments:True  # Reduce CUDA memory fragmentation

mkdir -p "$OUTPUT_DIR"
mkdir -p /tmp/_wids_cache

# Resolve HF cache snapshot if a repo id is provided and local path is missing.
if [ ! -d "$MODEL_DIR" ] && [[ "$MODEL_DIR" != /* ]] && [[ "$MODEL_DIR" == */* ]]; then
    HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
    HF_REPO_DIR="$HF_HOME_DIR/hub/models--${MODEL_DIR//\//--}/snapshots"
    if [ -d "$HF_REPO_DIR" ]; then
        LATEST_SNAPSHOT="$(ls -t "$HF_REPO_DIR" | head -n 1)"
        if [ -n "$LATEST_SNAPSHOT" ]; then
            MODEL_DIR="$HF_REPO_DIR/$LATEST_SNAPSHOT"
        fi
    fi
fi

HOSTFILE_SEQ="$OUTPUT_DIR/hostfile_seq"
if [ -f /etc/mpi/hostfile ]; then
    sed 's/=1/=8/g' /etc/mpi/hostfile > "$HOSTFILE_SEQ"
else
    echo "localhost slots=1" > "$HOSTFILE_SEQ"
fi

nnode=$(wc -l < "$HOSTFILE_SEQ")

set -x

echo "Output: $OUTPUT_DIR"

export PYTHONPATH=$PWD/pretrain:$PWD:$PYTHONPATH

ENV_FILE="pretrain/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi
if [ -n "$LD_PRELOAD" ] && [ ! -f "$LD_PRELOAD" ]; then
    unset LD_PRELOAD
fi
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
fi

hostfile="$HOSTFILE_SEQ"
if command -v ifconfig >/dev/null 2>&1; then
    TCP_NIC=$(ifconfig | grep -B1 " "$(hostname -i)" " | grep -o "^\w*")
else
    TCP_NIC=$(ip route get 1.1.1.1 | awk '{for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}')
fi
if [ -z "$TCP_NIC" ]; then
    TCP_NIC="lo"
fi

if [ -n "$MY_NODE_IP" ]; then
    MASTER_ADDR="$MY_NODE_IP"
elif command -v hostname >/dev/null 2>&1; then
    MASTER_ADDR="$(hostname -i | awk '{print $1}')"
else
    MASTER_ADDR="127.0.0.1"
fi
MASTER_PORT=8499
export MASTER_ADDR MASTER_PORT

mpirun --allow-run-as-root \
    -hostfile $hostfile \
    -mca btl self,tcp -mca pml ob1 \
    -mca plm_rsh_num_concurrent 600 \
    -mca routed_radix 600 \
    -mca btl_tcp_if_include $TCP_NIC \
    -mca oob_tcp_if_include $TCP_NIC \
    -mca btl_openib_allow_ib false \
    -x OMPI_MCA_btl=self,tcp \
    -x OMPI_MCA_pml=ob1 \
    -x OMPI_MCA_btl_tcp_if_include=$TCP_NIC \
    -x OMPI_MCA_oob_tcp_if_include=$TCP_NIC \
    -x OMPI_MCA_btl_openib_allow_ib=false \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_IB_GID_INDEX=3 \
    -x NCCL_SOCKET_IFNAME=$TCP_NIC \
    -x NCCL_IB_HCA=mlx5 \
    -x NCCL_DEBUG=WARN \
    -x NCCL_IB_QPS_PER_CONNECTION=4 \
    -x NCCL_NET_OVERHEAD=1000 \
    -x NCCL_IB_TIMEOUT=20 \
    -x LD_PRELOAD=$LD_PRELOAD \
    -x http_proxy="" \
    -x https_proxy="" \
    -x HOROVOD_MPI_THREADS_DISABLE=1 \
    -x MPI_THREAD_SINGLE=1 \
    -x NO_COLOR=1 \
    -x TERM=dumb \
    -x COLORTERM=0 \
    -x PYTHONIOENCODING=utf-8 \
    -x LD_LIBRARY_PATH=$LIBRARY_PATH \
    -x PATH \
    -x PYTHONPATH=$PYTHONPATH \
    -x JAVA_HOME=$JAVA_HOME \
    -x HIVE_HOME=$HIVE_HOME \
    -x CLASSPATH=$CLASSPATH \
    -x HADOOP_USER_NAME=$HADOOP_USER_NAME \
    -x HADOOP_HOME=$HADOOP_HOME \
    -x SPARK_HOME=$SPARK_HOME \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -x TOKENIZERS_PARALLELISM=false \
    bash -c "bash pretrain/scripts/numa_runner.sh python3 pretrain/recipes/train_qwen3.py \
        --model_dir $MODEL_DIR \
        --output_dir $OUTPUT_DIR \
        --dataset_config $DATASET_CONFIG \
        --use_tie_weights \
        --model_class Qwen3ForCausalLM \
        --monitor_datasource_loss \
        --monitor_datasource_cnt \
        # --max_length 32768 \
        --max_length 256 \
        --learning_rate 2e-4 \
        --min_lr 1e-4 \
        --weight_decay 0.1 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 500 \
        --num_training_steps 5000 \
        --save_checkpoint_per_step 50 \
        # --minibatch_size 16384 \
        --minibatch_size 1 \
        --logging_per_step 5 \
        --seed 19260817 \
        # --enable_profiler \
        # --use_fp32_weight \
        --enable_gradient_checkpointing \
        --use_chunked_loss_computer \
    "
