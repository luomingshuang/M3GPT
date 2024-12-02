#!/usr/bin/env bash
set -x

# source /mnt/lustre/share/spring/r0.3.0
ROOT=../../../
export PYTHONPATH=$ROOT:$PYTHONPATH

if [[ ! -d "logs" ]]; then
  mkdir logs
fi

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

################
gpus=${1-48}
job_name=${2-debug}
################
CONFIG=${3-${job_name}.yaml}

g=$((${gpus}<8?${gpus}:8))
echo 'start job:' ${job_name} ' config:' ${CONFIG}

AutoResume=checkpoints/${job_name}/ckpt_task_iter_newest.pth.tar

now=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=logs/${job_name}_${now}.log
echo 'log file: ' ${LOG_FILE}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# the available ngpus == ${gpus}
GLOG_vmodule=MemcachedClient=-1 MKL_SERVICE_FORCE_INTEL=1 \

# bash train_motiongpt_lm_no_slurm_motion_dance.sh 2 motiongpt_lm_motion_dance

torchrun --nnodes 1 --nproc_per_node ${gpus} --master_port 5689 \
${ROOT}/multitask.py \
    --expname ${job_name} \
    --config ${CONFIG} \
    --auto-resume='checkpoints/motiongpt_lm_only_text2motion/ckpt_task0_iter_310000.pth.tar' \
    --gpus ${gpus} \
    --method "no_slurm" \
    2>&1 | tee ${LOG_FILE}
