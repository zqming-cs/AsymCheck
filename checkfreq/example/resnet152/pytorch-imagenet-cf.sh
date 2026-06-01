#!/bin/bash

MASTER_ADDR=0.0.0.0
MASTER_PORT=29600
WORLD_SIZE=8
export CUDA_VISIBLE_DEVICES=0,1
DATA_DIR="imagenet"
OUT_DIR="output"
WORKER=4
SCRIPTS="scripts"
chk_prefix="resnet"
recovery_interval=30
stop_iteration=6000
mkdir -p $OUT_DIR


gpu=4
num_gpu=2

echo " Data dir is $DATA_DIR"
echo " Out dir is $OUT_DIR"


for arch in 'resnet152' ; do
	for workers in $WORKER; do
		for batch in 64; do

    # RUN 1 : CheckFreq adaptive tune
			result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_dali_fp32_cf_recovery_distributed"
			echo "result dir is $result_dir" 
			mkdir -p $result_dir
			echo "Now running $arch for $workers workers and $batch batch"
			deepspeed  --master_port=${MASTER_PORT} --hostfile hostfile_4.txt pytorch-imagenet-cf.py --adaptive_tune -a $arch -b $batch --workers $workers --epochs 1 --stop_iteration $stop_iteration --deterministic --noeval --barrier --checkfreq --chk-prefix $chk_prefix --cf_iterator --data $DATA_DIR



		done
	done
done
