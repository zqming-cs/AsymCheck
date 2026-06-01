

OUT_DIR=${OUT_DIR:-"../log"}
epochs="${epochs:-30}"


train_batch_size="${train_batch_size:-2}"
val_batch_size="${val_batch_size:-2}"



echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

export Save_Checkpoint="../checkpoint"

NGPU_PER_NODE=2
NUM_NODES=4

CUDA_VISIBLE_DEVICES=0,1,2,3
LR=${5:-0.00003}
SEED=${6:-12345}
MASTER_PORT=${7:-29500}
DROPOUT=${8:-0.1}
echo "lr is ${LR}"
echo "seed is $SEED"
echo "master port is $MASTER_PORT"
echo "dropout is ${DROPOUT}"


HOSTFILE=hostfile_4.txt


NGPU=$((NGPU_PER_NODE*NUM_NODES))
EFFECTIVE_BATCH_SIZE=24
MAX_GPU_BATCH_SIZE=3
PER_GPU_BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE/NGPU))
if [[ $PER_GPU_BATCH_SIZE -lt $MAX_GPU_BATCH_SIZE ]]; then
       GRAD_ACCUM_STEPS=1
else
       GRAD_ACCUM_STEPS=$((PER_GPU_BATCH_SIZE/MAX_GPU_BATCH_SIZE))
fi
JOB_NAME="deepspeed_${NGPU}GPUs_${EFFECTIVE_BATCH_SIZE}batch_size"


config_json=../deepspeed_bsz4_z3_config.json



CMD=" deepspeed --num_nodes ${NUM_NODES} --num_gpus ${NGPU_PER_NODE} \
      --master_port ${MASTER_PORT} \
      --hostfile ${HOSTFILE} \
      ../run_clm_no_trainer_bloom_ds_asym.py   \
      --job_name ${JOB_NAME} \
      --fp16 \
      --deepspeed \
      --deepspeed_config ${config_json} \
      "

CMD+=" --model_name_or_path bloom "
CMD+=" --dataset_name wikitext "
CMD+=" --dataset_config_name default"

CMD+=" --do_train   --block_size 512  "
CMD+=" --num_train_epochs=$epochs  "
CMD+=" --weight_decay 0.01   --learning_rate 5e-5"
CMD+=" --per_device_train_batch_size=$train_batch_size "
CMD+=" --per_device_eval_batch_size=$val_batch_size "

CMD+=" --output_dir   ../run_clm_no_trainer_bloom_output "



LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE





