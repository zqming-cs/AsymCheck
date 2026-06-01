
OUT_DIR=${OUT_DIR:-"./log"}
epochs="${epochs:-30}"
density="${density:-0.1}"

compressor="${compressor:-topk}"


train_batch_size="${train_batch_size:-4}"
val_batch_size="${val_batch_size:-4}"


echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi



export Save_Checkpoint="./gpt2_checkpoint"

NGPU_PER_NODE=2
NUM_NODES=4

CUDA_VISIBLE_DEVICES=0,1
LR=${5:-0.00003}
SEED=${6:-12345}
MASTER_PORT=${7:-29500}
DROPOUT=${8:-0.1}
ckpt_mode=1
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

config_json=deepspeed_bsz24_z3_config.json


CMD=" deepspeed  \
      --hostfile hostfile_4.txt \
      run_clm_no_trainer_ds_lib_torch_save.py  \
      --job_name ${JOB_NAME} \
      --fp16 \
      --deepspeed \
      --deepspeed_config ${config_json} \
      "


CMD+=" --dataset_name wikitext-103-raw-v1 --dataset_config_name default "
CMD+=" --model_name_or_path gpt2-large "
CMD+=" --output_dir  ./gpt2_checkpoint/ "
CMD+=" --num_train_epochs=$epochs  "
CMD+=" --do_train "
CMD+=" --density=$density --compressor=$compressor --memory=$memory --percent=$percent "
CMD+=" --per_device_train_batch_size=$train_batch_size "
CMD+=" --per_device_eval_batch_size=$val_batch_size "


CMD+=" --bert_model  bert-base-uncased "
CMD+=" --arch gpt-large "



LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE



