

OUT_DIR=${OUT_DIR:-"./log"}
num_train_epochs="${num_train_epochs:-80000}"
density="${density:-0.01}"
compressor="${compressor:-topkef}"

memory="${memory:-residual}"


per_device_train_batch_size="${per_device_train_batch_size:-32}"
per_device_eval_batch_size="${per_device_eval_batch_size:-32}"




train_dir=${train_dir:-"train"}
validation_dir=${validation_dir:-"val"}



# ViT-large
model_name_or_path=${model_name_or_path:-"vit-large-patch16-224-in21k"}

metric_accuracy=${metric_accuracy:-"accuracy"}

output_dir=${output_dir:-"./output"}


echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi


export save_checkpoint_path="./vit_checkpoint"



NGPU_PER_NODE=2
NUM_NODES=4

CUDA_VISIBLE_DEVICES=0,1
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
config_json=ds_fp16_z3_config.json
# 


CMD=" deepspeed   \
      --hostfile hostfile_4.txt \
      run_imagenet_no_trainer_ds_lib_asymcheck.py   \
      --job_name ${JOB_NAME} \
      --fp16 \
      --deepspeed \
      --deepspeed_config ${config_json} \
      "

CMD+=" --image_column_name image  "
CMD+=" --num_train_epochs=$num_train_epochs  "
CMD+=" --train_dir $train_dir  "
CMD+=" --do_train  "
CMD+=" --validation_dir $validation_dir  "
CMD+=" --model_name_or_path $model_name_or_path  "
CMD+=" --per_device_train_batch_size $per_device_train_batch_size  "
CMD+=" --per_device_eval_batch_size $per_device_eval_batch_size  "
CMD+=" --metric_accuracy $metric_accuracy  --with_tracking "
CMD+=" --density=$density --compressor=$compressor --memory=$memory --percent=$percent "
CMD+=" --output_dir $output_dir  "
CMD+=" --arch vit-large  "


LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE






