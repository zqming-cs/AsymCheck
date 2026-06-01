

OUT_DIR=${OUT_DIR:-"./log"}
num_train_epochs="${num_train_epochs:-80}"
density="${density:-0.1}"

compressor="${compressor:-topk}"



per_device_train_batch_size="${per_device_train_batch_size:-2}"
per_device_eval_batch_size="${per_device_eval_batch_size:-8}"


# imagenet
train_dir=${train_dir:-"train"}
validation_dir=${validation_dir:-"val"}

# ViT-base
model_name_or_path=${model_name_or_path:-"vit-large-patch16-224-in21k"}

metric_accuracy=${metric_accuracy:-"accuracy"}




echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

MASTER_PORT=29500
chk_prefix="vit"



CMD=" deepspeed  --master_port=${MASTER_PORT} --hostfile hostfile_4.txt run_imagenet_no_trainer_cf.py --image_column_name image   "



CMD+=" --image_column_name image  "
CMD+=" --num_train_epochs=$num_train_epochs  "

CMD+=" --train_dir $train_dir  "
CMD+=" --validation_dir $validation_dir  "
CMD+=" --model_name_or_path $model_name_or_path  "
CMD+=" --per_device_train_batch_size $per_device_train_batch_size  "
CMD+=" --per_device_eval_batch_size $per_device_eval_batch_size  "
CMD+=" --metric_accuracy $metric_accuracy  "
CMD+=" -a vit-large --checkfreq --chk-prefix $chk_prefix --cf_iterator --adaptive_tune"

LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE






