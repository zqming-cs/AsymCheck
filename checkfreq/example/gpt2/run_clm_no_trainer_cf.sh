
OUT_DIR=${OUT_DIR:-"./log"}
epochs="${epochs:-30}"
density="${density:-0.1}"

compressor="${compressor:-topk}"


train_batch_size="${train_batch_size:-3}"
val_batch_size="${val_batch_size:-4}"


echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi



export Save_Checkpoint="gpt"
chk_prefix="gpt"
MASTER_PORT=29500



CMD=" deepspeed  --master_port=${MASTER_PORT} --hostfile hostfile_4.txt  run_clm_no_trainer_cf.py"




CMD+=" --dataset_name wikitext-103-raw-v1 --dataset_config_name default"
CMD+=" --model_name_or_path gpt2-large "
CMD+=" --output_dir  ./gpt2_checkpoint/ "
CMD+=" --num_train_epochs=$epochs  "
CMD+=" --do_train "
CMD+=" --density=$density --compressor=$compressor --memory=$memory --percent=$percent "
CMD+=" --per_device_train_batch_size=$train_batch_size "
CMD+=" --per_device_eval_batch_size=$val_batch_size "
CMD+=" --bert_model  bert-base-uncased "
CMD+=" -a gpt-medium --checkfreq --chk-prefix $chk_prefix --cf_iterator "
CMD+=" --adaptive_tune"








LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE



