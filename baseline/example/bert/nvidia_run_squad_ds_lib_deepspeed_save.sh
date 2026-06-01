#~/bin/bash



NGPU_PER_NODE=2

MODEL_FILE="bert-large-uncased-whole-word-masking-pytorch_model.bin"
SQUAD_DIR="squad"
OUTPUT_DIR="./output_deepspped"


CUDA_VISIBLE_DEVICES=0,1
LR=${5:-0.00003}
SEED=${6:-12345}
MASTER_PORT=${7:-10086}
DROPOUT=${8:-0.1}
echo "lr is ${LR}"
echo "seed is $SEED"
echo "master port is $MASTER_PORT"
echo "dropout is ${DROPOUT}"


NUM_NODES=4




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

run_cmd="deepspeed  \
       --master_port=${MASTER_PORT} \
       --hostfile hostfile_4.txt \
       nvidia_run_squad_ds_lib_deepspeed_save.py \
       --bert_model bert-large-uncased \
       --do_train \
       --do_lower_case \
       --predict_batch_size 3 \
       --do_predict \
       --train_file $SQUAD_DIR/train-v1.1.json \
       --predict_file $SQUAD_DIR/dev-v1.1.json \
       --train_batch_size $PER_GPU_BATCH_SIZE \
       --learning_rate ${LR} \
       --num_train_epochs 20.0 \
       --max_seq_length 384 \
       --doc_stride 128 \
       --output_dir $OUTPUT_DIR \
       --job_name ${JOB_NAME} \
       --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
       --fp16 \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --dropout ${DROPOUT} \
       --model_file $MODEL_FILE \
       --seed ${SEED} \
       --ckpt_type HF \
       --origin_bert_config_file bert-large-uncased-whole-word-masking-config.json
       "
echo ${run_cmd}
eval ${run_cmd}











