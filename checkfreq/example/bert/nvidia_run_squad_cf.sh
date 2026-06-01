#~/bin/bash


NGPU_PER_NODE=2

MODEL_FILE="bert-large-uncased-whole-word-masking-pytorch_model.bin"
SQUAD_DIR="squad"
OUTPUT_DIR="./output"
NUM_NODES=4
NGPU=$((NGPU_PER_NODE*NUM_NODES))
EFFECTIVE_BATCH_SIZE=24
MAX_GPU_BATCH_SIZE=6

if [[ $PER_GPU_BATCH_SIZE -lt $MAX_GPU_BATCH_SIZE ]]; then
       GRAD_ACCUM_STEPS=1
else
       GRAD_ACCUM_STEPS=$((PER_GPU_BATCH_SIZE/MAX_GPU_BATCH_SIZE))
fi
PER_GPU_BATCH_SIZE=30
LR=3e-5
MASTER_PORT=29600
JOB_NAME="baseline_${NGPU}GPUs_${EFFECTIVE_BATCH_SIZE}batch_size"
arch=bert-large-uncased
chk_prefix="bert"
deepspeed  --master_port=${MASTER_PORT} --hostfile hostfile_4.txt \
	nvidia_run_squad_cf.py \
	--bert_model bert-large-uncased \
	-a bert-large-uncased\
	--do_train \
	--do_lower_case \
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
	--model_file $MODEL_FILE\
	--gradient_accumulation_steps 1\
	--checkfreq --chk-prefix $chk_prefix --cf_iterator --adaptive_tune

