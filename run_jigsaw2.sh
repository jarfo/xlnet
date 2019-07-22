#!/bin/bash

export TPU_NAME=xlnet
export JIGSAW_DIR=../input/jigsaw-unintended-bias-in-toxicity-classification
export GS_ROOT=gs://tpubert
export LARGE_DIR=models/xlnet_cased_L-24_H-1024_A-16

python run_jigsaw.py \
  --learning_rate=5e-6 \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --do_train=True \
  --do_eval=True \
  --eval_all_ckpt=True \
  --task_name=jigsaw \
  --data_dir=${JIGSAW_DIR} \
  --output_dir=${GS_ROOT}/xlnet1/proc_data/jigsaw \
  --model_dir=${GS_ROOT}/xlnet1/exp/jigsaw2 \
  --uncased=False \
  --spiece_model_file=${LARGE_DIR}/spiece.model \
  --model_config_path=${GS_ROOT}/${LARGE_DIR}/xlnet_config.json \
  --init_checkpoint=${GS_ROOT}/${LARGE_DIR}/xlnet_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=32 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --train_steps=200000 \
  --warmup_steps=25000 \
  --save_steps=5000 \
  --iterations=5000
