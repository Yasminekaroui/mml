#!/bin/bash
PROJECT_DIR=/home/karoui/marvl_project

TASK=12
LANG=zh
MODEL=xuniter
MODEL_CONFIG=ctrl_xuniter_base
TRTASK=NLVR2
TETASK=MaRVLzh
TASKS_CONFIG=xling_test_marvl
TEXT_PATH=${PROJECT_DIR}/marvl-code/data/${LANG}/annotations/marvl-${LANG}.jsonl
FEAT_PATH=${PROJECT_DIR}/datasets/marvl/features/marvl-${LANG}_boxes36.lmdb
PRETRAINED=${PROJECT_DIR}/pretrained_models/${MODEL}/pytorch_model_9.bin
OUTPUT_DIR=${PROJECT_DIR}/results/${MODEL}/${TRTASK}_${MODEL_CONFIG}/$TETASK/test

source ~/miniconda3/etc/profile.d/conda.sh

conda activate marvl

cd ../../../volta
python eval_task.py \
        --bert_model xlm-roberta-base \
        --config_file config/${MODEL_CONFIG}.json \
        --from_pretrained ${PRETRAINED} \
        --val_annotations_jsonpath ${TEXT_PATH} --val_features_lmdbpath ${FEAT_PATH} \
        --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --split test \
        --output_dir ${OUTPUT_DIR}

conda deactivate
