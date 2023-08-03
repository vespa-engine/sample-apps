#!/usr/bin/env bash
set -e

BASE_MODEL='intfloat/e5-small-v2'

OUTPUT_DIR="experiments/$1"
NEGATIVES_DIR="${OUTPUT_DIR}/negatives/"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/"
MODEL_DIR="${OUTPUT_DIR}/models/"

echo 'Ensuring directories exist'
mkdir -p "${NEGATIVES_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${MODEL_DIR}"
mkdir -p application-package/model/

echo 'Exporting and deploying base model'
python3 scripts/export_hf_model_from_hf.py \
      --hf_model "${BASE_MODEL}" \
      --output_dir application-package/model/
vespa deploy --wait 1800 application-package/

echo 'Feeding data to base model'
vespa feed --progress 10 "${DATA_DIR}/feed.jsonl"

echo 'Generating hard negatives'
python3 scripts/e5-hard-negatives.py \
      --endpoint "${VESPA_ENDPOINT}" \
      --certificate "${VESPA_CERTIFICATE}" \
      --key "${VESPA_KEY}" \
      --ranking summer-intern-special-cool-ranking-profile \
      --hits 100 \
      --queries "$2" \
      --qrels "$3" \
      --output_file "${NEGATIVES_DIR}/train.jsonl"

python3 scripts/e5-hard-negatives.py \
      --endpoint "${VESPA_ENDPOINT}" \
      --certificate "${VESPA_CERTIFICATE}" \
      --key "${VESPA_KEY}" \
      --ranking summer-intern-special-cool-ranking-profile \
      --hits 100 \
      --queries "$4" \
      --qrels "$5" \
      --output_file "${NEGATIVES_DIR}/dev.jsonl"

echo 'Training model'
deepspeed unilm/simlm/src/train_biencoder.py --deepspeed unilm/simlm/ds_config.json \
    --model_name_or_path "${BASE_MODEL}" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --add_pooler False \
    --t 0.02 \
    --seed 1234 \
    --do_train \
    --fp16 \
    --train_file "${NEGATIVES_DIR}/train.jsonl" \
    --validation_file "${NEGATIVES_DIR}/dev.jsonl" \
    --q_max_len 32 \
    --p_max_len 144 \
    --train_n_passages 2 \
    --dataloader_num_workers 1 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --use_scaled_loss True \
    --warmup_steps 1000 \
    --share_encoder True \
    --logging_steps 50 \
    --output_dir "${CHECKPOINT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --save_total_limit 2 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --disable_tqdm True \
    --report_to none

echo 'Exporting model to .onnx'
python3 scripts/export_hf_model_from_hf.py --hf_model "${CHECKPOINT_DIR}" --output_dir "${MODEL_DIR}"

echo 'Moving model to the application package'
mv "${MODEL_DIR}/model.onnx" "${MODEL_DIR}/tokenizer.json" application-package/model/

echo 'Deploying finetuned model'
vespa deploy --wait 1800 application-package/

echo 'Feeding data to finetuned model'
vespa feed --progress 10 "${DATA_DIR}/feed.jsonl"

echo 'Evaluating'
python3 scripts/evaluate.py \
                  --endpoint "${VESPA_ENDPOINT}" \
                  --certificate "${VESPA_CERTIFICATE}" \
                  --key "${VESPA_KEY}" \
                  --ranking summer-intern-special-cool-ranking-profile \
                  --queries "${DATA_DIR}/test-queries"
trec_eval -mndcg_cut.10 "${DATA_DIR}/test-qrels" summer-intern-special-cool-ranking-profile.run