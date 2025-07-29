#!/bin/bash

# テストするハイパーパラメータの組み合わせ
LEARNING_RATES=(0.01 0.001 0.0001)
IMAGE_SIZES=(64 128 224)

# Singularityコンテナ名
SIF_FILE="densenet_env.sif"

# 結果を保存するベースディレクトリ
OUTPUT_DIR="/mnt/dense_all_results"

# 各組み合わせで学習を実行
for lr in "${LEARNING_RATES[@]}"
do
    for size in "${IMAGE_SIZES[@]}"
    do
        echo "----------------------------------------------------"
        echo "Running with: Learning Rate = $lr, Image Size = $size"
        echo "----------------------------------------------------"

        singularity exec --nv -B $PWD:/mnt \
            $SIF_FILE \
            python3 /mnt/hyperparameter_tuning.py \
                --learning-rate $lr \
                --image-size $size \
                --output-dir $OUTPUT_DIR

        echo "Finished run with LR=$lr, Size=$size"
        echo ""
    done
done

_EOF_
