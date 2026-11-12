#!/bin/bash

echo "============================================================"
echo "Training Deep Ensemble (5 Models)"
echo "============================================================"
echo ""
echo "This will train 5 independent models with different seeds"
echo "Estimated time: 40-50 GPU hours total"
echo ""

DATASET=$1
NUM_CLASSES=$2
BATCH_SIZE=$3
MAX_LENGTH=$4

if [ -z "$DATASET" ]; then
    echo "Usage: ./train_deep_ensemble.sh <dataset> <num_classes> <batch_size> <max_length>"
    echo "Example: ./train_deep_ensemble.sh squad 2 8 384"
    exit 1
fi

for i in {1..5}; do
    echo ""
    echo "============================================================"
    echo "Training Ensemble Model $i/5"
    echo "============================================================"
    
    python scripts/train/train_baselines.py \
        --baseline vanilla \
        --dataset $DATASET \
        --num_classes $NUM_CLASSES \
        --epochs 3 \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_LENGTH \
        --seed $((42 + i)) \
        --output_dir experiments/baselines/${DATASET}/ensemble_${i}
    
    if [ $? -ne 0 ]; then
        echo "Error training model $i"
        exit 1
    fi
    
    echo "✓ Model $i completed"
done

echo ""
echo "============================================================"
echo "ALL 5 ENSEMBLE MODELS TRAINED!"
echo "============================================================"