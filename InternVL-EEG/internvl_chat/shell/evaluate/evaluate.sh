#!/usr/bin/env bash
# Evaluate UniMind checkpoints on EEG classification benchmarks.
#
# Usage:
#   bash <this_script> <CHECKPOINT_PATH> <DATASET_NAME> [--auto]
#
# Arguments:
#   CHECKPOINT_PATH  : path to checkpoint directory (contains .safetensors + config.json)
#   DATASET_NAME     : one of SEED | HMC | Workload | TUAB | TUEV | TUSL |
#                              SEEDIV | SHU | SleepEDF | SHHS
#   --auto           : optional flag to force single-GPU inference (overrides GPUS)
#
# Environment variables:
#   GPUS             : number of GPUs (default 1)
#   TEST_CONFIG      : path to test dataset JSON (default shell/data/test_datasets.json)
#   OUT_DIR          : output directory for results (default results/)
#
# Metrics reported: Balanced Accuracy, Cohen's Kappa, Weighted F1

set -euo pipefail
set -x

CHECKPOINT=${1:?"Usage: $0 <CHECKPOINT_PATH> <DATASET_NAME> [--auto]"}
DATASET=${2:?"Usage: $0 <CHECKPOINT_PATH> <DATASET_NAME> [--auto]"}

GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NODES=$((GPUS / GPUS_PER_NODE))

TEST_CONFIG=${TEST_CONFIG:-"shell/data/test_datasets.json"}
OUT_DIR=${OUT_DIR:-"results"}

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export MASTER_PORT=${MASTER_PORT:-63680}

# Save all args so we can forward any extra flags (e.g. --auto) to the Python script.
ARGS=("$@")

# Handle --auto flag: single-GPU inference regardless of GPUS setting.
for arg in "${ARGS[@]:2}"; do
  if [[ "$arg" == "--auto" ]]; then
    GPUS=1
    break
  fi
done

run_eval() {
  local ds_name=$1
  local answer_labels=$2
  local batch_size=${3:-32}

  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa_EEG_batch.py \
    --resample_method ctqformer_v2_2router_merge \
    --num_cquery_tokens 2 \
    --num_tquery_tokens 2 \
    --normalize_type zscore \
    --batch-size "${batch_size}" \
    --checkpoint "${CHECKPOINT}" \
    --test_dataset "${TEST_CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --answer_labels "${answer_labels}" \
    --datasets "${ds_name}" \
    "${ARGS[@]:2}"
}

case "${DATASET}" in
  SEED)
    run_eval SEED "['positive', 'negative', 'neutral']" 32 ;;
  HMC)
    run_eval HMC "['Sleep stage N1', 'Sleep stage N2', 'Sleep stage N3', 'Sleep stage R', 'Sleep stage W']" 48 ;;
  Workload)
    run_eval Workload "['high', 'low']" 24 ;;
  TUAB)
    run_eval TUAB "['normal', 'abnormal']" 48 ;;
  TUEV)
    run_eval TUEV "['Artifact', 'Background', 'Eye Movement', 'Generalized Paroxysmal Epileptiform Discharge', 'Periodic Lateralized Epileptiform Discharge', 'Spikes and Slow Waves']" 48 ;;
  TUSL)
    run_eval TUSL "['bckg', 'seiz', 'slow']" 32 ;;
  SEEDIV)
    run_eval SEEDIV "['neutral', 'sad', 'fear', 'happy']" 32 ;;
  SHU)
    run_eval SHU "['left hand', 'right hand']" 32 ;;
  SleepEDF)
    run_eval SleepEDF "['Sleep stage W', 'Sleep stage N1', 'Sleep stage N2', 'Sleep stage N3', 'Sleep stage R']" 32 ;;
  SHHS)
    run_eval SHHS "['Sleep stage W', 'Sleep stage N1', 'Sleep stage N2', 'Sleep stage N3', 'Sleep stage R']" 64 ;;
  *)
    echo "Unknown dataset: ${DATASET}"
    echo "Valid options: SEED HMC Workload TUAB TUEV TUSL SEEDIV SHU SleepEDF SHHS"
    exit 1 ;;
esac
