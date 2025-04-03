#!/bin/bash

# --- Configuration --- #
# Adjust these paths according to your project structure and R2 bucket layout

# Python executable/environment activation
# Example: source /path/to/your/venv/bin/activate
PYTHON_EXEC="python" # Or path to specific python executable

# Main Python script
MAIN_SCRIPT="main.py"

# Common Paths
CONFIG_FILE="config/config.yaml" # Path to your model hyperparameter config
STATS_FILE="../data/mean_std.json"             # Path to the generated statistics file
R2_ENV_FILE=".env"                     # Path to your R2 credentials file

# R2 Dataset Paths (Relative to R2 Bucket Root)
# These MUST match the output paths from db_writer_cloud.py and data_split.py
R2_TRAIN_DATA="integrated_data/train_split_parquet"
R2_VAL_DATA="integrated_data/val_split_parquet"
R2_TEST_DATA="integrated_data/test_split_parquet"
R2_PREDICT_INPUT_DATA="integrated_data/predict_input_parquet"
R2_MAIN_DATA="integrated_data/viridiplantae_dataset_partitioned_from_json"

# Local Output Directories/Files
BASE_OUTPUT_DIR="results"
TRAIN_OUTPUT_DIR="$BASE_OUTPUT_DIR/training_run_$(date +%Y%m%d_%H%M%S)" # Unique dir per training run
EVAL_OUTPUT_DIR="$BASE_OUTPUT_DIR/evaluation"
PRED_OUTPUT_DIR="$BASE_OUTPUT_DIR/predictions"

# Checkpoint Paths
RESUME_CHECKPOINT="$BASE_OUTPUT_DIR/training_run_YYYYMMDD_HHMMSS/epoch=X-step=Y.ckpt" # Path to resume from
EVAL_MODEL_CHECKPOINT="$BASE_OUTPUT_DIR/training_run_YYYYMMDD_HHMMSS/epoch=X-step=Y.ckpt" # Path to evaluate
PREDICT_MODEL_CHECKPOINT="$BASE_OUTPUT_DIR/training_run_YYYYMMDD_HHMMSS/epoch=X-step=Y.ckpt" # Path to predict with


# --- Helper Function --- #
usage() {
  echo "Usage: $0 {train|resume|evaluate|predict}"
  echo ""
  echo "Modes:"
  echo "  train       Train a new model from scratch."
  echo "  resume      Resume training from a checkpoint."
  echo "  evaluate    Evaluate a trained model on the test set."
  echo "  predict     Generate predictions for a new dataset."
  echo ""
  echo "Note: Adjust paths and variables inside the script before running."
  exit 1
}

# --- Mode Execution --- #

MODE=$1

# Create base output directory if it doesn't exist
mkdir -p "$BASE_OUTPUT_DIR"

echo "-----------------------------------------"
echo "Mode: $MODE"
echo "Config: $CONFIG_FILE"
echo "Stats File: $STATS_FILE"
echo "R2 Env File: $R2_ENV_FILE"
echo "-----------------------------------------"

if [[ "$MODE" == "train" ]]; then
  echo "Starting NEW training run..."
  mkdir -p "$TRAIN_OUTPUT_DIR" # Create unique dir for this run
  echo "Output Directory: $TRAIN_OUTPUT_DIR"

  $PYTHON_EXEC $MAIN_SCRIPT train \
    --config "$CONFIG_FILE" \
    --mean_std_json_path "$STATS_FILE" \
    --r2_env_path "$R2_ENV_FILE" \
    --output_dir "$TRAIN_OUTPUT_DIR" \
    --train_r2_path "$R2_TRAIN_DATA" \
    --val_r2_path "$R2_TEST_DATA" \
    --verbosity info

elif [[ "$MODE" == "resume" ]]; then
  echo "Resuming training from checkpoint: $RESUME_CHECKPOINT"
  # Resuming usually saves to the same directory structure or a new one
  mkdir -p "$TRAIN_OUTPUT_DIR" # Create unique dir for this *resumed* run session
  echo "Output Directory (Resumed): $TRAIN_OUTPUT_DIR"

  if [ ! -f "$RESUME_CHECKPOINT" ]; then
      echo "Error: Checkpoint file not found at $RESUME_CHECKPOINT"
      exit 1
  fi

  $PYTHON_EXEC $MAIN_SCRIPT train \
    --config "$CONFIG_FILE" \
    --mean_std_json_path "$STATS_FILE" \
    --r2_env_path "$R2_ENV_FILE" \
    --output_dir "$TRAIN_OUTPUT_DIR" \
    --train_r2_path "$R2_TRAIN_DATA" \
    --val_r2_path "$R2_VAL_DATA" \
    --model "$RESUME_CHECKPOINT" \
    --verbosity info

elif [[ "$MODE" == "evaluate" ]]; then
  echo "Evaluating model: $EVAL_MODEL_CHECKPOINT"
  mkdir -p "$EVAL_OUTPUT_DIR"
  echo "Evaluation Output Directory: $EVAL_OUTPUT_DIR"
  EVAL_RESULTS_FILE="eval_results_$(basename ${EVAL_MODEL_CHECKPOINT%.*}).txt" # Unique results filename

  if [ ! -f "$EVAL_MODEL_CHECKPOINT" ]; then
      echo "Error: Model checkpoint file not found at $EVAL_MODEL_CHECKPOINT"
      exit 1
  fi

  $PYTHON_EXEC $MAIN_SCRIPT evaluate \
    --config "$CONFIG_FILE" \
    --mean_std_json_path "$STATS_FILE" \
    --r2_env_path "$R2_ENV_FILE" \
    --output_dir "$EVAL_OUTPUT_DIR" \
    --test_r2_path "$R2_TEST_DATA" \
    --model "$EVAL_MODEL_CHECKPOINT" \
    --results_file "$EVAL_RESULTS_FILE" \
    --verbosity info

elif [[ "$MODE" == "predict" ]]; then
  echo "Generating predictions using model: $PREDICT_MODEL_CHECKPOINT"
  mkdir -p "$PRED_OUTPUT_DIR"
  echo "Prediction Output Directory: $PRED_OUTPUT_DIR"
  PRED_RESULTS_FILE="preds_on_$(basename ${R2_PREDICT_INPUT_DATA})_by_$(basename ${PREDICT_MODEL_CHECKPOINT%.*}).tsv"

  if [ ! -f "$PREDICT_MODEL_CHECKPOINT" ]; then
      echo "Error: Model checkpoint file not found at $PREDICT_MODEL_CHECKPOINT"
      exit 1
  fi

  $PYTHON_EXEC $MAIN_SCRIPT predict \
    --config "$CONFIG_FILE" \
    --mean_std_json_path "$STATS_FILE" \
    --r2_env_path "$R2_ENV_FILE" \
    --output_dir "$PRED_OUTPUT_DIR" \
    --predict_r2_path "$R2_PREDICT_INPUT_DATA" \
    --model "$PREDICT_MODEL_CHECKPOINT" \
    --results_file "$PRED_RESULTS_FILE" \
    --verbosity info

else
  echo "Error: Invalid mode specified."
  usage
fi

echo "-----------------------------------------"
echo "Script finished for mode: $MODE"
echo "-----------------------------------------"
exit 0