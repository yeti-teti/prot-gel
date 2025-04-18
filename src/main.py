import argparse
import os

from pred.model_runner import ModelRunner
from config.config import load_config

def main():
    parser = argparse.ArgumentParser(
        description="Protein Gelation Prediction Model (R2 Version)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )

    subparsers = parser.add_subparsers(dest='mode', help='Mode to run', required=True)

    # --- Train Mode ---
    train_parser = subparsers.add_parser('train', help='Train a new model or resume training',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_parser.add_argument('--config', required=True, help='Path to config YAML file (hyperparameters)')
    train_parser.add_argument('--mean_std_json_path', required=True, help='Path to the mean_std.json statistics file')
    train_parser.add_argument('--r2_env_path', default=".env", help='Path to the .env file with R2 credentials')
    train_parser.add_argument('--output_dir', required=True, help='Output directory for checkpoints and logs')
    train_parser.add_argument('--train_r2_path', required=True, help='R2 path to training dataset directory (e.g., integrated_data/train_split_parquet)')
    train_parser.add_argument('--val_r2_path', required=True, help='R2 path to validation dataset directory (e.g., integrated_data/val_split_parquet)')
    train_parser.add_argument('--model', help='Path to model checkpoint (.ckpt) to resume training from (optional)')
    train_parser.add_argument('--verbosity', default='info', choices=['debug', 'info', 'warning', 'error'], help='Logging verbosity')
    train_parser.add_argument('--r2_main_dataset_path', default=None, help='R2 path to the main (complete) dataset directory (optional, for potential future use)')


    # --- Evaluate Mode ---
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model on a test set',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    eval_parser.add_argument('--config', required=True, help='Path to config YAML file (used for model architecture)')
    eval_parser.add_argument('--mean_std_json_path', required=True, help='Path to the mean_std.json statistics file')
    eval_parser.add_argument('--r2_env_path', default=".env", help='Path to the .env file with R2 credentials')
    eval_parser.add_argument('--output_dir', required=True, help='Directory where evaluation results will be saved') # Output dir needed to save results
    eval_parser.add_argument('--test_r2_path', required=True, help='R2 path to test dataset directory (e.g., integrated_data/test_split_parquet)')
    eval_parser.add_argument('--model', required=True, help='Path to the trained model checkpoint (.ckpt) to evaluate')
    eval_parser.add_argument('--results_file', default="evaluation_results.txt", help='Filename for evaluation results within the output directory')
    eval_parser.add_argument('--verbosity', default='info', choices=['debug', 'info', 'warning', 'error'], help='Logging verbosity')
    eval_parser.add_argument('--r2_main_dataset_path', default=None, help='R2 path to the main (complete) dataset directory (optional, for potential future use)')

    # --- Predict Mode ---
    predict_parser = subparsers.add_parser('predict', help='Generate predictions for a dataset using a trained model',
                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    predict_parser.add_argument('--config', required=True, help='Path to config YAML file (used for model architecture)')
    predict_parser.add_argument('--mean_std_json_path', required=True, help='Path to the mean_std.json statistics file')
    predict_parser.add_argument('--r2_env_path', default=".env", help='Path to the .env file with R2 credentials')
    predict_parser.add_argument('--output_dir', required=True, help='Directory where predictions will be saved')
    predict_parser.add_argument('--predict_r2_path', required=True, help='R2 path to dataset directory for prediction')
    predict_parser.add_argument('--model', required=True, help='Path to the trained model checkpoint (.ckpt) to use for prediction')
    predict_parser.add_argument('--results_file', default="predictions.tsv", help='Filename for predictions within the output directory')
    predict_parser.add_argument('--verbosity', default='info', choices=['debug', 'info', 'warning', 'error'], help='Logging verbosity')
    predict_parser.add_argument('--r2_main_dataset_path', default=None, help='R2 path to the main (complete) dataset directory (optional, for potential future use)')


    args = parser.parse_args()

    try:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    except Exception as e:
        print(f"Error loading configuration file '{args.config}': {e}")
        return

    # Prepare arguments for ModelRunner, common across modes
    runner_args = {
        "config": config,
        "r2_main_dataset_path": args.r2_main_dataset_path, # Pass even if None
        "mean_std_json_path": args.mean_std_json_path,
        "r2_env_path": args.r2_env_path,
        "model_path": getattr(args, 'model', None), # Safely get model path
        "output_dir": args.output_dir, # Changed from output_path
        "verbosity": args.verbosity
    }

    # Add mode-specific R2 paths and run the corresponding method
    try:
        if args.mode == 'train':
            runner_args.update({
                "train_r2_path": args.train_r2_path, 
                "val_r2_path": args.val_r2_path   
            })
            runner = ModelRunner(**runner_args)
            runner.train()
        elif args.mode == 'evaluate':
            runner_args.update({
                "test_r2_path": args.test_r2_path
            })
            runner = ModelRunner(**runner_args)
            runner.evaluate(output_file=args.results_file)
        elif args.mode == 'predict':
             # predict_r2_path is passed directly to the method, not __init__ 
             runner = ModelRunner(**runner_args)
             runner.predict(predict_r2_path=args.predict_r2_path, out_file=args.results_file) # Pass paths here
        else:
            parser.print_help()

    except Exception as e:
        print(f"\nAn error occurred during execution:")
        print(f"Mode: {args.mode}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()