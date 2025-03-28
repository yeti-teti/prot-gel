import argparse
from pred.model_runner import ModelRunner
from pred.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Protein Gelation Prediction Model")
    subparsers = parser.add_subparsers(dest='mode', help='Mode to run the script in')

    # Train mode
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('train_path', nargs='?', default=None, help='Path to training dataset (optional)')
    train_parser.add_argument('-p', '--val_path', default=None, help='Path to validation dataset (optional)')
    train_parser.add_argument('-o', '--output', required=True, help='Output directory for checkpoints')
    train_parser.add_argument('--model', help='Path to model checkpoint to resume training')
    train_parser.add_argument('--config', required=True, help='Path to config YAML file')
    train_parser.add_argument('--verbosity', default='info', choices=['debug', 'info', 'warning', 'error'], help='Logging verbosity')

    # Evaluate mode
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('test_path', nargs='?', default=None, help='Path to test dataset (optional)')
    eval_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--config', required=True, help='Path to config YAML file')
    eval_parser.add_argument('--output', help='Output file for evaluation results')
    eval_parser.add_argument('--verbosity', default='info', choices=['debug', 'info', 'warning', 'error'], help='Logging verbosity')

    # Predict (sequence) mode
    predict_parser = subparsers.add_parser('sequence', help='Predict gelation for new data')
    predict_parser.add_argument('data_path', nargs='?', default=None, help='Path to dataset for prediction (optional)')
    predict_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    predict_parser.add_argument('--config', required=True, help='Path to config YAML file')
    predict_parser.add_argument('--output', help='Output file for predictions')
    predict_parser.add_argument('--verbosity', default='info', choices=['debug', 'info', 'warning', 'error'], help='Logging verbosity')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize and run the model runner
    if args.mode == 'train':
        runner = ModelRunner(
            config=config,
            train_path=args.train_path,
            val_path=args.val_path,
            model_path=args.model,
            output_path=args.output,
            verbosity=args.verbosity
        )
        runner.train()
    elif args.mode == 'evaluate':
        runner = ModelRunner(
            config=config,
            test_path=args.test_path,
            model_path=args.model,
            output_path=args.output,
            verbosity=args.verbosity
        )
        runner.evaluate()
    elif args.mode == 'sequence':
        runner = ModelRunner(
            config=config,
            test_path=args.data_path,
            model_path=args.model,
            output_path=args.output,
            verbosity=args.verbosity
        )
        runner.predict()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()