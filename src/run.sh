
# train
python main.py train -o model_run --config pred/config.yaml --verbosity debug

# Train with pretrained model
# python main.py train src/datasets/train_set.mgf -p src/datasets/val_set.mgf -o model_run --model outputs/epoch=21-step=216000.ckpt --config src/config.yaml --verbosity debug

# Evaluate
# python main.py evaluate --model outputs/epoch=0-step=7500.ckpt --config src/config.yaml --output eval_results.txt --verbosity info
# python main.py evaluate src/datasets/val_set.mgf --model outputs/epoch=0-step=7500.ckpt --config src/config.yaml --output eval_results.txt --verbosity info

# Predict
# python main.py sequence --model outputs/epoch=21-step=216000.ckpt --config src/config.yaml --output predictions.mztab --verbosity info
# python main.py sequence src/datasets/unknown_spectra.mgf --model outputs/epoch=21-step=216000.ckpt --config src/config.yaml --output predictions.mztab --verbosity info

