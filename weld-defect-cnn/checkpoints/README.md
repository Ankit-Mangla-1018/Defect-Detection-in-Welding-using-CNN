# Checkpoints

Model weights are excluded from version control (see `.gitignore`).

To generate `best_model.pt`:
```bash
python scripts/download_data.py   # fetch dataset
python scripts/train.py --config configs/baseline.yaml
```

The checkpoint will be saved here automatically.
