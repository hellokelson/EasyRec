# Alibaba CCP (Click Conversion Prediction) Dataset Example

## Dataset Information

- **Source**: [Tianchi Alibaba CCP Dataset](https://tianchi.aliyun.com/dataset/408)
- **Task**: Click prediction (binary classification)
- **Size**: 42M+ training samples, 43M+ test samples
- **Features**: Sparse multi-value features with user and item IDs

## Data Files

- `sample_skeleton_train.csv`: Training data with click/conversion labels
- `sample_skeleton_test.csv`: Test data
- `common_features_train.csv`: Additional sparse features (optional)
- `common_features_test.csv`: Additional sparse features (optional)

## Quick Start

### Option 1: Automated Setup (Recommended)

Run the quick test script to verify setup and preprocess data:

```bash
cd examples/data/ali_ccp
bash quick_test.sh
```

This script will:
1. Check if data files exist
2. Run preprocessing (100K training samples, 10K test samples)
3. Verify output files
4. Show next steps for training

### Option 2: Manual Setup

#### Step 1: Preprocess Data

The raw data format is complex. We simplify it to basic CSV format:

```bash
cd examples/data/ali_ccp
python preprocess.py
```

This creates:
- `ali_ccp_train_simple.csv` (100K samples for quick testing)
- `ali_ccp_test_simple.csv` (10K samples)

#### Step 2: Train DeepFM Model

```bash
# Return to EasyRec root directory
cd ../../..

# Single GPU training
python -m easy_rec.python.train_eval \
  --pipeline_config_path examples/configs/deepfm_on_ali_ccp.config

# Or use CUDA_VISIBLE_DEVICES to specify GPU
CUDA_VISIBLE_DEVICES=0 python -m easy_rec.python.train_eval \
  --pipeline_config_path examples/configs/deepfm_on_ali_ccp.config
```

#### Step 3: Evaluate Model

```bash
python -m easy_rec.python.eval \
  --pipeline_config_path examples/configs/deepfm_on_ali_ccp.config
```

#### Step 4: Export Model

```bash
python -m easy_rec.python.export \
  --pipeline_config_path examples/configs/deepfm_on_ali_ccp.config \
  --export_dir examples/ckpt/deepfm_ali_ccp_export
```

## Configuration Details

The config file `examples/configs/deepfm_on_ali_ccp.config` includes:

- **Features**: user_id, item_id (hash bucketed)
- **Model**: DeepFM with DNN layers [128, 64, 32]
- **Training**: 5000 steps, batch size 1024
- **Optimizer**: Adam with exponential decay

## Scaling to Full Dataset

To use the full dataset (42M samples), modify `preprocess.py`:

```python
parse_sample_skeleton(
    'sample_skeleton_train.csv',
    'ali_ccp_train_full.csv',
    max_samples=None  # Process all samples
)
```

Then update the config file paths and increase training steps.

## Notes

- The simplified version uses only user_id and item_id features
- For better performance, you can add the sparse features from `common_features_*.csv`
- Adjust `hash_bucket_size` based on unique user/item counts
- Consider using distributed training for the full dataset
