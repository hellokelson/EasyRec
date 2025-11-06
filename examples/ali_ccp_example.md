# Alibaba CCP Dataset Example with DeepFM

## Overview

This example demonstrates how to train a DeepFM model on the Alibaba Click Conversion Prediction (CCP) dataset from Tianchi.

**Dataset**: [Tianchi Alibaba CCP Dataset](https://tianchi.aliyun.com/dataset/408)

**Task**: Binary classification (conversion prediction)

**Model**: DeepFM

**Dataset Size**: 
- Training: 42.3M samples
- Test: 43M samples
- Features: user_id, item_id (simplified version)

## Prerequisites

1. Download the dataset from Tianchi and place files in `examples/data/ali_ccp/`:
   - `sample_skeleton_train.csv` (42.3M samples)
   - `sample_skeleton_test.csv` (43M samples)
   - `common_features_train.csv` (optional)
   - `common_features_test.csv` (optional)

2. Install EasyRec (see main README)

## Step-by-Step Guide

### Quick Start (Recommended)

For first-time users, use the automated setup script:

```bash
cd examples/data/ali_ccp
bash quick_test.sh
```

This will check data files, preprocess data, and show you the next steps.

### Data Preprocessing Options

The preprocessing script supports three dataset sizes:

#### Option 1: Small Dataset (Quick Testing)
```bash
cd examples/data/ali_ccp
python preprocess.py small
# or simply: python preprocess.py
```
- **Training**: 100K samples → `ali_ccp_train_small.csv`
- **Test**: 10K samples → `ali_ccp_test_small.csv`
- **Training time**: ~5 minutes on CPU
- **Use case**: Quick testing and development

#### Option 2: Medium Dataset (Balanced Performance)
```bash
cd examples/data/ali_ccp
python preprocess.py medium
```
- **Training**: 1M samples → `ali_ccp_train_medium.csv`
- **Test**: 100K samples → `ali_ccp_test_medium.csv`
- **Training time**: ~30-60 minutes on CPU
- **Use case**: Better model performance evaluation

#### Option 3: Full Dataset (Production Scale)
```bash
cd examples/data/ali_ccp
python preprocess.py full
```
- **Training**: 42.3M samples → `ali_ccp_train_full.csv`
- **Test**: 43M samples → `ali_ccp_test_full.csv`
- **Training time**: Several hours (recommend GPU/distributed training)
- **Use case**: Production-level model training

### Data Format After Preprocessing

All preprocessing options create the same simplified format:
```csv
label,user_id,item_id
0,bacff91692951881,9
1,bacff91692951881,10
0,bacff91692951881,20
```

**Label Distribution** (conversion rate ~4.4%):
- `0`: No conversion (~95.6%)
- `1`: Conversion (~4.4%)

### Model Training

#### 1. Update Config for Your Dataset Size

**For Small Dataset (default):**
```bash
# Config already points to small dataset
# examples/configs/deepfm_on_ali_ccp.config uses ali_ccp_train_correct.csv
```

**For Medium Dataset:**
```bash
# Update config file paths:
train_input_path: "examples/data/ali_ccp/ali_ccp_train_medium.csv"
eval_input_path: "examples/data/ali_ccp/ali_ccp_test_medium.csv"
num_steps: 20000  # Increase training steps
```

**For Full Dataset:**
```bash
# Update config file paths:
train_input_path: "examples/data/ali_ccp/ali_ccp_train_full.csv"
eval_input_path: "examples/data/ali_ccp/ali_ccp_test_full.csv"
num_steps: 100000  # Increase training steps significantly
batch_size: 4096   # Increase batch size for efficiency
```

#### 2. Train the Model

**CPU Training:**
```bash
cd /path/to/EasyRec
python -m easy_rec.python.train_eval \
  --pipeline_config_path examples/configs/deepfm_on_ali_ccp.config
```

**GPU Training:**
```bash
CUDA_VISIBLE_DEVICES=0 python -m easy_rec.python.train_eval \
  --pipeline_config_path examples/configs/deepfm_on_ali_ccp.config
```

**Distributed Training (for full dataset):**
```bash
# Use multiple GPUs or distributed setup
bash scripts/train_2gpu.sh examples/configs/deepfm_on_ali_ccp.config
```

### Model Evaluation

```bash
python -m easy_rec.python.eval \
  --pipeline_config_path examples/configs/deepfm_on_ali_ccp.config
```

### Model Export

```bash
python -m easy_rec.python.export \
  --pipeline_config_path examples/configs/deepfm_on_ali_ccp.config \
  --export_dir examples/ckpt/deepfm_ali_ccp_export
```

## Configuration Details

The config file `examples/configs/deepfm_on_ali_ccp.config` includes:

**Features:**
- `user_id`: String ID feature with hash bucketing (100K buckets)
- `item_id`: String ID feature with hash bucketing (100K buckets)
- Both embedded to 16 dimensions

**Model Architecture:**
- DeepFM with FM component + Deep component
- DNN layers: [128, 64, 32]
- Final DNN: [64, 32]
- L2 regularization: 1e-5

**Training Configuration:**
- Batch size: 1024 (increase for full dataset)
- Optimizer: Adam with exponential decay
- Initial learning rate: 0.001
- Steps: 5000 (increase for larger datasets)

## Expected Results

### Performance by Dataset Size

| Dataset Size | Training Samples | Expected AUC | Training Time (CPU) |
|--------------|------------------|--------------|-------------------|
| Small        | 100K            | 0.55-0.65    | ~5 minutes        |
| Medium       | 1M              | 0.65-0.75    | ~30-60 minutes    |
| Full         | 42.3M           | 0.75-0.85    | Several hours     |

### Sample Training Output
```
[INFO] global step 1000: lr = 0.001, cross_entropy_loss = 0.45, total_loss = 0.46
[INFO] auc = 0.68, global_step = 1000, loss = 0.45
```

## Advanced Usage

### Adding More Features

The `common_features_*.csv` files contain additional sparse features. To use them:

1. Modify `preprocess.py` to parse sparse features
2. Add feature definitions to the config
3. Use `MultiValueFeature` type for multi-value features

### Handling Class Imbalance

For the imbalanced dataset (95.6% vs 4.4%), consider:

1. **Weighted Loss:**
```protobuf
loss {
  classification_loss {
    weighted_cross_entropy_loss {
      pos_weight: 21.7  # 95.6/4.4 ratio
    }
  }
}
```

2. **Focal Loss:**
```protobuf
loss {
  classification_loss {
    focal_loss {
      alpha: 0.25
      gamma: 2.0
    }
  }
}
```

### Performance Optimization

**For Large Datasets:**
- Increase `batch_size` to 4096 or 8192
- Use multiple GPUs with distributed training
- Enable mixed precision training
- Increase `prefetch_size` and `num_parallel_calls`

**Memory Optimization:**
- Reduce `hash_bucket_size` if memory limited
- Use gradient checkpointing
- Reduce embedding dimensions

## Troubleshooting

**Out of Memory:**
- Reduce `batch_size` in config
- Reduce `hash_bucket_size`
- Use smaller dataset size

**Slow Training:**
- Increase `batch_size`
- Use GPU: `CUDA_VISIBLE_DEVICES=0`
- Use distributed training for full dataset

**Low AUC:**
- Increase training steps
- Add more features from `common_features_*.csv`
- Tune hyperparameters (learning rate, DNN layers)
- Handle class imbalance with weighted loss

**Data Processing Issues:**
- Ensure CSV files have proper format
- Check `with_header: true` in config
- Verify label distribution with preprocessing output

## File Structure

```
examples/data/ali_ccp/
├── sample_skeleton_train.csv      # Original training data (42.3M)
├── sample_skeleton_test.csv       # Original test data (43M)  
├── preprocess.py                  # Preprocessing script
├── quick_test.sh                  # Quick setup script
├── ali_ccp_train_small.csv       # Small dataset (100K)
├── ali_ccp_test_small.csv        # Small test (10K)
├── ali_ccp_train_medium.csv      # Medium dataset (1M)
├── ali_ccp_test_medium.csv       # Medium test (100K)
├── ali_ccp_train_full.csv        # Full dataset (42.3M)
└── ali_ccp_test_full.csv         # Full test (43M)
```

## References

- [EasyRec DeepFM Documentation](../../docs/source/models/deepfm.md)
- [Tianchi CCP Dataset](https://tianchi.aliyun.com/dataset/408)
- [DeepFM Paper](https://arxiv.org/abs/1703.04247)
