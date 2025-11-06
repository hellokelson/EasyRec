#!/bin/bash
# Quick test script for Ali CCP example

set -e

echo "=== Ali CCP DeepFM Example Quick Test ==="
echo ""

# Step 1: Check data files
echo "Step 1: Checking data files..."
if [ ! -f "sample_skeleton_train.csv" ]; then
    echo "ERROR: sample_skeleton_train.csv not found!"
    echo "Please download from https://tianchi.aliyun.com/dataset/408"
    exit 1
fi
echo "✓ Data files found"
echo ""

# Step 2: Preprocess data
echo "Step 2: Preprocessing data (100K samples)..."
python preprocess.py
echo "✓ Preprocessing complete"
echo ""

# Step 3: Check preprocessed files
echo "Step 3: Checking preprocessed files..."
if [ -f "ali_ccp_train_simple.csv" ]; then
    train_lines=$(wc -l < ali_ccp_train_simple.csv)
    echo "✓ Training data: $train_lines lines"
fi
if [ -f "ali_ccp_test_simple.csv" ]; then
    test_lines=$(wc -l < ali_ccp_test_simple.csv)
    echo "✓ Test data: $test_lines lines"
fi
echo ""

# Step 4: Show sample data
echo "Step 4: Sample data preview:"
echo "--- Training data (first 3 lines) ---"
head -3 ali_ccp_train_simple.csv
echo ""

# Step 5: Ready to train
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Train model:"
echo "   cd /home/zhangkap/sourcecode/EasyRec"
echo "   python -m easy_rec.python.train_eval \\"
echo "     --pipeline_config_path examples/configs/deepfm_on_ali_ccp.config"
echo ""
echo "2. Evaluate model:"
echo "   python -m easy_rec.python.eval \\"
echo "     --pipeline_config_path examples/configs/deepfm_on_ali_ccp.config"
echo ""
echo "3. Export model:"
echo "   python -m easy_rec.python.export \\"
echo "     --pipeline_config_path examples/configs/deepfm_on_ali_ccp.config \\"
echo "     --export_dir examples/ckpt/deepfm_ali_ccp_export"
