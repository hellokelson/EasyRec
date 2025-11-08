#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preprocess script for Alibaba CCP dataset with flexible data size options."""

import csv
import sys
from collections import defaultdict

def parse_features(feature_str):
    """Parse top sparse features from feature string."""
    features = {}
    if not feature_str:
        return features
    
    # Split by \u0001 and extract feature IDs
    parts = feature_str.split('\u0001')
    for part in parts:
        if '\u0002' in part and '\u0003' in part:
            try:
                feature_id, rest = part.split('\u0002', 1)
                feature_value, weight = rest.split('\u0003', 1)
                features[feature_id] = feature_value
            except:
                continue
    return features

def parse_sample_skeleton(input_file, output_file, max_samples=None):
    """
    Parse sample_skeleton format to enhanced CSV with top features.
    Format: sample_id,conversion,label,user_id,item_id,features
    Output: label,user_id,item_id,f1,f2,f3,f4,f5 (top 5 features)
    """
    
    # First pass: find most frequent features
    feature_counts = defaultdict(int)
    print("Analyzing features...")
    
    with open(input_file, 'r') as fin:
        count = 0
        for line in fin:
            if count >= 100000:  # Sample for analysis
                break
            parts = line.strip().split(',')
            if len(parts) >= 6:
                features = parse_features(parts[5])
                for feat_id in features:
                    feature_counts[feat_id] += 1
            count += 1
    
    # Select top 5 features
    top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    selected_features = [feat[0] for feat in top_features]
    print(f"Selected features: {selected_features}")
    
    # Second pass: generate enhanced data
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        writer = csv.writer(fout)
        header = ['label', 'user_id', 'item_id'] + [f'f_{f}' for f in selected_features]
        writer.writerow(header)
        
        count = 0
        for line in fin:
            if max_samples and count >= max_samples:
                break
            
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            
            conversion = parts[1]
            user_id = parts[3]
            item_id = parts[4]
            
            # Extract selected features
            features = parse_features(parts[5] if len(parts) > 5 else "")
            
            row = [conversion, user_id, item_id]
            for feat_id in selected_features:
                row.append(features.get(feat_id, "0"))
            
            writer.writerow(row)
            count += 1
            
            if count % 100000 == 0:
                print(f"Processed {count} samples...")
    
    print(f"Done! Processed {count} samples with {len(selected_features)} features")

def main():
    """Main function with command line argument support."""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'full':
            print("Processing FULL dataset with enhanced features...")
            parse_sample_skeleton(
                'sample_skeleton_train.csv',
                'ali_ccp_train_full.csv',
                max_samples=None
            )
            parse_sample_skeleton(
                'sample_skeleton_test.csv', 
                'ali_ccp_test_full.csv',
                max_samples=None
            )
            
        elif mode == 'large':
            print("Processing LARGE dataset with enhanced features...")
            parse_sample_skeleton(
                'sample_skeleton_train.csv',
                'ali_ccp_train_large.csv',
                max_samples=5000000
            )
            parse_sample_skeleton(
                'sample_skeleton_test.csv',
                'ali_ccp_test_large.csv', 
                max_samples=500000
            )
            
        elif mode == 'medium':
            print("Processing MEDIUM dataset with enhanced features...")
            parse_sample_skeleton(
                'sample_skeleton_train.csv',
                'ali_ccp_train_medium.csv',
                max_samples=1000000
            )
            parse_sample_skeleton(
                'sample_skeleton_test.csv',
                'ali_ccp_test_medium.csv', 
                max_samples=100000
            )
            
        elif mode == 'small':
            print("Processing SMALL dataset with enhanced features...")
            parse_sample_skeleton(
                'sample_skeleton_train.csv',
                'ali_ccp_train_small.csv',
                max_samples=100000
            )
            parse_sample_skeleton(
                'sample_skeleton_test.csv',
                'ali_ccp_test_small.csv',
                max_samples=10000
            )
        else:
            print("Usage: python preprocess.py [small|medium|large|full]")
            sys.exit(1)
    else:
        print("Processing SMALL dataset with enhanced features...")
        parse_sample_skeleton(
            'sample_skeleton_train.csv',
            'ali_ccp_train_small.csv',
            max_samples=100000
        )
        parse_sample_skeleton(
            'sample_skeleton_test.csv',
            'ali_ccp_test_small.csv',
            max_samples=10000
        )

if __name__ == '__main__':
    main()
