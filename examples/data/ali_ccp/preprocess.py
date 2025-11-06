#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preprocess script for Alibaba CCP dataset with flexible data size options."""

import csv
import sys

def parse_sample_skeleton(input_file, output_file, max_samples=None):
    """
    Parse sample_skeleton format to simple CSV.
    Format: sample_id,conversion,label,user_id,item_id,features
    Output: label,user_id,item_id (using conversion as binary label)
    
    Args:
        input_file: Input CSV file path
        output_file: Output CSV file path  
        max_samples: Maximum samples to process (None for all data)
    """
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(['label', 'user_id', 'item_id'])
        
        count = 0
        for line in fin:
            if max_samples and count >= max_samples:
                break
            
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            
            # Use conversion (second column) as binary label
            conversion = parts[1]
            user_id = parts[3]
            item_id = parts[4]
            
            writer.writerow([conversion, user_id, item_id])
            count += 1
            
            if count % 100000 == 0:
                print(f"Processed {count} samples...")
    
    print(f"Done! Processed {count} samples to {output_file}")

def main():
    """Main function with command line argument support."""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'full':
            # Process full dataset (42M+ samples)
            print("Processing FULL dataset (42M+ samples)...")
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
            
        elif mode == 'medium':
            # Process medium dataset (1M samples)
            print("Processing MEDIUM dataset (1M samples)...")
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
            # Process small dataset (100K samples) - default
            print("Processing SMALL dataset (100K samples)...")
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
            print("Usage: python preprocess.py [small|medium|full]")
            sys.exit(1)
    else:
        # Default: small dataset for quick testing
        print("Processing SMALL dataset (100K samples) - default...")
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
