#!/usr/bin/env python3
import os
import pandas as pd
import base64
import numpy as np
import argparse
from PIL import Image
import io
import json

def encode_image_bytes_to_base64(image_bytes):
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')

def convert_parquet_to_tsv(parquet_path, tsv_path, category):
    """
    Convert VTCBench parquet files to TSV format required by VLMEvalKit.
    
    Args:
        parquet_path (str): Path to input parquet file
        tsv_path (str): Path to output TSV file
        category (str): Category name (Retrieval, Memory, or Reasoning)
    """
    # Read parquet file
    df = pd.read_parquet(parquet_path)
    
    # Create new dataframe with required columns
    new_df = pd.DataFrame()
    
    # Required columns according to VLMEvalKit Development.md
    new_df['index'] = range(len(df))
    new_df['image'] = ''  # Will fill this later
    new_df['question'] = df['problem']
    # import pdb; pdb.set_trace()
    # Process answers, converting lists to JSON strings
    new_df['answer'] = df['answers'].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    
    # Add category
    new_df['category'] = category
    # import ipdb; ipdb.set_trace()
    # Process images and encode to base64
    base64_images = []
    for idx, row in df.iterrows():
        images = row['images']
        # Process multiple images as a list of base64 strings
        if len(images) > 0:
            # Create a list of base64 encoded images
            image_list = []
            for img_dict in images:
                img_bytes = img_dict['bytes']
                base64_img = encode_image_bytes_to_base64(img_bytes)
                image_list.append(base64_img)
            
            # If only one image, store it directly; otherwise store as JSON list
            if len(image_list) == 1:
                base64_images.append(image_list[0])
            else:
                base64_images.append(json.dumps(image_list))
        else:
            base64_images.append('')
    
    new_df['image'] = base64_images
    
    # Save to TSV
    new_df.to_csv(tsv_path, sep='\t', index=False)
    print(f"Converted {len(new_df)} samples from {parquet_path} to {tsv_path}")


def combine_all_categories(output_path):
    """
    Combine all three category TSV files into one.
    
    Args:
        output_path (str): Path to output combined TSV file
    """
    categories = ['Retrieval', 'Memory', 'Reasoning']
    dataframes = []
    
    for category in categories:
        tsv_file = f'/hdddata/zhaohongbo/VLMEvalKit/vtcbench/{category.lower()}.tsv'
        if os.path.exists(tsv_file):
            df = pd.read_csv(tsv_file, sep='\t')
            dataframes.append(df)
        else:
            print(f"Warning: {tsv_file} not found")
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.to_csv(output_path, sep='\t', index=False)
        print(f"Combined {len(dataframes)} categories into {output_path}")
    else:
        print("No category files found to combine")


def main():
    parser = argparse.ArgumentParser(description='Convert VTCBench parquet files to TSV format')
    parser.add_argument('--category', choices=['retrieval', 'memory', 'reasoning', 'all'], 
                        default='all', help='Category to convert')
    parser.add_argument('--output-dir', default='/hdddata/zhaohongbo/VLMEvalKit/vtcbench', 
                        help='Output directory for TSV files')
    
    args = parser.parse_args()
    
    if args.category == 'all':
        # Convert all categories
        categories = ['retrieval', 'memory', 'reasoning']
        # categories = ['debug_combined_samples']
        for cat in categories:
            parquet_path = f'/hdddata/zhaohongbo/VLMEvalKit/vtcbench/{cat}.parquet'
            tsv_path = f'/hdddata/zhaohongbo/VLMEvalKit/vtcbench/{cat}.tsv'
            if os.path.exists(parquet_path):
                convert_parquet_to_tsv(parquet_path, tsv_path, cat.capitalize())
            else:
                print(f"Warning: {parquet_path} not found")
        
        # Combine all into one file
        combined_path = '/hdddata/zhaohongbo/VLMEvalKit/vtcbench/vtcbench.tsv'
        combine_all_categories(combined_path)
    else:
        # Convert specific category
        parquet_path = f'/hdddata/zhaohongbo/VLMEvalKit/vtcbench/{args.category}.parquet'
        tsv_path = f'/hdddata/zhaohongbo/VLMEvalKit/vtcbench/{args.category}.tsv'
        if os.path.exists(parquet_path):
            convert_parquet_to_tsv(parquet_path, tsv_path, args.category.capitalize())
        else:
            print(f"Warning: {parquet_path} not found")


if __name__ == '__main__':
    main()