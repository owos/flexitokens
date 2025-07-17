#!/usr/bin/env python3
# Compression rate evaluation script for BPE tokenizers on FLORES dataset
# run with:
# python src/eval/bpe_compression_rate.py --tokenizer_path /path/to/tokenizer --output_dir /path/to/output
import os
import argparse
import json
from datetime import datetime
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np

# FLORES language mapping
flores_language_mapping = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "uk": "ukr_Cyrl",
    "ru": "rus_Cyrl",
    "be": "bel_Cyrl",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "te": "tel_Telu",
}

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate compression rate of a BPE tokenizer on FLORES dataset')
    parser.add_argument('--tokenizer_path', type=str, default="data/bpe_tokenizer_31000",
                        help='Path to the tokenizer')
    parser.add_argument('--output_dir', type=str, default="results",
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--cache_dir', type=str, default="~/owos/.cache", help='Cache directory')
    return parser.parse_args()

def load_flores_dataset(target_languages):
    """Load FLORES dataset for the specified languages."""
    dataset_dict = DatasetDict()
    for lang in target_languages:
        dataset = load_dataset("facebook/flores", flores_language_mapping[lang], split="devtest")
        dataset_dict[lang] = dataset
    return dataset_dict

def calculate_compression_rate(tokenizer, dataset_dict):
    """Calculate compression rate for each language."""
    results = {}
    
    for lang, dataset in tqdm(dataset_dict.items(), desc="Processing languages"):
        compression_rates = [] 
        token_to_byte_rates = []  
        token_counts = []
        byte_counts = []
        
        for example in tqdm(dataset, desc=f"Processing {lang} sentences", leave=False):
            sentence = example["sentence"]
            encoded_tokens = tokenizer.encode(sentence)
            byte_length = len(sentence.encode('utf-8'))
            # Skip empty sentences or those that don't tokenize
            if byte_length == 0 or len(encoded_tokens) == 0:
                raise ValueError(f"Empty sentence or tokenization error for: {sentence}")
                
            sentence_bytes_per_token = byte_length / len(encoded_tokens)
            sentence_tokens_per_byte = len(encoded_tokens) / byte_length
            
            compression_rates.append(sentence_bytes_per_token)
            token_to_byte_rates.append(sentence_tokens_per_byte)
            token_counts.append(len(encoded_tokens))
            byte_counts.append(byte_length)
        avg_compression_rate = np.mean(compression_rates) 
        avg_token_to_byte_rate = np.mean(token_to_byte_rates) 
        avg_tokens_per_sentence = np.mean(token_counts)
        avg_bytes_per_sentence = np.mean(byte_counts)
        
        total_tokens = sum(token_counts)
        total_bytes = sum(byte_counts)
        total_compression_rate = total_bytes / total_tokens if total_tokens > 0 else 0
        
        results[lang] = {
            "total_tokens": total_tokens,
            "total_bytes": total_bytes,
            "avg_tokens_per_sentence": avg_tokens_per_sentence,
            "avg_bytes_per_sentence": avg_bytes_per_sentence,
            "avg_bytes_per_token_per_sentence": avg_compression_rate,  # Average of per-sentence rates
            "avg_tokens_per_byte_per_sentence": avg_token_to_byte_rate,  # Average of per-sentence rates
            "total_bytes_per_token": total_compression_rate,  # Overall dataset rate
            "compression_rates": {
                "min": min(compression_rates) if compression_rates else 0,
                "max": max(compression_rates) if compression_rates else 0,
                "median": sorted(compression_rates)[len(compression_rates)//2] if compression_rates else 0
            }
        }
        
        print(f"{lang}: {avg_compression_rate:.4f} bytes/token (per sentence avg), {total_compression_rate:.4f} bytes/token (overall)")
    
    return results

def write_results(results, output_dir, tokenizer_path):
    """Save results to a file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract tokenizer name from path
    tokenizer_name = os.path.basename(tokenizer_path.rstrip('/'))
    
    # Create results file path
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_path = os.path.join(output_dir, f"{tokenizer_name}_compression_rates_{timestamp}.json")
    
    # Save results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {results_path}")

def main():
    args = parse_args()
    target_languages = ["en", "es", "ru", "uk", "hi", "te"]
    
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        cache_dir=args.cache_dir,
        use_fast=True
    )
    print(f"Tokenizer loaded with vocabulary size: {len(tokenizer)}")
    
    print("Loading FLORES dataset...")
    dataset_dict = load_flores_dataset(target_languages)
    
    print("Calculating compression rates...")
    results = calculate_compression_rate(tokenizer, dataset_dict)
    print(results)
    quit()
    
    # Calculate overall average compression rate
    avg_bytes_per_token = sum(lang_stats["bytes_per_token"] for lang_stats in results.values()) / len(results)
    avg_tokens_per_byte = sum(lang_stats["tokens_per_byte"] for lang_stats in results.values()) / len(results)
    
    print(f"\nOverall average: {avg_bytes_per_token:.4f} bytes/token ({avg_tokens_per_byte:.4f} tokens/byte)")
    
    # Save detailed results
    results["summary"] = {
        "avg_bytes_per_token": avg_bytes_per_token,
        "avg_tokens_per_byte": avg_tokens_per_byte
    }
    
    write_results(results, args.output_dir, args.tokenizer_path)

if __name__ == "__main__":
    main()