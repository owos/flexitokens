#!/usr/bin/env python3
# Byte-to-byte ratio evaluation script for FLORES dataset
# run with:
# python src/eval/byte_to_byte_ratio.py --target_languages "en,es,ru,uk,hi,te"
import argparse
from datasets import load_dataset, DatasetDict, concatenate_datasets
import numpy as np

# FLORES language mapping
FLORES_LANGUAGE_MAPPING = {
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

# Define argument parser
def pargs():
    parser = argparse.ArgumentParser(description="Compute byte-to-byte ratios for FLORES dataset")
    parser.add_argument(
        "--target_languages", 
        type=str, 
        default="en,es,fr,uk,ru,be,hi,bn,te", 
        help="Comma-separated list of target languages"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size for processing the dataset"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    return args

# Load FLORES dataset for specified languages
def load_flores_dataset(target_languages):
    dataset_dict = DatasetDict()
    for lang in target_languages:
        dataset = load_dataset("facebook/flores", FLORES_LANGUAGE_MAPPING[lang], split="dev")
        dataset_dict[lang] = dataset
    return dataset_dict

def load_combined_flores_dataset(target_languages):
    dataset_dict = DatasetDict()
    for lang in target_languages:
        dev = load_dataset("facebook/flores", FLORES_LANGUAGE_MAPPING[lang], split="dev")
        devtest = load_dataset("facebook/flores", FLORES_LANGUAGE_MAPPING[lang], split="devtest")
        combined = concatenate_datasets([dev, devtest])  # Combine dev and devtest splits
        dataset_dict[lang] = combined
    return dataset_dict

# Compute byte-to-byte ratios
def compute_byte_to_byte_ratios(dataset_dict):
    ratios = {}
    ratio_std = {}
    for lang, dataset in dataset_dict.items():
        total_byte_length = []

        for example in dataset["sentence"]:
            byte_length = len(example.encode("utf-8"))
            total_byte_length.append(byte_length)

        average_bytes = np.mean(total_byte_length)
        average_std = np.std(total_byte_length)
        ratios[lang] = average_bytes
        ratio_std[lang] = average_std
    return ratios, ratio_std



# Main execution
if __name__ == "__main__":
    args = pargs()
    target_languages = args.target_languages.split(",")
    scaling_factor = 10
    
    # Load dataset
    dataset = load_combined_flores_dataset(target_languages)
    
    # Compute byte-to-byte ratios
    byte_to_byte_ratios, byte_to_byte_std  = compute_byte_to_byte_ratios(dataset)
     # Print results
    print("Byte-to-Byte Ratios by Language:")
    for lang, ratio in byte_to_byte_ratios.items():
        print(f"{lang}: {ratio:.2f}")
    
    print("Byte-to-Byte Ratios Standard Deviation by Language:")
    for lang, ratio in byte_to_byte_std.items():
        print(f"{lang}: {ratio:.2f}")
    # conver ratio to to real ratio by dividing by the lowest and multiplying by the scaling factor. Finally, round to 2 decimal places and print as before
    
    # Convert ratio to real ratio
    lowest_ratio = min(byte_to_byte_ratios.values())
    lowest_std = min(byte_to_byte_std.values())
    priors = {lang: 1/((ratio / lowest_ratio) * scaling_factor) for lang, ratio in byte_to_byte_ratios.items()}
    priors_std = {lang: 1/(ratio) * scaling_factor for lang, ratio in byte_to_byte_std.items()}  # scale or no scale?

    # Print results
    print("Prior by Language:")
    for lang, ratio in priors.items():
        print(f"{lang}: {round(ratio, 3)}")
    print("Prior Standard Deviation by Language:")
    for lang, ratio in priors_std.items():
        print(f"{lang}: {round(ratio, 3)}")

    rounded_priors = ",".join(str(round(ratio, 3)) for ratio in priors.values())
    print(rounded_priors)
    rounded_priors_std = [str(round(ratio, 3)) for ratio in priors_std.values()]
    rounded_priors_std_str  = ",".join(rounded_priors_std)
    print(rounded_priors_std_str)
    breakpoint()

en: 0.2
es: 0.165
# fr: 0.162
ru: 0.1
uk: 0.107
# be: 0.096
hi: 0.078
# bn: 0.076
te: 0.074

